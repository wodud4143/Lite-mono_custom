from __future__ import absolute_import, division, print_function
import os
import json
from pathlib import Path
import time
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import os.path as osp
import tensorrt as trt
import random

# GA
from deap import base, creator, tools

# 프로젝트 의존 (경로에 있어야 함)
from layers import disp_to_depth
from utils import readlines
from options import LiteMonoOptions
import datasets

cv2.setNumThreads(0)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# ==============================
# TensorRT runner
# ==============================
class TRTDepthEngine:
    def __init__(self, engine_path: str, prefer_async_v3: bool = False):
        assert os.path.exists(engine_path), f"엔진 파일을 찾을 수 없습니다: {engine_path}"
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_binding, self.output_binding = None, None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_binding = name
            else:
                self.output_binding = name
        assert self.input_binding and self.output_binding, "엔진 입/출력 바인딩 점검"
        self.prefer_async_v3 = prefer_async_v3 and hasattr(self.context, 'execute_async_v3')

    def _ensure_shape(self, x: torch.Tensor):
        shape = tuple(x.shape)
        if tuple(self.context.get_tensor_shape(self.input_binding)) != shape:
            self.context.set_input_shape(self.input_binding, shape)
        return shape

    @torch.inference_mode()
    def infer(self, x_np: np.ndarray) -> np.ndarray:
        assert x_np.ndim == 4, f"입력은 (N,C,H,W) 여야 합니다. got {x_np.shape}"
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        self._ensure_shape(x)
        out_shape = tuple(self.context.get_tensor_shape(self.output_binding))
        if any(d < 0 for d in out_shape):
            out_shape = (x.shape[0], 1, x.shape[2], x.shape[3])
        y = torch.empty(out_shape, dtype=torch.float32, device=device)
        self.context.set_tensor_address(self.input_binding, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_binding, int(y.data_ptr()))
        ok = self.context.execute_async_v3(0) if self.prefer_async_v3 else self.context.execute_v2()
        if not ok:
            raise RuntimeError("TensorRT 실행 실패")
        return y.detach().float().cpu().numpy()

# ==============================
# Metrics
# ==============================
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def batch_post_process_disparity(l_disp, r_disp):
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

# ==============================
# 좌/우 분할 + 감마 보정 전처리 (GA 대상)
# ==============================
def apply_clahe_rgb(img, clip_limit=3.0, tile=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(max(0.1, clip_limit)), tileGridSize=(int(tile), int(tile)))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

def adjust_gamma(image, gamma=1.0):
    gamma = float(max(1e-3, gamma))
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255.0 for i in range(256)], dtype=np.float32)
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)

def brightness_enhancement(img,
                           clip_limit=3.0,
                           brightness_threshold=8.0,
                           suppress_gamma=0.7,
                           gamma_boost_low=1.5,
                           gamma_boost_med=1.2,
                           gb_low_thresh=50.0,
                           gb_high_thresh=180.0,
                           clahe_tile=8,
                           region_overbright=200.0,
                           center_ratio=0.2):
    """
    좌/우 분할 + 감마 보정 버전
    - 전역 과밝음 억제(suppress_gamma)
    - 전역 저조도면 CLAHE 강도 증가(clip_limit + boost)
    - 중앙 타겟 밝기에 맞춰 좌/우를 개별 감마 보정
    - center_ratio: 중앙 영역 비율 (0~1, 예: 0.2 = 20%)
    """
    # 입력 방어
    if img is None or img.size == 0:
        return img
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if h < 4 or w < 4:
        return img

    # 타겟 밝기 (중앙 영역 - center_ratio로 조절)
    center_ratio = float(np.clip(center_ratio, 0.05, 0.5))  # 5%~50% 범위로 제한
    slice_width = max(int(w * center_ratio / 2), 1)
    c1, c2 = max(0, w // 2 - slice_width), min(w, w // 2 + slice_width)
    center_slice = img[:, c1:c2]
    target_brightness = float(np.mean(cv2.cvtColor(center_slice, cv2.COLOR_BGR2GRAY)))

    # 전역 밝기
    global_brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

    # 1) 전체 과밝음 억제
    if global_brightness > gb_high_thresh:
        img = adjust_gamma(img, gamma=float(suppress_gamma))

    # 2) CLAHE (저조도일수록 강하게)
    clahe_limit = float(clip_limit + 1.5) if global_brightness < gb_low_thresh else float(clip_limit)
    try:
        img = apply_clahe_rgb(img, clip_limit=clahe_limit, tile=int(max(1, clahe_tile)))
    except cv2.error:
        # CLAHE 실패 시 원본 그대로 반환
        pass

    # 3) 좌/우 분할 후 개별 감마 보정
    left_img = img[:, :w // 2]
    right_img = img[:, w // 2:]

    left_brightness = float(np.mean(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)))
    right_brightness = float(np.mean(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)))

    def enhance_region(region_img, region_brightness):
        diff = abs(target_brightness - region_brightness)

        # 매우 밝은 영역은 억제
        if region_brightness > region_overbright:
            return adjust_gamma(region_img, gamma=float(suppress_gamma))

        # 저조도(전역) 또는 타겟과 차이가 큰 경우: 부스팅
        if (global_brightness < gb_low_thresh) or (diff > brightness_threshold):
            est_gamma = np.log(target_brightness + 1e-6) / np.log(region_brightness + 1e-6)
            boost = gamma_boost_low if global_brightness < (gb_low_thresh - 10.0) else gamma_boost_med
            blended = float(np.clip((est_gamma + boost) / 2.0, 0.6, 2.0))
            return adjust_gamma(region_img, gamma=blended)

        # 그렇지 않으면 변화 없음
        return region_img

    left_img = enhance_region(left_img, left_brightness)
    right_img = enhance_region(right_img, right_brightness)
    return np.hstack((left_img, right_img))

# ==============================
# 평가 루틴 (TRT + 전처리)
# ==============================
def evaluate_trt(opt, engine_path, preprocess_params=None, verbose=True):
    MIN_DEPTH, MAX_DEPTH = 1e-3, 80
    if verbose:
        print(f"-> TensorRT 엔진 로딩: {engine_path}")
    trt_runner = TRTDepthEngine(engine_path, prefer_async_v3=True)

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

    eval_ratio = getattr(opt, 'eval_ratio', 1.0)
    if 0.0 < eval_ratio < 1.0:
        if verbose: print(f"-> 전체 테스트셋의 {eval_ratio:.2%} 샘플링")
        random.seed(42); random.shuffle(filenames)
        filenames = filenames[:int(len(filenames) * eval_ratio)]
        if verbose: print(f"-> 평가 샘플 수: {len(filenames)}")

    enc_h, enc_w = getattr(opt, 'height', 192), getattr(opt, 'width', 640)
    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames, enc_h, enc_w, [0], 4, is_train=False)
    dataloader = DataLoader(dataset, 1, False, num_workers=0, pin_memory=True, drop_last=False)

    pred_disps = []
    if verbose: print(f"-> 입력 {enc_w}x{enc_h} (post_process={opt.post_process})")
    for data in dataloader:
        input_color_t = data[("color", 0, 0)]  # (1,3,H,W) [0,1] RGB
        if preprocess_params:
            # To HWC uint8 BGR → 전처리 → back to torch
            img_np_chw = input_color_t.squeeze(0).numpy()
            img_np_hwc = np.transpose(img_np_chw, (1, 2, 0))
            img_uint8 = (img_np_hwc * 255.0).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            try:
                img_bgr = brightness_enhancement(img_bgr, **preprocess_params)
            except Exception:
                # 전처리 실패 시 원본 사용
                pass
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_f32 = (img_rgb / 255.0).astype(np.float32)
            img_chw = np.transpose(img_f32, (2, 0, 1))
            input_color_t = torch.from_numpy(img_chw).unsqueeze(0)

        if opt.post_process:
            x0 = input_color_t.numpy()
            x1 = torch.flip(input_color_t, dims=[3]).numpy()
            out0, out1 = trt_runner.infer(x0), trt_runner.infer(x1)
            disp0, disp1 = out0[:, 0], out1[:, 0]
            pred_disps.append(batch_post_process_disparity(disp0, disp1[:, :, ::-1]))
        else:
            out = trt_runner.infer(input_color_t.numpy())
            pred_disps.append(out[:, 0])

    pred_disps = np.concatenate(pred_disps, axis=0)
    if opt.no_eval:
        if verbose: print("-> no_eval=True: 메트릭 스킵")
        return None

    # GT 로드 + 메트릭
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    errors = []
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]; gt_h, gt_w = gt_depth.shape[:2]
        pred_disp = cv2.resize(pred_disps[i], (gt_w, gt_h))
        pred_depth = 1.0 / np.maximum(pred_disp, 1e-12)

        mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
        if opt.eval_split == "eigen":
            crop = np.array([0.40810811 * gt_h, 0.99189189 * gt_h, 0.03594771 * gt_w, 0.96405229 * gt_w]).astype(np.int32)
            crop_mask = np.zeros(mask.shape, dtype=bool)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
            mask &= crop_mask

        pred_depth = pred_depth[mask]; gt_depth = gt_depth[mask]
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            pred_depth *= ratio

        pred_depth = np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_errors(gt_depth, pred_depth))

    mean_errors = np.array(errors).mean(0)
    if verbose:
        print("\n" + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    return mean_errors

# ==============================
# GA 설정 (a1 최대화)
# ==============================
# 탐색 파라미터와 범위
PARAM_BOUNDS = {
    # CLAHE
    'clip_limit': (1.0, 6.0),
    'clahe_tile': (4, 16),
    # 임계들
    'brightness_threshold': (2.0, 20.0),   # 좌/우가 타겟과 얼마나 차이나야 보정할지
    'gb_low_thresh': (40.0, 70.0),         # 전역 저조도 기준
    'gb_high_thresh': (160.0, 220.0),      # 전역 과밝음 기준
    'region_overbright': (180.0, 235.0),   # 영역 과밝음 기준
    # 감마 계수
    'suppress_gamma': (0.5, 0.9),          # 억제용 감마(전체/영역)
    'gamma_boost_low': (1.3, 1.8),         # 매우 어두울 때 부스트
    'gamma_boost_med': (1.05, 1.4),        # 덜 어두울 때 부스트
    # 중앙 영역 비율 (새로 추가)
    'center_ratio': (0.05, 0.6),           # 타겟 밝기 측정할 중앙 영역 비율 (5%~40%)
}

# a1 최대화 → FitnessMax
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# 속성 생성기 등록
toolbox.register("attr_clip_limit",     random.uniform, *PARAM_BOUNDS['clip_limit'])
toolbox.register("attr_clahe_tile",     random.randint, int(PARAM_BOUNDS['clahe_tile'][0]), int(PARAM_BOUNDS['clahe_tile'][1]))
toolbox.register("attr_bright_th",      random.uniform, *PARAM_BOUNDS['brightness_threshold'])
toolbox.register("attr_gb_low",         random.uniform, *PARAM_BOUNDS['gb_low_thresh'])
toolbox.register("attr_gb_high",        random.uniform, *PARAM_BOUNDS['gb_high_thresh'])
toolbox.register("attr_region_over",    random.uniform, *PARAM_BOUNDS['region_overbright'])
toolbox.register("attr_sup_gamma",      random.uniform, *PARAM_BOUNDS['suppress_gamma'])
toolbox.register("attr_gamma_boost_lo", random.uniform, *PARAM_BOUNDS['gamma_boost_low'])
toolbox.register("attr_gamma_boost_md", random.uniform, *PARAM_BOUNDS['gamma_boost_med'])
toolbox.register("attr_center_ratio",   random.uniform, *PARAM_BOUNDS['center_ratio'])

attributes = (
    toolbox.attr_clip_limit,
    toolbox.attr_clahe_tile,
    toolbox.attr_bright_th,
    toolbox.attr_gb_low,
    toolbox.attr_gb_high,
    toolbox.attr_region_over,
    toolbox.attr_sup_gamma,
    toolbox.attr_gamma_boost_lo,
    toolbox.attr_gamma_boost_md,
    toolbox.attr_center_ratio,
)
toolbox.register("individual", tools.initCycle, creator.Individual, attributes, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_params(individual, opt, engine_path):
    # 개체 → 파라미터 dict
    params = {
        'clip_limit':        float(individual[0]),
        'clahe_tile':        int(max(1, round(individual[1]))),
        'brightness_threshold': float(individual[2]),
        'gb_low_thresh':     float(individual[3]),
        'gb_high_thresh':    float(individual[4]),
        'region_overbright': float(individual[5]),
        'suppress_gamma':    float(individual[6]),
        'gamma_boost_low':   float(individual[7]),
        'gamma_boost_med':   float(individual[8]),
        'center_ratio':      float(individual[9]),
    }

    # 논리 제약: gb_low < gb_high
    if params['gb_low_thresh'] >= params['gb_high_thresh']:
        params['gb_low_thresh'], params['gb_high_thresh'] = params['gb_high_thresh'] - 1.0, params['gb_high_thresh']

    mean_errors = evaluate_trt(opt, engine_path, preprocess_params=params, verbose=False)
    if mean_errors is None:
        return (0.0,)  # a1 최적화 → 실패는 0으로

    a1 = float(mean_errors[4])
    print("    - 파라미터:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in params.items()})
    print(f"    - 결과 (a1): {a1:.4f}")
    return (a1,)

def tune_with_ga(opt, engine_path):
    POP_SIZE = 60
    NGEN = 50
    CXPB = 0.8
    MUTPB = 0.2
    ELITE_SIZE = 5

    toolbox.register("evaluate", evaluate_params, opt=opt, engine_path=engine_path)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(5)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print(f"🚀 GA 탐색 시작 (세대: {NGEN}, 개체수: {POP_SIZE}, 엘리트: {ELITE_SIZE})")

    # 초기 평가
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    for gen in range(1, NGEN + 1):
        # 선택(엘리트 제외 수만큼)
        offspring = toolbox.select(pop, len(pop) - ELITE_SIZE)
        offspring = list(map(toolbox.clone, offspring))

        # 교차
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # 변이
        for m in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(m)
                del m.fitness.values

        # 재평가
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 엘리트 보존
        elites = tools.selBest(pop, ELITE_SIZE)
        offspring.extend(toolbox.clone(e) for e in elites)

        # 세대 교체
        pop[:] = offspring
        hof.update(pop)

        record = stats.compile(pop)
        print(f"세대 {gen}: {{'avg': {record['avg']:.4f}, 'max': {record['max']:.4f}}}")

    print("\n✅ 탐색 완료!")
    best_ind = hof[0]
    best_params = {
        'clip_limit':      float(best_ind[0]),
        'clahe_tile':      int(max(1, round(best_ind[1]))),
        'brightness_threshold': float(best_ind[2]),
        'gb_low_thresh':   float(best_ind[3]),
        'gb_high_thresh':  float(best_ind[4]),
        'region_overbright': float(best_ind[5]),
        'suppress_gamma':  float(best_ind[6]),
        'gamma_boost_low': float(best_ind[7]),
        'gamma_boost_med': float(best_ind[8]),
        'center_ratio':    float(best_ind[9]),
    }

    print("🏆 최적 파라미터")
    print(json.dumps(best_params, indent=4, ensure_ascii=False))
    print(f"\n최고 성능 (a1): {best_ind.fitness.values[0]:.4f}")

# ==============================
# 실행부
# ==============================
if __name__ == "__main__":
    options = LiteMonoOptions()
    opt = options.parse()

    model_type = 'v4_3_R'
    default_engine = osp.join("onnx_output", f'optimized_{model_type}.engine')

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--engine_path', type=str, default=default_engine)
    p.add_argument('--eval_ratio', type=float, default=1.0, help='평가에 사용할 테스트셋 비율 (0~1)')
    args, _ = p.parse_known_args()

    setattr(opt, 'eval_ratio', args.eval_ratio)

    if not os.path.exists(args.engine_path):
        raise ValueError(f"엔진 파일을 찾을 수 없습니다: {args.engine_path}")

    tune_with_ga(opt, args.engine_path)