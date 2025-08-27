from __future__ import absolute_import, division, print_function
import os
from pathlib import Path
import time
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import os.path as osp
import tensorrt as trt

from layers import disp_to_depth
from utils import readlines
from options import LiteMonoOptions
import datasets


cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# =============================
# TensorRT 런타임 래퍼
# =============================
class TRTDepthEngine:
    """단일 fused 엔진(입력 1, 출력 1: disparity/disp)을 가정한 경량 래퍼.
    - 엔진은 NCHW float 입력을 받는다고 가정
    - 동적/정적 배치 모두 지원 (동적이면 set_input_shape 사용)
    - PyCUDA 의존 없이 torch CUDA 텐서의 data_ptr()로 execute_v2 수행
    """
    def __init__(self, engine_path: str, prefer_async_v3: bool = False):
        assert os.path.exists(engine_path), f"엔진 파일을 찾을 수 없습니다: {engine_path}"
        self.logger = trt.Logger(trt.Logger.ERROR)  # 경고 억제(특히 hasImplicitBatchDimension 관련)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # 바인딩 이름/인덱스 정리
        self.input_binding = None
        self.output_binding = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_binding = name
            else:
                self.output_binding = name
        assert self.input_binding is not None and self.output_binding is not None, "엔진 입/출력 바인딩을 확인하세요."

        # 실행 방식 결정
        self.prefer_async_v3 = prefer_async_v3 and hasattr(self.context, 'execute_async_v3')

    def _ensure_shape(self, x: torch.Tensor):
        """동적 shape 엔진일 때 입력 shape 설정"""
        shape = tuple(x.shape)
        # TensorRT 10은 explicit batch 고정 → 항상 set_input_shape 수행
        curr = self.context.get_tensor_shape(self.input_binding)
        if tuple(curr) != shape:
            self.context.set_input_shape(self.input_binding, shape)
        return shape

    @torch.inference_mode()
    def infer(self, x_np: np.ndarray) -> np.ndarray:
        """x_np: (N,C,H,W) float32, range [0,1] or training에 맞춘 정규화
        returns: (N,1,H,W) float32 disparity
        """
        assert x_np.ndim == 4, f"입력은 (N,C,H,W) 여야 합니다. got {x_np.shape}"
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        self._ensure_shape(x)

        # 출력 shape 조회 (explicit batch)
        out_shape = tuple(self.context.get_tensor_shape(self.output_binding))
        if any(d < 0 for d in out_shape):
            N, _, H, W = x.shape
            out_shape = (N, 1, H, W)
        else:
            out_shape = tuple(self.context.get_tensor_shape(self.output_binding))
            # TensorRT가 -1 포함하면 입력 기반으로 추정
            if any(d < 0 for d in out_shape):
                N, _, H, W = x.shape
                out_shape = (N, 1, H, W)

        y = torch.empty(out_shape, dtype=torch.float32, device=device)

        # 텐서 바인딩
        self.context.set_tensor_address(self.input_binding, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_binding, int(y.data_ptr()))

        # 실행
        if self.prefer_async_v3:
            ok = self.context.execute_async_v3(stream_handle=0)  # 기본 스트림
        else:
            ok = self.context.execute_v2()
        if not ok:
            raise RuntimeError("TensorRT 실행 실패: execute_v2/async_v3 반환값 False")

        return y.detach().float().cpu().numpy()


# =============================
# 평가 유틸
# =============================

def time_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
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


# =============================
# 메인 평가 루틴 (TensorRT)
# =============================

def evaluate_trt(opt, engine_path=None):
    """TensorRT 엔진으로 평가
    - 엔진 출력: (N,1,H,W) disp 가정
    - 입력 전처리: [0,1] 스케일, RGB, NCHW
    """
    MIN_DEPTH, MAX_DEPTH = 1e-3, 80

    # 엔진 경로 결정
    if engine_path is None:
        # 1) opt.engine_path 가 있으면 사용, 2) load_weights_folder/model.engine 기본
        engine_path = getattr(opt, 'engine_path', None)
        if engine_path is None:
            engine_path = os.path.join(os.path.expanduser(opt.load_weights_folder), 'model.engine')

    print(f"-> TensorRT 엔진 로딩: {engine_path}")
    trt_runner = TRTDepthEngine(engine_path, prefer_async_v3=True)

    # splits/ 평가 세트 구성
    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

    # height/width 추론 기준: 엔진 입력 텐서 shape 또는 저장된 encoder.pth에서 읽을 수 있으나
    # 여기서는 옵션의 eval 높/너비를 우선. 없으면 엔진 profile에서 유추.
    # 대부분 훈련 시 (H,W) = (192,640) 형태를 사용.
    enc_h = getattr(opt, 'height', 192)
    enc_w = getattr(opt, 'width', 640)

    # DataLoader (batch=1 권장: 동적 shape/좌우플립 포스트프로세스 시 처리 단순화)
    # datasets.KITTIRAWDataset은 ("color", 0, 0) 키의 torch.Tensor (B,3,H,W) 를 반환한다고 가정
    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames, enc_h, enc_w, [0], 4, is_train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    pred_disps = []

    print(f"-> 입력 크기 {enc_w}x{enc_h} 로 예측 수행 (post_process={opt.post_process})")

    # 타이밍 측정
    latencies = []
    warmup = 20
    seen = 0

    for data in dataloader:
        # (1,3,H,W) torch tensor in [0,1]
        input_color_t = data[("color", 0, 0)]  # CPU tensor
        # 좌우 flip 포스트프로세스: 엔진 2회 실행 후 결합
        if opt.post_process:
            # 원본
            x0 = input_color_t.numpy()  # (1,3,H,W), float32 assumed
            # 좌우 flip (W축 flip)
            x1 = torch.flip(input_color_t, dims=[3]).numpy()

            # warmup + 타이밍
            if seen < warmup:
                _ = trt_runner.infer(x0)
                _ = trt_runner.infer(x1)
            t1 = time_sync()
            out0 = trt_runner.infer(x0)  # (1,1,H,W)
            out1 = trt_runner.infer(x1)
            t2 = time_sync()
            latencies.append((t2 - t1) * 1000.0)

            # disp_to_depth 앞서 disp만 추출
            disp0 = out0[:, 0, :, :]  # (1,H,W)
            disp1 = out1[:, 0, :, :]
            disp_pp = batch_post_process_disparity(disp0, disp1[:, :, ::-1])  # (1,H,W)
            pred_disps.append(disp_pp)
        else:
            x = input_color_t.numpy()  # (1,3,H,W)
            if seen < warmup:
                _ = trt_runner.infer(x)
            t1 = time_sync()
            out = trt_runner.infer(x)
            t2 = time_sync()
            latencies.append((t2 - t1) * 1000.0)

            disp = out[:, 0, :, :]  # (1,H,W)
            pred_disps.append(disp)

        seen += 1

    pred_disps = np.concatenate(pred_disps, axis=0)  # (N,H,W)

    if opt.save_pred_disps:
        base_dir = os.path.expanduser(getattr(opt, 'load_weights_folder', Path(engine_path).parent))
        os.makedirs(base_dir, exist_ok=True)
        output_path = os.path.join(base_dir, f"disps_{opt.eval_split}_split_trt.npy")
        print("-> 예측 Disparity 저장:", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> no_eval=True: 평가 스킵")
        return None

    # GT 로드 및 메트릭 계산
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    print("-> 메트릭 계산 (median scaling)")

    errors, ratios = [], []
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1.0 / np.maximum(pred_disp, 1e-12)

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_h, 0.99189189 * gt_h,
                             0.03594771 * gt_w, 0.96405229 * gt_w]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth = np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f" Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

        mean_errors = np.array(errors).mean(0)

    # 결과 출력
    # 출력 식별자: load_weights_folder 없으면 엔진 파일명 사용
    run_id = Path(engine_path).stem if engine_path else "trt_run"
    print(run_id + "\n")

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")

    if len(latencies) > 0:
        lat = np.array(latencies[warmup:]) if len(latencies) > warmup else np.array(latencies)
        print(f"\n  TRT Inference Latency per sample: mean={lat.mean():.2f} ms | p50={np.percentile(lat,50):.2f} | p90={np.percentile(lat,90):.2f} | n={len(lat)}")

    print("\n-> Done (TRT)!\n")
    return mean_errors


if __name__ == "__main__":
    # 기존 LiteMonoOptions를 그대로 사용하면서 engine_path만 추가로 주입
    options = LiteMonoOptions()
    opt = options.parse()
    ver = 'fp16'
    model_type = 'lite_v4'
    engine_path = osp.join("onnx_output", f'optimized_{model_type}_{ver}.engine')
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--engine_path', type=str, default=engine_path, help='TensorRT engine 파일 경로 (기본: <load_weights_folder>/model.engine)')
    args, _ = p.parse_known_args()
    if args.engine_path is not None:
        setattr(opt, 'engine_path', args.engine_path)

    evaluate_trt(opt, engine_path=getattr(opt, 'engine_path', None))
