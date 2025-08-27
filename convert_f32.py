#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LiteMono Encoder+Decoder 체크포인트 → ONNX → TensorRT 엔진(FP32/FP16/INT8) 변환 스크립트
- FP32, FP16 항상 시도
- INT8 은 보정 이미지가 있을 때만(EntropyCalibrator2). PyCUDA가 없으면 INT8 스킵
- 입력 텐서: (N,3,H,W) float32, [0,1] 스케일 가정
"""
import os
import os.path as osp
import argparse
from glob import glob

import numpy as np
import torch
import onnx
import onnxoptimizer
from onnx import numpy_helper
import tensorrt as trt

from PIL import Image
from torchvision import transforms

import networks

# =============================
# 모델 래퍼
# =============================
class FullModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)
        disp = out[("disp", 0)]  # (N,1,H,W)
        return disp


def custom_load_state_dict(loaded_enc, loaded_dec, height=192, width=640, device="cpu"):
    with torch.no_grad():
        encoder = networks.LiteMono(model="lite-mono", drop_path_rate=0.2, width=width, height=height)
        decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=[0, 1, 2])

        enc_state_dict, dec_state_dict = encoder.state_dict(), decoder.state_dict()

        encoder.load_state_dict({
            k: v for k, v in loaded_enc.items()
            if k in enc_state_dict and enc_state_dict[k].shape == v.shape
        })
        decoder.load_state_dict({
            k: v for k, v in loaded_dec.items()
            if k in dec_state_dict and dec_state_dict[k].shape == v.shape
        })

        encoder.to(device).eval()
        decoder.to(device).eval()
    return encoder, decoder

# =============================
# INT8 Calibrator (선택 사항)
# =============================
class _MaybeCalibrator:
    def __init__(self):
        try:
            import pycuda.driver as cuda  # type: ignore
            import pycuda.autoinit  # noqa: F401
            self.cuda = cuda
            self.ok = True
        except Exception:
            self.cuda = None
            self.ok = False

class ImageBatchStream:
    def __init__(self, image_dir, batch_size, n_batches, height, width):
        self.paths = sorted([p for p in glob(osp.join(image_dir, "**", "*")) if p.lower().endswith((".jpg",".jpeg",".png",".bmp"))])
        assert len(self.paths) > 0, f"보정 이미지가 없습니다: {image_dir}"
        self.bs = batch_size
        self.n_batches = n_batches
        self.h, self.w = height, width
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),  # [0,1]
        ])
        self._i = 0

    def reset(self):
        self._i = 0

    def next_batch(self):
        if self._i >= self.n_batches:
            return None
        imgs = []
        for _ in range(self.bs):
            p = self.paths[(self._i * self.bs + _) % len(self.paths)]
            img = Image.open(p).convert("RGB")
            t = self.transform(img)
            imgs.append(t.unsqueeze(0))
        self._i += 1
        return torch.cat(imgs, dim=0).numpy().astype(np.float32)  # (B,3,H,W)


def make_int8_calibrator(image_dir, height, width, batch_size=8, n_batches=100):
    maybe = _MaybeCalibrator()
    if not maybe.ok:
        return None

    cuda = maybe.cuda
    stream = ImageBatchStream(image_dir, batch_size, n_batches, height, width)

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.stream = stream
            self.d_input = None
            self.cache = osp.join(image_dir, f"calib_cache_H{height}_W{width}_B{batch_size}.bin")

        def get_batch_size(self):
            return self.stream.bs

        def get_batch(self, names):
            batch = self.stream.next_batch()
            if batch is None:
                return None
            if self.d_input is None:
                self.d_input = cuda.mem_alloc(batch.nbytes)
            cuda.memcpy_htod(self.d_input, batch)
            return [int(self.d_input)]

        def read_calibration_cache(self):
            if osp.exists(self.cache):
                with open(self.cache, 'rb') as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache, 'wb') as f:
                f.write(cache)

    return EntropyCalibrator()

# =============================
# TensorRT 엔진 빌드
# =============================

def build_engine(onnx_file_path, precision="fp32", height=192, width=640, max_batch=8, calibrator=None):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX Parse Error:", parser.get_error(i))
            raise RuntimeError("ONNX 파싱 실패")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # Optimization Profile (입력명은 export 시 지정한 "input")
    profile = builder.create_optimization_profile()
    min_shape = (1, 3, height, width)
    opt_shape = (1, 3, height, width)
    max_shape = (1, 3, height, width)
    profile.set_shape("input", min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    prec = precision.lower()
    if prec == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif prec == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is None:
            print("[INT8] Calibrator가 없어 FP32로 대체됩니다.")
        else:
            config.int8_calibrator = calibrator
    # FP32는 기본

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"{precision.upper()} 엔진 빌드 실패")

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    return engine

# =============================
# ONNX Export & Optimize
# =============================

def export_and_optimize_onnx(onnx_dir, model, height=192, width=640, model_type="lite_mono", device="cpu"):
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = osp.join(onnx_dir, f"{model_type}.onnx")
    dummy = torch.randn(1, 3, height, width, device=device)

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes=None,  # 고정 해상도/배치 기준
        export_params=True,
        do_constant_folding=True,
        opset_version=17,
    )

    model_onnx = onnx.load(onnx_path)
    passes = [
        'eliminate_deadend','eliminate_identity','eliminate_nop_transpose',
        'fuse_consecutive_transposes','fuse_bn_into_conv','fuse_pad_into_conv','fuse_add_bias_into_conv'
    ]
    optimized = onnxoptimizer.optimize(model_onnx, passes)

    # INT64 → INT32 변환(일부 백엔드 호환용)
    for init in optimized.graph.initializer:
        if init.data_type == onnx.TensorProto.INT64:
            arr = numpy_helper.to_array(init).astype('int32')
            init.CopyFrom(numpy_helper.from_array(arr, init.name))

    opt_onnx_path = osp.join(onnx_dir, f"optimized_{model_type}.onnx")
    onnx.save(optimized, opt_onnx_path)
    return opt_onnx_path

# =============================
# 메인 + 함수형 진입점(run_build)
# =============================

def run_build(*, enc, dec, onnx_dir, height=192, width=640, device='cuda', model_type='lite_mono', calib_dir=None, calib_batches=100, calib_batch_size=8):
    device_t = torch.device(device if torch.cuda.is_available() and str(device).startswith('cuda') else 'cpu')

    enc_sd = torch.load(enc, map_location='cpu')
    dec_sd = torch.load(dec, map_location='cpu')

    encoder, decoder = custom_load_state_dict(enc_sd, dec_sd, height=height, width=width, device=device_t)
    model = FullModel(encoder, decoder).eval().to(device_t)

    opt_onnx = export_and_optimize_onnx(onnx_dir, model, height, width, model_type, device=device_t)

    calibrator = None
    if calib_dir:
        calibrator = make_int8_calibrator(
            image_dir=calib_dir,
            height=height,
            width=width,
            batch_size=calib_batch_size,
            n_batches=calib_batches,
        )
        if calibrator is None:
            print('[INT8] PyCUDA 미탑재로 INT8 스킵')

    for prec in ["fp32", "fp16", "int8"]:
        if prec == 'int8' and (calib_dir is None or calibrator is None):
            continue
        try:
            engine = build_engine(opt_onnx, precision=prec, height=height, width=width, calibrator=calibrator)
            out_path = osp.join(onnx_dir, f"optimized_{model_type}_{prec}.engine")
            with open(out_path, 'wb') as f:
                f.write(engine.serialize())
            print(f"[{prec.upper()}] 엔진 생성 완료 → {out_path}")
        except Exception as e:
            print(f"[{prec.upper()}] 엔진 생성 실패: {e}")

    print('\nDone.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--enc', required=False, help='encoder.pth 경로')
    ap.add_argument('--dec', required=False, help='depth.pth 경로')
    ap.add_argument('--onnx_dir', required=False, help='ONNX/엔진 출력 디렉토리')
    ap.add_argument('--height', type=int, default=192)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--model_type', default='lite_mono')
    ap.add_argument('--calib_dir', default=None, help='INT8 보정 이미지 폴더(없으면 INT8 스킵)')
    ap.add_argument('--calib_batches', type=int, default=100)
    ap.add_argument('--calib_batch_size', type=int, default=8)
    args, unknown = ap.parse_known_args()

    # 인자 없으면 inline 예시 사용
    if not args.enc or not args.dec or not args.onnx_dir:
        device = 'cuda'
        model_type = 'lite_v4'
        weight_num = 97
        enc_model_path = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\{0}\models\weights_{1}\encoder.pth".format(model_type,weight_num)
        dec_model_path = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\{0}\models\weights_{1}\depth.pth".format(model_type,weight_num)
        onnx_dir = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\onnx_output"

        run_build(
            enc=enc_model_path,
            dec=dec_model_path,
            onnx_dir=onnx_dir,
            height=192,
            width=640,
            device=device,
            model_type=model_type,
            # calib_dir=r"C:\\path\\to\\calib_images",  # INT8 사용 시 지정
            calib_dir=None,
            calib_batches=100,
            calib_batch_size=8,
        )
        return

    # 인자가 모두 있으면 CLI 모드로 수행
    run_build(
        enc=args.enc,
        dec=args.dec,
        onnx_dir=args.onnx_dir,
        height=args.height,
        width=args.width,
        device=args.device,
        model_type=args.model_type,
        calib_dir=args.calib_dir,
        calib_batches=args.calib_batches,
        calib_batch_size=args.calib_batch_size,
    )


if __name__ == '__main__':
    main()
