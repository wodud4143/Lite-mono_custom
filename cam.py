import os
import time
import torch
import numpy as np
import tensorrt as trt
import onnx
import onnxruntime as ort
import pycuda.driver as cuda
import pycuda.autoinit

# -------------------------------------------------------------------
# 1. PyTorch → ONNX
# -------------------------------------------------------------------

class LiteMonoONNXConverter:
    def __init__(self, weights_folder, model_type="lite-mono"):
        self.weights_folder = weights_folder
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model_info()

    def _load_model_info(self):
        encoder_path = os.path.join(self.weights_folder, "encoder.pth")
        decoder_path = os.path.join(self.weights_folder, "depth.pth")
        if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.weights_folder}")
        
        # 딕셔너리 형태로 로드 (height, width와 state_dict 포함)
        self.encoder_dict = torch.load(encoder_path, map_location=self.device)
        self.decoder_dict = torch.load(decoder_path, map_location=self.device)
        # feed_height, feed_width 추출
        self.feed_height = self.encoder_dict["height"]
        self.feed_width = self.encoder_dict["width"]
        print(f"[정보] ONNX 변환 입력 크기: {self.feed_width} x {self.feed_height}")

    def create_pytorch_model(self):
        import networks  # 사용자의 프로젝트 네트워크 모듈
        from networks import LiteMono
        from networks import DepthDecoder

        # Encoder 생성 후 가중치 로드
        encoder = LiteMono(model=self.model_type,
                          height=self.feed_height,
                          width=self.feed_width)
        enc_state = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in self.encoder_dict.items() if k in enc_state})

        # Decoder 생성 후 가중치 로드
        depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
        dec_state = depth_decoder.state_dict()
        depth_decoder.load_state_dict({k: v for k, v in self.decoder_dict.items() if k in dec_state})

        # 통합 모델 정의
        class LiteMonoComplete(torch.nn.Module):
            def __init__(self, encoder, decoder):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, x):
                features = self.encoder(x)
                outputs = self.decoder(features)
                # "disp", 0 에 해당하는 disparity 반환
                return outputs[("disp", 0)]

        complete_model = LiteMonoComplete(encoder, depth_decoder)
        complete_model.to(self.device).eval()
        return complete_model

    def convert_to_onnx(self, pytorch_model, onnx_path, dynamic_batch=True):
        print("\n[진행] PyTorch → ONNX 변환 시작")
        dummy_input = torch.randn(1, 3, self.feed_height, self.feed_width).to(self.device)

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "disparity": {0: "batch_size"}
            }

        try:
            torch.onnx.export(
                pytorch_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["disparity"],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            print(f"[완료] ONNX 모델 저장: {onnx_path}")
            self._verify_onnx_model(onnx_path)
            return True
        except Exception as e:
            print(f"[오류] ONNX 변환 실패: {e}")
            return False

    def _verify_onnx_model(self, onnx_path):
        print("[진행] ONNX 모델 검증 중")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        graph = onnx_model.graph
        inputs = [inp.name for inp in graph.input]
        outputs = [outp.name for outp in graph.output]
        print(f"[정보] ONNX 입력: {inputs}, 출력: {outputs}")
        print("[완료] ONNX 모델 검증 성공")


# -------------------------------------------------------------------
# 2. ONNX → TensorRT
# -------------------------------------------------------------------

class ONNXToTensorRTBuilder:
    def __init__(self, max_workspace_size=1 << 30):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.trt_version = trt.__version__
        self.max_workspace_size = max_workspace_size
        print(f"[정보] TensorRT 버전: {self.trt_version}")

    def build_engine(self, onnx_path, engine_path, precision="fp16", max_batch_size=8):
        print("\n[진행] ONNX → TensorRT 엔진 빌드 시작")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 파일이 없습니다: {onnx_path}")

        # 1) ONNX 모델 검증
        self._verify_onnx_model(onnx_path)

        # 2) Builder, Network, Parser 생성
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)

        print("[진행] ONNX 파싱 중...")
        if not parser.parse_from_file(onnx_path):
            for i in range(parser.num_errors):
                print(f"[파싱 오류] {i}: {parser.get_error(i)}")
            return False
        print("[완료] ONNX 파싱 성공")

        # 3) BuilderConfig 설정
        config = builder.create_builder_config()
        # 워크스페이스 설정
        if hasattr(config, "max_workspace_size"):
            config.max_workspace_size = self.max_workspace_size
            print(f"[정보] 워크스페이스 크기 설정(legacy): {self.max_workspace_size // (1024*1024)} MB")
        elif hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.max_workspace_size)
            print(f"[정보] 워크스페이스 크기 설정(new): {self.max_workspace_size // (1024*1024)} MB")
        else:
            print("[경고] 워크스페이스 설정 방법을 찾을 수 없음")

        # 정밀도 설정
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[정보] FP16 정밀도 활성화")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("[정보] INT8 정밀도 활성화 (캘리브레이션 필요)")
        else:
            print("[정보] FP32 정밀도 사용")

        # 동적 배치 프로파일 설정
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        shape = input_tensor.shape  # [-1, 3, H, W] 형태일 가능성

        # shape[0]이 -1이면 동적 배치 모드
        if shape[0] == -1:
            min_shape = (1, shape[1], shape[2], shape[3])
            opt_shape = (max_batch_size // 2, shape[1], shape[2], shape[3])
            max_shape = (max_batch_size, shape[1], shape[2], shape[3])
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            print(f"[정보] 동적 배치 설정 → min: {min_shape}, opt: {opt_shape}, max: {max_shape}")
        else:
            print(f"[정보] 고정 입력 크기 사용: {tuple(shape)}")

        # Strict 타입 등 추가 플래그 (선택)
        if hasattr(trt.BuilderFlag, "STRICT_TYPES"):
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # 4) 엔진 빌드 및 직렬화
        print("[진행] TensorRT 엔진 빌드 중... (잠시 대기)")
        start_time = time.time()
        try:
            serialized_engine = builder.build_serialized_network(network, config)
        except AttributeError:
            engine = builder.build_engine(network, config)
            if engine is None:
                print("[오류] 엔진 빌드 실패")
                return False
            serialized_engine = engine.serialize()
        build_time = time.time() - start_time
        print(f"[완료] 엔진 빌드 소요 시간: {build_time:.2f}초")

        if serialized_engine is None:
            print("[오류] 엔진 직렬화 실패")
            return False

        # 5) 엔진 저장
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        engine_size = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"[완료] TensorRT 엔진 저장: {engine_path} ({engine_size:.2f} MB)")
        return True

    def _verify_onnx_model(self, onnx_path):
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[정보] ONNX 검증 통과")


# -------------------------------------------------------------------
# 3. 메인 함수: PyTorch → ONNX → TensorRT
# -------------------------------------------------------------------

def main():
    # 사용자 환경에 맞게 경로 수정
    weights_folder = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\stem_change\models\weights_139"
    onnx_output_dir = "./onnx_output"
    trt_output_dir = "./tensorrt_output"

    os.makedirs(onnx_output_dir, exist_ok=True)
    os.makedirs(trt_output_dir, exist_ok=True)

    onnx_path = os.path.join(onnx_output_dir, "litemono_stem.onnx")
    trt_path = os.path.join(trt_output_dir, "litemono_stem.engine")

    try:
        # 1) PyTorch 모델 생성 및 ONNX 변환
        print("\n==================== Lite-Mono ONNX 변환 ====================")
        converter = LiteMonoONNXConverter(weights_folder, model_type="lite-mono")
        pytorch_model = converter.create_pytorch_model()
        success_onnx = converter.convert_to_onnx(pytorch_model, onnx_path, dynamic_batch=True)
        if not success_onnx:
            print("[중단] ONNX 변환 실패로 종료")
            return

        # 2) ONNX → TensorRT 빌드
        print("\n==================== ONNX → TensorRT ====================")
        trt_builder = ONNXToTensorRTBuilder(max_workspace_size=1 << 30)
        success_trt = trt_builder.build_engine(
            onnx_path=onnx_path,
            engine_path=trt_path,
            precision="fp16",
            max_batch_size=8
        )
        if not success_trt:
            print("[중단] TensorRT 빌드 실패로 종료")
            return

        # 3) (선택) ONNX vs TensorRT 결과 비교
        # -- 검증용 예시: 임의 데이터로 inference 결과가 유사한지 확인
        print("\n==================== ONNX vs TensorRT 결과 비교 ====================")
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        test_shape = (2, 3, converter.feed_height, converter.feed_width)
        test_input = np.random.randn(*test_shape).astype(np.float32)

        # ONNX 추론
        start_onnx = time.time()
        onnx_out = session.run([session.get_outputs()[0].name], {input_name: test_input})[0]
        onnx_time = (time.time() - start_onnx) * 1000

        # TensorRT 추론
        from collections import namedtuple

        # TensorRT 엔진 인퍼런스용 간단 클래스
        class TRTInference:
            def __init__(self, engine_path):
                self.logger = trt.Logger(trt.Logger.WARNING)
                with open(engine_path, "rb") as f:
                    engine_data = f.read()
                runtime = trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(engine_data)
                self.context = self.engine.create_execution_context()
                # 바인딩 index 추출
                self.input_binding = None
                self.output_binding = None
                for i in range(self.engine.num_bindings):
                    if self.engine.binding_is_input(i):
                        self.input_binding = i
                        self.input_name = self.engine.get_binding_name(i)
                    else:
                        self.output_binding = i
                        self.output_name = self.engine.get_binding_name(i)
                self.stream = cuda.Stream()
                self.d_input = None
                self.d_output = None

            def infer(self, np_input):
                batch_size = np_input.shape[0]
                # 바인딩 shape 설정
                if not self.engine.has_implicit_batch_dimension:
                    self.context.set_binding_shape(self.input_binding, np_input.shape)
                    output_shape = self.context.get_binding_shape(self.output_binding)
                else:
                    output_shape = (batch_size,) + tuple(self.engine.get_binding_shape(self.output_binding)[1:])

                input_size = np_input.nbytes
                output_size = int(np.prod(output_shape) * 4)

                if self.d_input is None or input_size != getattr(self, "_last_i", 0):
                    if hasattr(self.d_input, "free"):
                        self.d_input.free()
                    self.d_input = cuda.mem_alloc(input_size)
                    self._last_i = input_size

                if self.d_output is None or output_size != getattr(self, "_last_o", 0):
                    if hasattr(self.d_output, "free"):
                        self.d_output.free()
                    self.d_output = cuda.mem_alloc(output_size)
                    self._last_o = output_size

                cuda.memcpy_htod_async(self.d_input, np_input.ravel(), self.stream)

                if self.engine.has_implicit_batch_dimension:
                    bindings = [int(self.d_input), int(self.d_output)]
                    self.context.execute_async(batch_size, bindings, self.stream.handle)
                else:
                    bindings = [int(self.d_input), int(self.d_output)]
                    self.context.execute_async_v2(bindings, self.stream.handle)

                output = np.empty(output_shape, dtype=np.float32)
                cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
                self.stream.synchronize()
                return output

        trt_infer = TRTInference(trt_path)
        start_trt = time.time()
        trt_out = trt_infer.infer(test_input)
        trt_time = (time.time() - start_trt) * 1000

        # 비교 결과
        diff = np.abs(onnx_out - trt_out)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        speedup = onnx_time / trt_time if trt_time > 0 else float("inf")

        print(f"[시간] ONNX 실행: {onnx_time:.2f} ms | TensorRT 실행: {trt_time:.2f} ms | 속도향상: {speedup:.2f}x")
        print(f"[오차] 최대 차이: {max_diff:.6f} | 평균 차이: {mean_diff:.6f}")
        if max_diff < 1e-3:
            print("[검증] ONNX vs TensorRT 결과 일치!")
        else:
            print("[검증] ONNX vs TensorRT 결과에 다소 차이 발생!")

        print("\n========= ✅ 전체 변환 및 검증 완료 =========")
        print(f"ONNX 파일: {onnx_path}")
        print(f"TensorRT 엔진: {trt_path}")
        print(f"ONNX 크기: {os.path.getsize(onnx_path)/(1024*1024):.2f} MB")
        print(f"TensorRT 크기: {os.path.getsize(trt_path)/(1024*1024):.2f} MB")

    except Exception as e:
        print(f"[오류] {e}")


if __name__ == "__main__":
    main()
