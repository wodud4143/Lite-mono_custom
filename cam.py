

# region pth -> ONNX 변환
# """
# Lite-Mono PyTorch → ONNX 변환
# TensorRT 없이 ONNX만 사용하는 버전
# """

# import torch
# import torch.nn as nn
# import numpy as np
# import os
# import time
# import argparse
# from pathlib import Path

# # 원본 모듈들 임포트
# import networks
# from layers import disp_to_depth

# # ONNX 관련 (TensorRT 없음)
# import onnx
# import onnxruntime as ort

# class LiteMonoONNX:
#     """Lite-Mono ONNX 변환 및 추론 클래스 (TensorRT 미사용)"""
    
#     def __init__(self, weights_folder, model_type="lite-mono"):
#         self.weights_folder = weights_folder
#         self.model_type = model_type
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # 모델 정보 로드
#         self._load_model_info()
        
#     def _load_model_info(self):
#         """모델 정보 및 가중치 로드"""
#         encoder_path = os.path.join(self.weights_folder, "encoder.pth")
#         decoder_path = os.path.join(self.weights_folder, "depth.pth")
        
#         if not os.path.exists(encoder_path) or not os.path.exists(decoder_path):
#             raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {self.weights_folder}")
        
#         # 모델 딕셔너리 로드
#         self.encoder_dict = torch.load(encoder_path, map_location=self.device)
#         self.decoder_dict = torch.load(decoder_path, map_location=self.device)
        
#         # 입력 크기 정보
#         self.feed_height = self.encoder_dict['height']
#         self.feed_width = self.encoder_dict['width']
        
#         print(f"모델 입력 크기: {self.feed_width} x {self.feed_height}")
        
#     def create_pytorch_model(self):
#         """PyTorch 통합 모델 생성"""
        
#         # 인코더 생성
#         encoder = networks.LiteMono(
#             model=self.model_type,
#             height=self.feed_height,
#             width=self.feed_width
#         )
        
#         # 인코더 가중치 로드
#         model_dict = encoder.state_dict()
#         encoder.load_state_dict({k: v for k, v in self.encoder_dict.items() if k in model_dict})
        
#         # 디코더 생성
#         depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
        
#         # 디코더 가중치 로드
#         depth_model_dict = depth_decoder.state_dict()
#         depth_decoder.load_state_dict({k: v for k, v in self.decoder_dict.items() if k in depth_model_dict})
        
#         # 통합 모델 클래스
#         class LiteMonoComplete(nn.Module):
#             def __init__(self, encoder, decoder):
#                 super().__init__()
#                 self.encoder = encoder
#                 self.decoder = decoder
            
#             def forward(self, x):
#                 # 인코더로 특징 추출
#                 features = self.encoder(x)
#                 # 디코더로 깊이 예측
#                 outputs = self.decoder(features)
#                 # disparity만 반환
#                 return outputs[("disp", 0)]
        
#         # 통합 모델 생성
#         complete_model = LiteMonoComplete(encoder, depth_decoder)
#         complete_model.to(self.device)
#         complete_model.eval()
        
#         return complete_model
    
#     def convert_to_onnx(self, pytorch_model, onnx_path, dynamic_batch=True):
#         """PyTorch → ONNX 변환"""
        
#         print("=== PyTorch → ONNX 변환 시작 ===")
        
#         # 입력 텐서 생성
#         input_shape = (1, 3, self.feed_height, self.feed_width)
#         dummy_input = torch.randn(input_shape).to(self.device)
        
#         # 동적 축 설정
#         dynamic_axes = None
#         if dynamic_batch:
#             dynamic_axes = {
#                 'input': {0: 'batch_size'},
#                 'disparity': {0: 'batch_size'}
#             }
        
#         try:
#             # ONNX 내보내기
#             torch.onnx.export(
#                 pytorch_model,                          # 모델
#                 dummy_input,                           # 더미 입력
#                 onnx_path,                            # 저장 경로
#                 export_params=True,                    # 파라미터 포함
#                 opset_version=11,                      # ONNX opset 버전
#                 do_constant_folding=True,              # 상수 폴딩 최적화
#                 input_names=['input'],                 # 입력 이름
#                 output_names=['disparity'],            # 출력 이름
#                 dynamic_axes=dynamic_axes,             # 동적 축
#                 verbose=False                          # 상세 로그
#             )
            
#             print(f"ONNX 변환 완료: {onnx_path}")
            
#             # ONNX 모델 검증
#             self._verify_onnx_model(onnx_path)
            
#             return True
            
#         except Exception as e:
#             print(f"ONNX 변환 실패: {e}")
#             import traceback
#             traceback.print_exc()
#             return False
    
#     def _verify_onnx_model(self, onnx_path):
#         """ONNX 모델 검증"""
#         try:
#             onnx_model = onnx.load(onnx_path)
#             onnx.checker.check_model(onnx_model)
#             print("ONNX 모델 검증 성공")
            
#             # 모델 정보 출력
#             graph = onnx_model.graph
#             print(f"ONNX 모델 정보:")
#             print(f"  입력: {[input.name for input in graph.input]}")
#             print(f"  출력: {[output.name for output in graph.output]}")
            
#         except Exception as e:
#             print(f"ONNX 모델 검증 실패: {e}")

# class ONNXInference:
#     """ONNX Runtime을 사용한 추론 클래스"""
    
#     def __init__(self, onnx_path, use_gpu=True):
#         """ONNX 모델 로드"""
        
#         # ONNX Runtime 프로바이더 설정
#         providers = []
#         if use_gpu and ort.get_device() == 'GPU':
#             providers.append('CUDAExecutionProvider')
#         providers.append('CPUExecutionProvider')
        
#         # ONNX Runtime 세션 생성
#         self.session = ort.InferenceSession(onnx_path, providers=providers)
        
#         # 입출력 정보
#         self.input_name = self.session.get_inputs()[0].name
#         self.output_name = self.session.get_outputs()[0].name
        
#         input_shape = self.session.get_inputs()[0].shape
#         output_shape = self.session.get_outputs()[0].shape
        
#         print(f"ONNX Runtime 세션 생성 완료")
#         print(f"사용 중인 프로바이더: {self.session.get_providers()}")
#         print(f"입력: {self.input_name}, shape: {input_shape}")
#         print(f"출력: {self.output_name}, shape: {output_shape}")
    
#     def infer(self, input_data):
#         """추론 실행"""
        
#         # 입력 데이터 검증
#         if not isinstance(input_data, np.ndarray):
#             raise TypeError("입력 데이터는 numpy array여야 합니다")
        
#         if input_data.dtype != np.float32:
#             input_data = input_data.astype(np.float32)
        
#         # 추론 실행
#         try:
#             result = self.session.run(
#                 [self.output_name], 
#                 {self.input_name: input_data}
#             )
#             return result[0]
#         except Exception as e:
#             print(f"ONNX 추론 실패: {e}")
#             raise

# def compare_pytorch_onnx(pytorch_model, onnx_path, test_input):
#     """PyTorch와 ONNX 결과 비교"""
    
#     print("=== PyTorch vs ONNX 결과 비교 ===")
    
#     # PyTorch 추론
#     pytorch_model.eval()
#     with torch.no_grad():
#         torch_input = torch.from_numpy(test_input).cuda()
#         torch_output = pytorch_model(torch_input).cpu().numpy()
    
#     # ONNX 추론
#     onnx_inference = ONNXInference(onnx_path)
#     onnx_output = onnx_inference.infer(test_input)
    
#     # 결과 비교
#     diff = np.abs(torch_output - onnx_output)
#     max_diff = np.max(diff)
#     mean_diff = np.mean(diff)
    
#     print(f"최대 차이: {max_diff:.6f}")
#     print(f"평균 차이: {mean_diff:.6f}")
    
#     if max_diff < 1e-4:
#         print("✅ PyTorch와 ONNX 결과가 일치합니다!")
#     else:
#         print("⚠️ PyTorch와 ONNX 결과에 차이가 있습니다.")
    
#     return max_diff < 1e-4

# def benchmark_inference(onnx_path, input_shape, num_runs=100):
#     """ONNX 추론 성능 벤치마크"""
    
#     print(f"=== ONNX 추론 성능 테스트 ({num_runs}회) ===")
    
#     # ONNX 추론기 생성
#     onnx_inference = ONNXInference(onnx_path)
    
#     # 테스트 데이터 생성
#     test_input = np.random.randn(*input_shape).astype(np.float32)
    
#     # 워밍업 (첫 번째 추론은 초기화 시간 포함)
#     _ = onnx_inference.infer(test_input)
    
#     # 성능 측정
#     times = []
#     for i in range(num_runs):
#         start_time = time.time()
#         _ = onnx_inference.infer(test_input)
#         end_time = time.time()
#         times.append((end_time - start_time) * 1000)  # ms
    
#     # 통계 계산
#     avg_time = np.mean(times)
#     min_time = np.min(times)
#     max_time = np.max(times)
#     std_time = np.std(times)
    
#     print(f"평균 추론 시간: {avg_time:.2f} ms")
#     print(f"최소 추론 시간: {min_time:.2f} ms")
#     print(f"최대 추론 시간: {max_time:.2f} ms")
#     print(f"표준편차: {std_time:.2f} ms")
#     print(f"초당 추론 횟수: {1000/avg_time:.1f} FPS")

# def test_real_image(onnx_path, image_path, feed_height, feed_width):
#     """실제 이미지로 테스트"""
    
#     print(f"=== 실제 이미지 테스트: {image_path} ===")
    
#     try:
#         import PIL.Image as pil
#         from torchvision import transforms
#         import matplotlib.pyplot as plt
#         import matplotlib.cm as cm
        
#         # 이미지 로드
#         image = pil.open(image_path).convert('RGB')
#         original_size = image.size
#         print(f"원본 이미지 크기: {original_size}")
        
#         # 전처리
#         image_resized = image.resize((feed_width, feed_height), pil.LANCZOS)
#         transform = transforms.ToTensor()
#         input_tensor = transform(image_resized).unsqueeze(0).numpy()
        
#         # 추론
#         onnx_inference = ONNXInference(onnx_path)
        
#         start_time = time.time()
#         disparity = onnx_inference.infer(input_tensor)
#         end_time = time.time()
        
#         print(f"추론 시간: {(end_time - start_time) * 1000:.2f} ms")
#         print(f"출력 shape: {disparity.shape}")
        
#         # 결과 시각화
#         disp_np = disparity.squeeze()
        
#         plt.figure(figsize=(15, 5))
        
#         plt.subplot(1, 3, 1)
#         plt.imshow(image)
#         plt.title('Original Image')
#         plt.axis('off')
        
#         plt.subplot(1, 3, 2)
#         plt.imshow(disp_np, cmap='magma')
#         plt.title('Disparity Map')
#         plt.axis('off')
        
#         plt.subplot(1, 3, 3)
#         plt.imshow(disp_np, cmap='plasma')
#         plt.title('Disparity Map (Plasma)')
#         plt.axis('off')
        
#         plt.tight_layout()
#         plt.savefig('onnx_inference_result.png', dpi=150, bbox_inches='tight')
#         plt.show()
        
#         print("결과 이미지 저장: onnx_inference_result.png")
        
#     except Exception as e:
#         print(f"이미지 테스트 실패: {e}")
#         import traceback
#         traceback.print_exc()

# def main():
#     """메인 실행 함수"""
    
#     # 설정
#     weights_folder = r'C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\Ghost_only_CDC_pre_100ep\models\weights_98'
#     output_dir = "./onnx_output"
#     model_type = "lite-mono"
    
#     # 출력 디렉토리 생성
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 파일 경로
#     onnx_path = os.path.join(output_dir, "litemono.onnx")
    
#     try:
#         # 1. ONNX 변환기 생성
#         print("=== Lite-Mono ONNX 변환 시작 ===")
#         converter = LiteMonoONNX(weights_folder, model_type)
        
#         # 2. PyTorch 모델 생성
#         print("\n=== PyTorch 모델 생성 ===")
#         pytorch_model = converter.create_pytorch_model()
#         print("PyTorch 모델 생성 완료")
        
#         # 3. PyTorch → ONNX 변환
#         print(f"\n=== PyTorch → ONNX 변환 ===")
#         success = converter.convert_to_onnx(pytorch_model, onnx_path, dynamic_batch=True)
        
#         if not success:
#             print("ONNX 변환 실패")
#             return
        
#         # 4. 테스트 데이터 생성
#         test_input = np.random.randn(
#             2, 3, 
#             converter.feed_height, 
#             converter.feed_width
#         ).astype(np.float32)
        
#         # 5. PyTorch vs ONNX 비교
#         compare_pytorch_onnx(pytorch_model, onnx_path, test_input)
        
#         # 6. 성능 벤치마크
#         benchmark_inference(onnx_path, test_input.shape)
        
#         # 7. 실제 이미지 테스트 (이미지 파일이 있는 경우)
#         # test_image_path = "path/to/your/test/image.jpg"
#         # if os.path.exists(test_image_path):
#         #     test_real_image(onnx_path, test_image_path, converter.feed_height, converter.feed_width)
        
#         print(f"\n=== 변환 완료! ===")
#         print(f"ONNX 파일: {onnx_path}")
#         print(f"ONNX 파일 크기: {os.path.getsize(onnx_path) / (1024*1024):.2f} MB")
        
#     except Exception as e:
#         print(f"오류 발생: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()




# region Onnx to TensorRT

"""
ONNX → TensorRT 엔진 변환 (최신 TensorRT API 호환)
TensorRT 8.x/9.x/10.x 버전 호환성 문제 해결
"""

import tensorrt as trt
import numpy as np
import os
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path

# ONNX 관련
import onnx
import onnxruntime as ort

class ONNXToTensorRTFixed:
    """ONNX 모델을 TensorRT 엔진으로 변환하는 클래스 (최신 API 호환)"""
    
    def __init__(self):
        # TensorRT 로거 설정
        self.logger = trt.Logger(trt.Logger.INFO)
        
        # TensorRT 버전 확인
        self.trt_version = trt.__version__
        print(f"TensorRT 버전: {self.trt_version}")
        
    def build_engine_from_onnx(self, onnx_path, engine_path, 
                              precision='fp16', max_batch_size=8, 
                              max_workspace_size=1 << 30):
        """
        ONNX 파일에서 TensorRT 엔진 빌드 (최신 API 호환)
        """
        
        print(f"=== ONNX → TensorRT 변환 시작 ===")
        print(f"ONNX 파일: {onnx_path}")
        print(f"엔진 파일: {engine_path}")
        
        # ONNX 파일 존재 확인
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX 파일을 찾을 수 없습니다: {onnx_path}")
        
        # ONNX 모델 검증
        self._verify_onnx_model(onnx_path)
        
        # Builder와 Network 생성
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.logger)
        
        # ONNX 파일 파싱
        print("ONNX 파일 파싱 중...")
        success = parser.parse_from_file(onnx_path)
        
        if not success:
            print("❌ ONNX 파싱 실패:")
            for idx in range(parser.num_errors):
                print(f"  에러 {idx}: {parser.get_error(idx)}")
            return False
        
        print("✅ ONNX 파싱 성공")
        
        # 네트워크 정보 출력
        self._print_network_info(network)
        
        # Builder Config 설정 (API 버전 호환)
        config = builder.create_builder_config()
        
        # 워크스페이스 크기 설정 (버전별 호환)
        if hasattr(config, 'max_workspace_size'):
            # TensorRT 7.x/8.x 버전
            config.max_workspace_size = max_workspace_size
            print(f"워크스페이스 설정 (legacy): {max_workspace_size // (1024*1024)} MB")
        elif hasattr(config, 'set_memory_pool_limit'):
            # TensorRT 8.5+ 버전
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
            print(f"워크스페이스 설정 (new): {max_workspace_size // (1024*1024)} MB")
        else:
            print("⚠️ 워크스페이스 설정 방법을 찾을 수 없습니다.")
        
        # 정밀도 설정
        if precision == 'fp16' and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ FP16 정밀도 사용")
        elif precision == 'int8' and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("✅ INT8 정밀도 사용")
            # INT8 사용시 캘리브레이션 필요 (여기서는 생략)
        else:
            print("✅ FP32 정밀도 사용")
        
        # 최적화 프로파일 설정 (동적 배치 크기)
        profile = builder.create_optimization_profile()
        
        # 입력 텐서 정보
        input_tensor = network.get_input(0)
        input_name = input_tensor.name
        input_shape = input_tensor.shape
        
        print(f"입력 정보: {input_name}, shape: {input_shape}")
        
        # 동적 shape 설정
        if input_shape[0] == -1:  # 동적 배치 크기
            # 배치 크기만 동적
            min_shape = (1, input_shape[1], input_shape[2], input_shape[3])
            opt_shape = (max_batch_size // 2, input_shape[1], input_shape[2], input_shape[3])
            max_shape = (max_batch_size, input_shape[1], input_shape[2], input_shape[3])
            
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            print(f"동적 배치 설정:")
            print(f"  최소: {min_shape}")
            print(f"  최적: {opt_shape}")
            print(f"  최대: {max_shape}")
        else:
            print(f"고정 입력 크기: {input_shape}")
        
        # 추가 최적화 플래그 설정
        try:
            # TensorRT 8.5+에서 사용 가능한 최적화
            if hasattr(trt.BuilderFlag, 'STRICT_TYPES'):
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            
            # GPU fallback 활성화
            if hasattr(trt.BuilderFlag, 'GPU_FALLBACK'):
                config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
                
        except Exception as e:
            print(f"추가 최적화 설정 실패 (무시): {e}")
        
        # 엔진 빌드
        print("TensorRT 엔진 빌드 중... (시간이 오래 걸릴 수 있습니다)")
        start_time = time.time()
        
        try:
            # TensorRT 8.5+ 버전
            serialized_engine = builder.build_serialized_network(network, config)
        except AttributeError:
            # TensorRT 7.x/8.x 버전 (legacy)
            engine = builder.build_engine(network, config)
            if engine is None:
                print("❌ 엔진 빌드 실패")
                return False
            serialized_engine = engine.serialize()
        
        build_time = time.time() - start_time
        print(f"✅ 엔진 빌드 완료 (소요 시간: {build_time:.2f}초)")
        
        if serialized_engine is None:
            print("❌ 엔진 빌드 실패")
            return False
        
        # 엔진 저장
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        
        print(f"✅ TensorRT 엔진 저장 완료: {engine_path}")
        
        # 엔진 크기 출력
        engine_size = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"엔진 파일 크기: {engine_size:.2f} MB")
        
        return True
    
    def _verify_onnx_model(self, onnx_path):
        """ONNX 모델 검증"""
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX 모델 검증 성공")
        except Exception as e:
            print(f"❌ ONNX 모델 검증 실패: {e}")
            raise
    
    def _print_network_info(self, network):
        """네트워크 정보 출력"""
        print(f"네트워크 정보:")
        print(f"  입력 개수: {network.num_inputs}")
        print(f"  출력 개수: {network.num_outputs}")
        print(f"  레이어 개수: {network.num_layers}")
        
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            print(f"  입력 {i}: {input_tensor.name}, shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        
        for i in range(network.num_outputs):
            output_tensor = network.get_output(i)
            print(f"  출력 {i}: {output_tensor.name}, shape: {output_tensor.shape}, dtype: {output_tensor.dtype}")

class TensorRTEngineFixed:
    """TensorRT 엔진 실행 클래스 (최신 API 호환)"""
    
    def __init__(self, engine_path):
        """TensorRT 엔진 로드"""
        
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 엔진 파일 로드
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        # 런타임 생성 및 엔진 역직렬화
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        if self.engine is None:
            raise RuntimeError(f"TensorRT 엔진 로드 실패: {engine_path}")
        
        # 실행 컨텍스트 생성
        self.context = self.engine.create_execution_context()
        
        # 바인딩 정보 확인 (API 버전 호환)
        self.input_binding = None
        self.output_binding = None
        
        # TensorRT 10.x+ 버전 호환
        if hasattr(self.engine, 'num_io_tensors'):
            # 새로운 API
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_binding = i
                    self.input_name = name
                else:
                    self.output_binding = i
                    self.output_name = name
        else:
            # 기존 API
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                is_input = self.engine.binding_is_input(i)
                
                if is_input:
                    self.input_binding = i
                    self.input_name = name
                else:
                    self.output_binding = i
                    self.output_name = name
        
        print(f"✅ TensorRT 엔진 로드 완료")
        print(f"입력: {self.input_name} (binding {self.input_binding})")
        print(f"출력: {self.output_name} (binding {self.output_binding})")
        
        # 메모리 할당은 실제 추론 시 동적으로 수행
        self.d_input = None
        self.d_output = None
        self.stream = cuda.Stream()
    
    def infer(self, input_data):
        """추론 실행 (API 버전 호환)"""
        
        # 입력 데이터 검증
        if not isinstance(input_data, np.ndarray):
            raise TypeError("입력 데이터는 numpy array여야 합니다")
        
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        batch_size = input_data.shape[0]
        
        # 동적 shape 설정 (API 버전 호환)
        if hasattr(self.engine, 'num_io_tensors'):
            # TensorRT 10.x+ 새로운 API
            self.context.set_input_shape(self.input_name, input_data.shape)
            output_shape = self.context.get_tensor_shape(self.output_name)
        else:
            # 기존 API
            if hasattr(self.engine, 'has_implicit_batch_dimension') and not self.engine.has_implicit_batch_dimension:
                self.context.set_binding_shape(self.input_binding, input_data.shape)
                output_shape = self.context.get_binding_shape(self.output_binding)
            else:
                # Legacy implicit batch mode
                output_shape = (batch_size,) + tuple(self.engine.get_binding_shape(self.output_binding))
        
        # GPU 메모리 할당 (필요시에만) - PyCUDA 타입 변환 문제 해결
        input_size = int(input_data.nbytes)  # numpy.int64 → int 변환
        output_size = int(np.prod(output_shape) * 4)  # float32 = 4 bytes, int로 변환
        
        if self.d_input is None or input_size != getattr(self, '_last_input_size', 0):
            if self.d_input is not None:
                self.d_input.free()
            self.d_input = cuda.mem_alloc(input_size)
            self._last_input_size = input_size
        
        if self.d_output is None or output_size != getattr(self, '_last_output_size', 0):
            if self.d_output is not None:
                self.d_output.free()
            self.d_output = cuda.mem_alloc(output_size)
            self._last_output_size = output_size
        
        # 입력 데이터를 GPU로 복사
        cuda.memcpy_htod_async(self.d_input, input_data.ravel(), self.stream)
        
        # 추론 실행 (API 버전 호환)
        if hasattr(self.engine, 'num_io_tensors'):
            # TensorRT 10.x+ 새로운 API
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))
            success = self.context.execute_async_v3(self.stream.handle)
        else:
            # 기존 API
            bindings = [int(self.d_input), int(self.d_output)]
            
            if hasattr(self.engine, 'has_implicit_batch_dimension') and self.engine.has_implicit_batch_dimension:
                success = self.context.execute_async(batch_size, bindings, self.stream.handle)
            else:
                success = self.context.execute_async_v2(bindings, self.stream.handle)
        
        if not success:
            raise RuntimeError("TensorRT 추론 실행 실패")
        
        # 결과를 CPU로 복사
        output = np.empty(output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        
        # 스트림 동기화
        self.stream.synchronize()
        
        return output
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'd_input') and self.d_input:
            self.d_input.free()
        if hasattr(self, 'd_output') and self.d_output:
            self.d_output.free()

def compare_onnx_tensorrt_fixed(onnx_path, engine_path, test_input):
    """ONNX Runtime과 TensorRT 결과 비교 (수정된 버전)"""
    
    print("=== ONNX Runtime vs TensorRT 결과 비교 ===")
    
    try:
        # ONNX Runtime 추론
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        start_time = time.time()
        onnx_result = session.run([output_name], {input_name: test_input})
        onnx_time = time.time() - start_time
        
        # TensorRT 추론
        trt_engine = TensorRTEngineFixed(engine_path)
        
        start_time = time.time()
        trt_result = trt_engine.infer(test_input)
        trt_time = time.time() - start_time
        
        # 결과 비교
        onnx_output = onnx_result[0]
        diff = np.abs(onnx_output - trt_result)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"ONNX Runtime 시간: {onnx_time * 1000:.2f} ms")
        print(f"TensorRT 시간: {trt_time * 1000:.2f} ms")
        print(f"속도 향상: {onnx_time / trt_time:.2f}x")
        print(f"최대 차이: {max_diff:.6f}")
        print(f"평균 차이: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            print("✅ ONNX Runtime과 TensorRT 결과가 일치합니다!")
        else:
            print("⚠️ ONNX Runtime과 TensorRT 결과에 차이가 있습니다.")
        
        return max_diff < 1e-3
        
    except Exception as e:
        print(f"❌ 비교 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_tensorrt_fixed(engine_path, input_shape, num_runs=100):
    """TensorRT 추론 성능 벤치마크 (수정된 버전)"""
    
    print(f"=== TensorRT 추론 성능 테스트 ({num_runs}회) ===")
    
    try:
        # TensorRT 엔진 로드
        trt_engine = TensorRTEngineFixed(engine_path)
        
        # 테스트 데이터 생성
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # 워밍업
        for _ in range(10):
            _ = trt_engine.infer(test_input)
        
        # 성능 측정
        times = []
        for i in range(num_runs):
            start_time = time.time()
            _ = trt_engine.infer(test_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # ms
        
        # 통계 계산
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        
        print(f"평균 추론 시간: {avg_time:.2f} ms")
        print(f"최소 추론 시간: {min_time:.2f} ms")
        print(f"최대 추론 시간: {max_time:.2f} ms")
        print(f"표준편차: {std_time:.2f} ms")
        print(f"초당 추론 횟수: {1000/avg_time:.1f} FPS")
        
    except Exception as e:
        print(f"❌ 벤치마크 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 실행 함수"""
    
    # 설정
    onnx_dir = "./onnx_output"
    engine_dir = "./tensorrt_output"
    onnx_path = os.path.join(onnx_dir, "litemono.onnx")
    engine_path = os.path.join(engine_dir, "litemono.engine")
    
    # 출력 디렉토리 생성
    os.makedirs(engine_dir, exist_ok=True)
    
    try:
        # ONNX 파일 존재 확인
        if not os.path.exists(onnx_path):
            print(f"❌ ONNX 파일을 찾을 수 없습니다: {onnx_path}")
            print("먼저 PyTorch → ONNX 변환을 실행하세요.")
            return
        
        # 1. ONNX → TensorRT 변환
        print("=== ONNX → TensorRT 엔진 변환 ===")
        converter = ONNXToTensorRTFixed()
        
        success = converter.build_engine_from_onnx(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision='fp16',        # 'fp32', 'fp16', 'int8'
            max_batch_size=8,        # 최대 배치 크기
            max_workspace_size=1 << 30  # 1GB
        )
        
        if not success:
            print("❌ TensorRT 엔진 변환 실패")
            return
        
        # 2. 테스트 데이터 생성
        # ONNX 모델에서 입력 크기 확인
        session = ort.InferenceSession(onnx_path)
        input_shape = session.get_inputs()[0].shape
        
        # 동적 배치 크기 처리
        if input_shape[0] == -1 or isinstance(input_shape[0], str):
            test_shape = (2, input_shape[1], input_shape[2], input_shape[3])
        else:
            test_shape = tuple(input_shape)
        
        test_input = np.random.randn(*test_shape).astype(np.float32)
        
        print(f"테스트 입력 shape: {test_input.shape}")
        
        # 3. ONNX vs TensorRT 결과 비교
        compare_onnx_tensorrt_fixed(onnx_path, engine_path, test_input)
        
        # 4. TensorRT 성능 벤치마크
        benchmark_tensorrt_fixed(engine_path, test_input.shape)
        
        print(f"\n=== ✅ 변환 완료! ===")
        print(f"ONNX 파일: {onnx_path}")
        print(f"TensorRT 엔진: {engine_path}")
        
        # 파일 크기 비교
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        engine_size = os.path.getsize(engine_path) / (1024 * 1024)
        
        print(f"\n파일 크기 비교:")
        print(f"  ONNX: {onnx_size:.2f} MB")
        print(f"  TensorRT: {engine_size:.2f} MB")
        print(f"  압축률: {onnx_size/engine_size:.2f}x")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()




# region ONNX / RT 밴치마크
# """
# PyCUDA 타입 오류 완전 해결 버전
# numpy.int64 → int 변환 문제 해결
# """

# import tensorrt as trt
# import numpy as np
# import os
# import time
# import pycuda.driver as cuda
# import pycuda.autoinit

# import onnx
# import onnxruntime as ort

# class TensorRTEngineFixed:
#     """PyCUDA 타입 오류가 해결된 TensorRT 엔진 클래스"""
    
#     def __init__(self, engine_path):
#         """TensorRT 엔진 로드"""
        
#         self.logger = trt.Logger(trt.Logger.WARNING)
        
#         # 엔진 파일 로드
#         with open(engine_path, 'rb') as f:
#             engine_data = f.read()
        
#         # 런타임 생성 및 엔진 역직렬화
#         runtime = trt.Runtime(self.logger)
#         self.engine = runtime.deserialize_cuda_engine(engine_data)
        
#         if self.engine is None:
#             raise RuntimeError(f"TensorRT 엔진 로드 실패: {engine_path}")
        
#         # 실행 컨텍스트 생성
#         self.context = self.engine.create_execution_context()
        
#         # 바인딩 정보 확인
#         self._setup_bindings()
        
#         # 메모리 할당은 실제 추론 시 동적으로 수행
#         self.d_input = None
#         self.d_output = None
#         self.stream = cuda.Stream()
        
#         print(f"✅ TensorRT 엔진 로드 완료")
#         print(f"입력: {self.input_name} (binding {self.input_binding})")
#         print(f"출력: {self.output_name} (binding {self.output_binding})")
    
#     def _setup_bindings(self):
#         """바인딩 설정 (버전 호환)"""
#         self.input_binding = None
#         self.output_binding = None
        
#         if hasattr(self.engine, 'num_io_tensors'):
#             # TensorRT 10.x+
#             for i in range(self.engine.num_io_tensors):
#                 name = self.engine.get_tensor_name(i)
#                 if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
#                     self.input_binding = i
#                     self.input_name = name
#                 else:
#                     self.output_binding = i
#                     self.output_name = name
#         else:
#             # TensorRT 8.x/9.x
#             for i in range(self.engine.num_bindings):
#                 name = self.engine.get_binding_name(i)
#                 if self.engine.binding_is_input(i):
#                     self.input_binding = i
#                     self.input_name = name
#                 else:
#                     self.output_binding = i
#                     self.output_name = name
    
#     def infer(self, input_data):
#         """추론 실행 (PyCUDA 타입 오류 해결)"""
        
#         # 입력 데이터 검증
#         if not isinstance(input_data, np.ndarray):
#             raise TypeError("입력 데이터는 numpy array여야 합니다")
        
#         if input_data.dtype != np.float32:
#             input_data = input_data.astype(np.float32)
        
#         batch_size = input_data.shape[0]
        
#         # 동적 shape 설정
#         if hasattr(self.engine, 'num_io_tensors'):
#             # TensorRT 10.x+
#             self.context.set_input_shape(self.input_name, input_data.shape)
#             output_shape = self.context.get_tensor_shape(self.output_name)
#         else:
#             # TensorRT 8.x/9.x
#             if hasattr(self.engine, 'has_implicit_batch_dimension') and not self.engine.has_implicit_batch_dimension:
#                 self.context.set_binding_shape(self.input_binding, input_data.shape)
#                 output_shape = self.context.get_binding_shape(self.output_binding)
#             else:
#                 # Legacy implicit batch mode
#                 output_shape = (batch_size,) + tuple(self.engine.get_binding_shape(self.output_binding))
        
#         # GPU 메모리 할당 (PyCUDA 타입 변환 문제 해결)
#         input_size = int(input_data.nbytes)  # 명시적으로 int 변환
#         output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)  # 명시적으로 int 변환
        
#         # 디버그 정보
#         print(f"디버그 - input_size: {input_size} (type: {type(input_size)})")
#         print(f"디버그 - output_size: {output_size} (type: {type(output_size)})")
        
#         # 메모리 할당 (필요시에만)
#         if self.d_input is None or input_size != getattr(self, '_last_input_size', 0):
#             if self.d_input is not None:
#                 self.d_input.free()
#             try:
#                 self.d_input = cuda.mem_alloc(input_size)
#                 self._last_input_size = input_size
#                 print(f"✅ 입력 메모리 할당 성공: {input_size} bytes")
#             except Exception as e:
#                 print(f"❌ 입력 메모리 할당 실패: {e}")
#                 raise
        
#         if self.d_output is None or output_size != getattr(self, '_last_output_size', 0):
#             if self.d_output is not None:
#                 self.d_output.free()
#             try:
#                 self.d_output = cuda.mem_alloc(output_size)
#                 self._last_output_size = output_size
#                 print(f"✅ 출력 메모리 할당 성공: {output_size} bytes")
#             except Exception as e:
#                 print(f"❌ 출력 메모리 할당 실패: {e}")
#                 raise
        
#         # 입력 데이터를 GPU로 복사
#         cuda.memcpy_htod_async(self.d_input, input_data.ravel(), self.stream)
        
#         # 추론 실행 (버전별 호환)
#         if hasattr(self.engine, 'num_io_tensors'):
#             # TensorRT 10.x+
#             self.context.set_tensor_address(self.input_name, int(self.d_input))
#             self.context.set_tensor_address(self.output_name, int(self.d_output))
#             success = self.context.execute_async_v3(self.stream.handle)
#         else:
#             # TensorRT 8.x/9.x
#             bindings = [int(self.d_input), int(self.d_output)]
            
#             if hasattr(self.engine, 'has_implicit_batch_dimension') and self.engine.has_implicit_batch_dimension:
#                 success = self.context.execute_async(batch_size, bindings, self.stream.handle)
#             else:
#                 success = self.context.execute_async_v2(bindings, self.stream.handle)
        
#         if not success:
#             raise RuntimeError("TensorRT 추론 실행 실패")
        
#         # 결과를 CPU로 복사
#         output = np.empty(output_shape, dtype=np.float32)
#         cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        
#         # 스트림 동기화
#         self.stream.synchronize()
        
#         return output
    
#     def __del__(self):
#         """리소스 정리"""
#         if hasattr(self, 'd_input') and self.d_input:
#             self.d_input.free()
#         if hasattr(self, 'd_output') and self.d_output:
#             self.d_output.free()

# def safe_mem_alloc_test():
#     """메모리 할당 테스트"""
#     print("=== PyCUDA 메모리 할당 테스트 ===")
    
#     # 다양한 크기와 타입으로 테스트
#     test_sizes = [
#         1024,                    # int
#         np.int32(2048),         # numpy.int32
#         np.int64(4096),         # numpy.int64
#         int(np.int64(8192)),    # int(numpy.int64)
#     ]
    
#     for i, size in enumerate(test_sizes):
#         try:
#             print(f"테스트 {i+1}: size={size}, type={type(size)}")
#             mem = cuda.mem_alloc(size)
#             print(f"  ✅ 성공: {size} bytes 할당")
#             mem.free()
#         except Exception as e:
#             print(f"  ❌ 실패: {e}")
#             # 타입 변환 시도
#             try:
#                 safe_size = int(size)
#                 mem = cuda.mem_alloc(safe_size)
#                 print(f"  ✅ 타입 변환 후 성공: {safe_size} bytes 할당")
#                 mem.free()
#             except Exception as e2:
#                 print(f"  ❌ 타입 변환 후에도 실패: {e2}")

# def compare_onnx_tensorrt_safe(onnx_path, engine_path, test_input):
#     """안전한 ONNX vs TensorRT 비교 (PyCUDA 오류 해결)"""
    
#     print("=== ONNX Runtime vs TensorRT 결과 비교 (안전 버전) ===")
    
#     try:
#         # ONNX Runtime 추론
#         session = ort.InferenceSession(onnx_path)
#         input_name = session.get_inputs()[0].name
#         output_name = session.get_outputs()[0].name
        
#         start_time = time.time()
#         onnx_result = session.run([output_name], {input_name: test_input})
#         onnx_time = time.time() - start_time
        
#         # TensorRT 추론
#         trt_engine = TensorRTEngineFixed(engine_path)
        
#         start_time = time.time()
#         trt_result = trt_engine.infer(test_input)
#         trt_time = time.time() - start_time
        
#         # 결과 비교
#         onnx_output = onnx_result[0]
#         diff = np.abs(onnx_output - trt_result)
#         max_diff = np.max(diff)
#         mean_diff = np.mean(diff)
        
#         print(f"ONNX Runtime 시간: {onnx_time * 1000:.2f} ms")
#         print(f"TensorRT 시간: {trt_time * 1000:.2f} ms")
#         print(f"속도 향상: {onnx_time / trt_time:.2f}x")
#         print(f"최대 차이: {max_diff:.6f}")
#         print(f"평균 차이: {mean_diff:.6f}")
        
#         if max_diff < 1e-3:
#             print("✅ ONNX Runtime과 TensorRT 결과가 일치합니다!")
#         else:
#             print("⚠️ ONNX Runtime과 TensorRT 결과에 차이가 있습니다.")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ 비교 중 오류 발생: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def benchmark_tensorrt_safe(engine_path, input_shape, num_runs=50):
#     """안전한 TensorRT 벤치마크"""
    
#     print(f"=== TensorRT 추론 성능 테스트 (안전 버전, {num_runs}회) ===")
    
#     try:
#         # TensorRT 엔진 로드
#         trt_engine = TensorRTEngineFixed(engine_path)
        
#         # 테스트 데이터 생성
#         test_input = np.random.randn(*input_shape).astype(np.float32)
        
#         # 워밍업 (적게)
#         print("워밍업 중...")
#         for i in range(3):
#             _ = trt_engine.infer(test_input)
#             print(f"  워밍업 {i+1}/3 완료")
        
#         # 성능 측정
#         print(f"성능 측정 시작 ({num_runs}회)...")
#         times = []
#         for i in range(num_runs):
#             start_time = time.time()
#             _ = trt_engine.infer(test_input)
#             end_time = time.time()
#             times.append((end_time - start_time) * 1000)  # ms
            
#             if (i + 1) % 10 == 0:
#                 print(f"  진행: {i+1}/{num_runs}")
        
#         # 통계 계산
#         avg_time = np.mean(times)
#         min_time = np.min(times)
#         max_time = np.max(times)
#         std_time = np.std(times)
        
#         print(f"평균 추론 시간: {avg_time:.2f} ms")
#         print(f"최소 추론 시간: {min_time:.2f} ms")
#         print(f"최대 추론 시간: {max_time:.2f} ms")
#         print(f"표준편차: {std_time:.2f} ms")
#         print(f"초당 추론 횟수: {1000/avg_time:.1f} FPS")
        
#     except Exception as e:
#         print(f"❌ 벤치마크 중 오류 발생: {e}")
#         import traceback
#         traceback.print_exc()

# def main():
#     """메인 함수 (PyCUDA 오류 해결)"""
    
#     # 먼저 메모리 할당 테스트
#     safe_mem_alloc_test()
    
#     # 설정
#     onnx_dir = "./onnx_output"
#     engine_dir = "./tensorrt_output"
#     onnx_path = os.path.join(onnx_dir, "litemono.onnx")
#     engine_path = os.path.join(engine_dir, "litemono.engine")
    
#     try:
#         # 파일 존재 확인
#         if not os.path.exists(engine_path):
#             print(f"❌ TensorRT 엔진 파일을 찾을 수 없습니다: {engine_path}")
#             return
        
#         if not os.path.exists(onnx_path):
#             print(f"❌ ONNX 파일을 찾을 수 없습니다: {onnx_path}")
#             return
        
#         # 테스트 데이터 생성
#         session = ort.InferenceSession(onnx_path)
#         input_shape = session.get_inputs()[0].shape
        
#         # 동적 배치 크기 처리
#         if input_shape[0] == -1 or isinstance(input_shape[0], str):
#             test_shape = (1, input_shape[1], input_shape[2], input_shape[3])  # 배치 크기 1로 시작
#         else:
#             test_shape = tuple(input_shape)
        
#         test_input = np.random.randn(*test_shape).astype(np.float32)
        
#         print(f"테스트 입력 shape: {test_input.shape}")
        
#         # ONNX vs TensorRT 결과 비교
#         compare_onnx_tensorrt_safe(onnx_path, engine_path, test_input)
        
#         # TensorRT 성능 벤치마크 (더 안전하게)
#         benchmark_tensorrt_safe(engine_path, test_input.shape, num_runs=30)
        
#         print(f"\n=== ✅ 테스트 완료! ===")
        
#     except Exception as e:
#         print(f"❌ 오류 발생: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()