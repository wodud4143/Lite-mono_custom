# import torch
# import torch.nn as nn
# import numpy as np
# import os
# import time
# from pathlib import Path

# # 원본 모듈들 임포트
# import networks
# from layers import disp_to_depth

# # TensorRT 관련
# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit

# class LiteMonoTensorRTJetPack:
    
#     def __init__(self, weights_folder, model_type="lite-mono"):
#         self.weights_folder = weights_folder
#         self.model_type = model_type
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # TensorRT 버전 확인
#         print(f"TensorRT 버전: {trt.__version__}")
#         print(f"PyTorch 버전: {torch.__version__}")
        
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

#     def convert_to_tensorrt_engine(self, engine_path, use_fp16=False):
        
        
#         try:
#             # 1. PyTorch → ONNX (임시 파일)
#             temp_onnx_path = "proposal3.onnx"
#             input_shape = (1, 3, self.feed_height, self.feed_width)
            
#             if not self._pytorch_to_onnx(pytorch_model, temp_onnx_path, input_shape):
#                 return False
            
#             # # 2. ONNX → TensorRT Engine
#             success = self._onnx_to_tensorrt_jetpack(temp_onnx_path, engine_path, use_fp16)
            
#             # # 3. 임시 파일 정리
#             if os.path.exists(temp_onnx_path):
#                 os.remove(temp_onnx_path)
#                 print("임시 ONNX 파일 삭제됨")
            
#             return success
            
#         except Exception as e:
#             print(f"TensorRT Engine 변환 실패: {e}")
#             import traceback
#             traceback.print_exc()
#             return False

#     def _pytorch_to_onnx(self, pytorch_model, onnx_path, input_shape):
#         """PyTorch → ONNX 변환"""
        
#         print("1. PyTorch → ONNX 변환 중...")
        
#         try:
#             # 더미 입력 생성
#             dummy_input = torch.randn(input_shape).to(self.device)
            
#             # ONNX 내보내기 
#             torch.onnx.export(
#                 pytorch_model,
#                 dummy_input,
#                 onnx_path,
#                 export_params=True,
#                 opset_version=11,  
#                 do_constant_folding=True,
#                 input_names=['input'],
#                 output_names=['output'],
#                 dynamic_axes=None,  # 고정 크기만 사용 (adaptive pooling 문제)
#                 verbose=False,
#                 keep_initializers_as_inputs=False,
#                 export_modules_as_functions=False
#             )
            
#             print("ONNX 변환 성공")
#             return True
            
#         except Exception as e:
#             print(f"ONNX 변환 실패: {e}")
#             # Adaptive pooling 문제가 있을 경우 opset 다운그레이드
#             try:

#                 torch.onnx.export(
#                     pytorch_model,
#                     dummy_input,
#                     onnx_path,
#                     export_params=True,
#                     opset_version=11,
#                     do_constant_folding=True,
#                     input_names=['input'],
#                     output_names=['output'],
#                     dynamic_axes=None,
#                     verbose=False
#                 )
#                 print("ONNX 변환 성공 (opset 11)")
#                 return True
#             except Exception as e2:
#                 print(f"ONNX 변환 실패: {e2}")
#                 return False

#     def _onnx_to_tensorrt_jetpack(self, onnx_path, engine_path, use_fp16=False):
#         """ONNX → TensorRT Engine"""
        
#         print("2. ONNX → TensorRT Engine 변환 중...")
        
#         try:
#             # TensorRT 10.x 로거
#             TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
#             # 빌더 생성
#             builder = trt.Builder(TRT_LOGGER)
            
#             # 네트워크 정의 생성 (explicit batch)
#             network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#             network = builder.create_network(network_flags)
            
#             # 빌더 설정
#             config = builder.create_builder_config()
            
            
#             try:
#                 # TensorRT 10.x 방식
#                 config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
#                 print("TensorRT 10.x 메모리 설정 사용")
#             except AttributeError:
#                 # 이전 버전 호환
#                 config.max_workspace_size = 1 << 30
#                 print("TensorRT 이전 버전 메모리 설정 사용")
            
            
            
#             # ONNX 파서 생성
#             parser = trt.OnnxParser(network, TRT_LOGGER)
            
#             # ONNX 파일 파싱
#             print("ONNX 파일 파싱 중...")
#             with open(onnx_path, 'rb') as model:
#                 if not parser.parse(model.read()):
#                     print("ONNX 파싱 실패:")
#                     for error in range(parser.num_errors):
#                         error_msg = parser.get_error(error)
#                         print(f"  오류 {error}: {error_msg}")
#                     return False
            
#             print("ONNX 파싱 성공")
            
#             # 프로파일 설정 (JetPack 6.2d에서 권장)
#             profile = builder.create_optimization_profile()
#             input_name = network.get_input(0).name
#             input_shape = (1, 3, self.feed_height, self.feed_width)
            
#             profile.set_shape(input_name, input_shape, input_shape, input_shape)
#             config.add_optimization_profile(profile)
            
#             # 엔진 빌드
#             print("TensorRT Engine 빌드 중...")
#             serialized_engine = builder.build_serialized_network(network, config)
            
#             if serialized_engine is None:
#                 print("Engine 빌드 실패")
#                 return False
            
#             # 엔진 파일 저장
#             with open(engine_path, 'wb') as f:
#                 f.write(serialized_engine)
            
#             print(f"TensorRT Engine 저장 완료: {engine_path}")
#             return True
            
#         except Exception as e:
#             print(f"TensorRT Engine 변환 실패: {e}")
#             import traceback
#             traceback.print_exc()
#             return False

# class JetPackTensorRTInference:
#     """JetPack 6.2d용 TensorRT Engine 추론 클래스"""
    
#     def __init__(self, engine_path):
#         """TensorRT Engine 로드"""
        
#         self.logger = trt.Logger(trt.Logger.WARNING)
        
#         # 엔진 로드
#         print(f"TensorRT Engine 로드 중: {engine_path}")
#         with open(engine_path, 'rb') as f:
#             engine_data = f.read()
        
#         runtime = trt.Runtime(self.logger)
#         self.engine = runtime.deserialize_cuda_engine(engine_data)
        
#         if self.engine is None:
#             raise RuntimeError(f"TensorRT Engine 로드 실패: {engine_path}")
        
#         # 실행 컨텍스트 생성
#         self.context = self.engine.create_execution_context()
        
#         # 메모리 할당
#         self._allocate_buffers()
        
#         print(f"TensorRT Engine 로드 완료")
#         print(f"입력 shape: {self.input_shape}")
#         print(f"출력 shape: {self.output_shape}")
    
#     def _allocate_buffers(self):
#         """GPU 메모리 할당"""
        
#         self.inputs = []
#         self.outputs = []
#         self.bindings = []
#         self.stream = cuda.Stream()
        
#         for i in range(self.engine.num_io_tensors):
#             tensor_name = self.engine.get_tensor_name(i)
            
#             if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
#                 # 입력 텐서
#                 self.input_shape = self.engine.get_tensor_shape(tensor_name)
#                 dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
#                 size = trt.volume(self.input_shape)
                
#                 # 메모리 할당
#                 host_mem = cuda.pagelocked_empty(size, dtype)
#                 device_mem = cuda.mem_alloc(host_mem.nbytes)
                
#                 self.inputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
                
#             else:
#                 # 출력 텐서
#                 self.output_shape = self.engine.get_tensor_shape(tensor_name)
#                 dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
#                 size = trt.volume(self.output_shape)
                
#                 # 메모리 할당
#                 host_mem = cuda.pagelocked_empty(size, dtype)
#                 device_mem = cuda.mem_alloc(host_mem.nbytes)
                
#                 self.outputs.append({'host': host_mem, 'device': device_mem, 'name': tensor_name})
    
#     def infer(self, input_data):
#         """추론 실행 (JetPack 6.2d 호환)"""
        
#         # 입력 데이터를 GPU로 복사
#         np.copyto(self.inputs[0]['host'], input_data.ravel())
#         cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
#         # 텐서 주소 설정 (TensorRT 10.x 방식)
#         for inp in self.inputs:
#             self.context.set_tensor_address(inp['name'], int(inp['device']))
#         for out in self.outputs:
#             self.context.set_tensor_address(out['name'], int(out['device']))
        
#         # 추론 실행
#         self.context.execute_async_v3(stream_handle=self.stream.handle)
        
#         # 결과를 CPU로 복사
#         cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
#         self.stream.synchronize()
        
#         # 결과 반환
#         return self.outputs[0]['host'].reshape(self.output_shape)

# def main():
#     """메인 실행 함수"""
    
#     # 설정
#     weights_folder = "./liteweight"
#     output_dir = "./tensorrt_output"
#     model_type = "lite-mono"
#     use_fp16 = True  # Jetson에서는 FP16 권장
    
#     # 출력 디렉토리 생성
#     os.makedirs(output_dir, exist_ok=True)
    
#     # 파일 경로
#     engine_path = os.path.join(output_dir, "litemono_f32.engine")
    
#     try:
#         # 1. TensorRT Engine 변환기 생성
#         print("=== JetPack 6.2d Lite-Mono TensorRT Engine 변환 시작 ===")
#         converter = LiteMonoTensorRTJetPack(weights_folder, model_type)
        
#         # # 2. PyTorch 모델 생성
#         # print("\n=== PyTorch 모델 생성 ===")
#         # pytorch_model = converter.create_pytorch_model()
#         # print("PyTorch 모델 생성 완료")
        
#         # 3. PyTorch → TensorRT Engine 변환
#         print(f"\n=== PyTorch → TensorRT Engine 변환 ===")
#         success = converter.convert_to_tensorrt_engine(engine_path, use_fp16)
        
#         if not success:
#             print("TensorRT Engine 변환 실패")
#             return
        
#         # 4. 간단한 추론 테스트
#         print(f"\n=== 추론 테스트 ===")
#         try:
#             inference = JetPackTensorRTInference(engine_path)
            
#             # 테스트 데이터
#             test_input = np.random.randn(1, 3, converter.feed_height, converter.feed_width).astype(np.float32)
            
#             # 추론 실행
#             start_time = time.time()
#             result = inference.infer(test_input)
#             end_time = time.time()
            
#             print(f"추론 시간: {(end_time - start_time) * 1000:.2f} ms")
#             print(f"출력 shape: {result.shape}")
#             print("추론 테스트 성공")
            
#         except Exception as e:
#             print(f"추론 테스트 실패: {e}")
        
#         print(f"\n=== 변환 완료===")
#         print(f"TensorRT Engine 파일: {engine_path}")
#         print(f"Engine 파일 크기: {os.path.getsize(engine_path) / (1024*1024):.2f} MB")
        
#     except Exception as e:
#         print(f"오류 발생: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()




import os
import tensorrt as trt

class OnnxToTensorRT:
    """
    ONNX 모델을 TensorRT 엔진으로 변환하는 유틸리티 클래스
    JetPack 6.2d / TRT 10.x 환경에 맞춤
    """
    def __init__(self, workspace_size=1<<30):
        self.logger = trt.Logger(trt.Logger.WARNING)
        # 빌더 설정
        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        # 워크스페이스 메모리 제한 (예: 1GB)
        try:
            self.config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE, workspace_size
            )
        except AttributeError:
            # 이전 TRT 버전 호환
            self.config.max_workspace_size = workspace_size

    def convert(self, onnx_path: str, engine_path: str,
                feed_shape: tuple, use_fp16: bool = False) -> bool:
        """
        :param onnx_path: 변환할 ONNX 파일 경로
        :param engine_path: 저장할 TensorRT 엔진 파일 경로
        :param feed_shape: (batch, channel, height, width) 형식의 입력 크기
        :param use_fp16: FP16 모드 사용 여부
        :return: 변환 성공 여부
        """
        # 네트워크 생성 (explicit batch)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = self.builder.create_network(network_flags)
        parser = trt.OnnxParser(network, self.logger)

        # ONNX 파싱
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("❌ ONNX 파싱 실패")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return False
        print("✅ ONNX 파싱 성공")

        # 최적화 프로파일 설정
        profile = self.builder.create_optimization_profile()
        input_name = network.get_input(0).name
        profile.set_shape(input_name, feed_shape, feed_shape, feed_shape)
        self.config.add_optimization_profile(profile)

        # FP16 설정
        if use_fp16:
            self.config.set_flag(trt.BuilderFlag.FP16)
            print("▶️ FP16 모드 활성화")

        # 엔진 빌드
        print("⏳ TensorRT 엔진 빌드 중…")
        serialized_engine = self.builder.build_serialized_network(network, self.config)
        if serialized_engine is None:
            print("❌ 엔진 빌드 실패")
            return False

        # 파일로 저장
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(serialized_engine)
        print(f"✅ 엔진 저장 완료: {engine_path}")
        return True



def main():
    # ONNX 모델 경로
    onnx_model_path = "proposal3.onnx"
    # 출력할 TensorRT 엔진 경로
    engine_output_path = "./tensorrt_output/liteproposal_f32.engine"
    # 입력 크기 (batch, channel, height, width)
    feed_shape = (1, 3, 192, 640)  # 예: 192×640 해상도

    converter = OnnxToTensorRT(workspace_size=1<<30)
    success = converter.convert(
        onnx_path=onnx_model_path,
        engine_path=engine_output_path,
        feed_shape=feed_shape,
        use_fp16=False
    )

    if success:
        print("🎉 ONNX → TensorRT 변환 완료")
    else:
        print("⚠️ 변환 중 오류 발생")

if __name__ == "__main__":
    main()