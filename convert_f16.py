import torch
import tensorrt as trt
import numpy as np
import os
import time


import networks

def simple_fp16_conversion():

    
    print(" TensorRT 변환 시작")
    print(f"TensorRT 버전: {trt.__version__}")
    
    # # 1. 모델 로드
    # weights_folder = "./liteweight"
    # encoder_path = os.path.join(weights_folder, "encoder.pth")
    # decoder_path = os.path.join(weights_folder, "depth.pth")
    
    # encoder_dict = torch.load(encoder_path, map_location='cuda')
    # decoder_dict = torch.load(decoder_path, map_location='cuda')
    
    # feed_height = encoder_dict['height']
    # feed_width = encoder_dict['width']
    
    # print(f"모델 입력 크기: {feed_width} x {feed_height}")
    
    # # 2. 모델 생성
    # encoder = networks.LiteMono(
    #     model="lite-mono",
    #     height=feed_height,
    #     width=feed_width
    # )
    # encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    
    # depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    # depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_decoder.state_dict()})
    
    # class SimpleModel(torch.nn.Module):
    #     def __init__(self, encoder, decoder):
    #         super().__init__()
    #         self.encoder = encoder
    #         self.decoder = decoder
        
    #     def forward(self, x):
    #         features = self.encoder(x)
    #         outputs = self.decoder(features)
    #         return outputs[("disp", 0)]
    
    # model = SimpleModel(encoder, depth_decoder).cuda().eval()
    
    # # 3. ONNX 변환
    # dummy_input = torch.randn(1, 3, feed_height, feed_width).cuda()
    onnx_path = "proposal3.onnx"
    
    # print("ONNX 변환 중...")
    # try:
    #     torch.onnx.export(
    #         model,
    #         dummy_input,
    #         onnx_path,
    #         export_params=True,
    #         opset_version=11,  
    #         do_constant_folding=True,
    #         input_names=['input'],
    #         output_names=['output'],
    #         dynamic_axes=None,
    #         verbose=False
    #     )
    #     print("ONNX 변환 성공")
    # except Exception as e:
    #     print(f"ONNX 변환 실패: {e}")
    #     return False
    
    # 4. TensorRT FP16 변환 
    print("TensorRT FP16 변환 중...")
    
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)  # 에러만 출력
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
   
    config = builder.create_builder_config()
    
    # 메모리 설정
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    except:
        config.max_workspace_size = 1 << 30
    
    # FP16 활성화
    config.set_flag(trt.BuilderFlag.FP16)
    print("FP16 활성화")
    
    # GPU 전용
    config.default_device_type = trt.DeviceType.GPU
    
    # ONNX 파싱
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("ONNX 파싱 실패")
            for error in range(parser.num_errors):
                print(f"  {parser.get_error(error)}")
            return False
    
    print("ONNX 파싱 성공")
    
    # 엔진 빌드
    print("FP16 Engine 빌드 중...")
    start_time = time.time()
    
    try:
        # TensorRT 10.x 방식
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("Engine 빌드 실패")
            return False
        
        build_time = time.time() - start_time
        print(f"Engine 빌드 완료 ({build_time:.1f}초)")
        
        # TensorRT 10.x 호환 저장
        engine_path = "simple_fp16_ori.engine"
        
        # IHostMemory 객체 처리 (TensorRT 10.x)
        try:
            # TensorRT 10.x 방식 - bytes() 변환
            engine_data = bytes(serialized_engine)
            
            # 크기 계산 시도
            try:
                size_mb = serialized_engine.size() / 1024 / 1024
            except:
                # size() 메서드가 없는 경우 len() 대신 다른 방법
                size_mb = len(engine_data) / 1024 / 1024
                
        except Exception as convert_error:
            print(f"데이터 변환 오류: {convert_error}")
            # 직접 저장 시도
            engine_data = serialized_engine
            size_mb = 0  # 크기는 나중에 계산
        
        # 파일 저장
        try:
            with open(engine_path, 'wb') as f:
                f.write(engine_data)
            
            # 저장 후 실제 파일 크기 확인
            actual_size = os.path.getsize(engine_path) / 1024 / 1024
            
            print(f"FP16 Engine 저장: {engine_path}")
            print(f"파일 크기: {actual_size:.2f} MB")
            
        except Exception as save_error:
            print(f"파일 저장 실패: {save_error}")
            return False
        
        # 정리
        os.remove(onnx_path)
        
        # 간단 테스트
        test_simple_fp16(engine_path)
        
        return True
        
    except Exception as e:
        build_time = time.time() - start_time
        print(f"Engine 빌드 실패 ({build_time:.1f}초)")
        print(f"오류: {e}")
        return False

def test_simple_fp16(engine_path):
    
    print("FP16 Engine 테스트")
    
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        # 엔진 로드
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # 메모리 할당 
        input_shape = (1, 3, 192, 640)
        output_shape = (1, 1, 192, 640)
        
        input_size = int(np.prod(input_shape) * 4)
        output_size = int(np.prod(output_shape) * 4)
        
        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)
        stream = cuda.Stream()
        
        # 테스트 데이터
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # 10회 추론 테스트
        times = []
        for i in range(10):
            start_time = time.time()
            
            # 데이터 전송
            cuda.memcpy_htod_async(d_input, test_input, stream)
            
            # 추론 (TensorRT 10.x)
            if hasattr(engine, 'num_io_tensors'):
                for j in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(j)
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        context.set_tensor_address(name, int(d_input))
                    else:
                        context.set_tensor_address(name, int(d_output))
                context.execute_async_v3(stream_handle=stream.handle)
            else:
                # Fallback
                bindings = [int(d_input), int(d_output)]
                context.execute_async_v2(bindings, stream.handle)
            
            # 결과 복사
            output = np.empty(output_shape, dtype=np.float32)
            cuda.memcpy_dtoh_async(output, d_output, stream)
            stream.synchronize()
            
            end_time = time.time()
            times.append((end_time - start_time) * 1000)
        
        avg_time = np.mean(times)
        fps = 1000 / avg_time
        
        print(f"FP16 테스트 ")
        print(f"평균 추론 시간: {avg_time:.2f} ms")
        print(f"FPS: {fps:.1f}")
        
        
    except Exception as e:
        print(f"테스트 실패: {e}")

if __name__ == "__main__":
    print("FP16 TensorRT 변환")
    success = simple_fp16_conversion()
    
    if success:
        print("\nP16 변환 성공!")
    else:
        print("\nFP16 변환 실패")