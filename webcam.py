"""
젯슨 나노용 실시간 깊이 추정 (TensorRT 10.x)
입력/출력 모두 192x640으로 통일
"""

import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.cm as cm

class RealtimeDepthEstimatorTRT10:
    """실시간 깊이 추정 클래스 (TensorRT 10.x 최적화)"""
    
    def __init__(self, engine_path):
        """TensorRT 엔진 로드"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 엔진 로드
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # TensorRT 10.x 바인딩 설정
        self._setup_bindings()
        
        # 입출력 크기 (고정 - 192x640)
        self.input_shape = (1, 3, 192, 640)
        self.output_shape = (1, 1, 192, 640)
        
        # GPU 메모리 할당
        input_size = int(np.prod(self.input_shape) * 4)  # float32
        output_size = int(np.prod(self.output_shape) * 4)
        
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        self.stream = cuda.Stream()
        
        print(f"✅ TensorRT 10.x 엔진 로드 완료")
        print(f"입력: {self.input_name} - {self.input_shape}")
        print(f"출력: {self.output_name} - {self.output_shape}")
        
    def _setup_bindings(self):
        """TensorRT 10.x 바인딩 설정"""
        # TensorRT 10.x는 num_io_tensors 사용
        if hasattr(self.engine, 'num_io_tensors'):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                
                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                else:
                    self.output_name = name
        else:
            # Fallback for older versions
            self.input_name = "input"
            self.output_name = "disparity"
    
    def preprocess_frame(self, frame):
        """프레임 전처리 (192x640으로 직접 리사이즈)"""
        # 웹캠 프레임을 바로 192x640으로 리사이즈
        resized = cv2.resize(frame, (640, 192))
        
        # BGR → RGB 변환
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 정규화 및 차원 변경
        normalized = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(normalized, (2, 0, 1))  # (H,W,C) → (C,H,W)
        batch = np.expand_dims(tensor, axis=0)  # 배치 차원 추가
        
        return batch
    
    def predict(self, input_data):
        """TensorRT 10.x 추론"""
        # GPU로 데이터 복사
        cuda.memcpy_htod_async(self.d_input, input_data.ravel(), self.stream)
        
        # TensorRT 10.x 추론 실행
        if hasattr(self.engine, 'num_io_tensors'):
            # TensorRT 10.x 방식
            self.context.set_tensor_address(self.input_name, int(self.d_input))
            self.context.set_tensor_address(self.output_name, int(self.d_output))
            success = self.context.execute_async_v3(self.stream.handle)
        else:
            # Fallback
            bindings = [int(self.d_input), int(self.d_output)]
            success = self.context.execute_async_v2(bindings, self.stream.handle)
        
        if not success:
            raise RuntimeError("TensorRT 추론 실행 실패")
        
        # CPU로 결과 복사
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return output
    
    def postprocess_disparity(self, disparity):
        """Disparity 후처리 (192x640 그대로 유지)"""
        # 첫 번째 배치, 첫 번째 채널 선택
        disp = disparity[0, 0]  # (192, 640)
        
        # 컬러맵 적용 (magma)
        normalized = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
        colored = cm.magma(normalized)[:, :, :3]  # RGB만 사용
        colored_uint8 = (colored * 255).astype(np.uint8)
        
        # BGR로 변환 (OpenCV 표시용)
        bgr = cv2.cvtColor(colored_uint8, cv2.COLOR_RGB2BGR)
        
        return bgr  # (192, 640, 3)

def simple_main():
    """초간단 버전 - 젯슨 나노 TensorRT 10.x용"""
    
    ENGINE_PATH = "./tensorrt_output/litemono.engine"
    
    # 모델 로드
    try:
        estimator = RealtimeDepthEstimatorTRT10(ENGINE_PATH)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 웹캠 초기화 (더 작은 해상도로 시작)
    cap = cv2.VideoCapture(0)
    
    # 젯슨 나노 최적화 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # FPS 제한으로 안정성 향상
    
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다")
        return
    
    print("✅ 웹캠 초기화 완료")
    print("실시간 깊이 추정 시작 (ESC로 종료)")
    print("입력/출력: 192x640")
    
    # 성능 측정용
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다")
            break
        
        try:
            # 추론 시작 시간
            start_time = time.time()
            
            # 전처리 (192x640으로 리사이즈)
            input_tensor = estimator.preprocess_frame(frame)
            
            # TensorRT 추론
            disparity = estimator.predict(input_tensor)
            
            # 후처리 (192x640 그대로)
            depth_img = estimator.postprocess_disparity(disparity)
            
            # 추론 시간 계산
            inference_time = time.time() - start_time
            total_time += inference_time
            frame_count += 1
            
            # 원본 프레임도 192x640으로 리사이즈
            original_resized = cv2.resize(frame, (640, 192))
            
            # 좌우로 배치 (원본 | 깊이)
            combined = np.hstack([original_resized, depth_img])
            
            # 성능 정보 오버레이
            avg_time = total_time / frame_count if frame_count > 0 else 0
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            cv2.putText(combined, f"Original (192x640)", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(combined, f"Depth (192x640)", (650, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(combined, f"Inference: {inference_time*1000:.1f}ms", 
                       (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(combined, f"Avg FPS: {fps:.1f}", 
                       (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 화면 표시
            cv2.imshow('Depth Estimation (192x640)', combined)
            
        except Exception as e:
            print(f"⚠️ 처리 오류: {e}")
            continue
        
        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    # 최종 성능 통계
    if frame_count > 0:
        avg_fps = frame_count / total_time
        avg_inference_time = (total_time / frame_count) * 1000
        print(f"\n=== 성능 통계 ===")
        print(f"처리된 프레임: {frame_count}")
        print(f"평균 FPS: {avg_fps:.2f}")
        print(f"평균 추론 시간: {avg_inference_time:.1f}ms")
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
    print("프로그램 종료")

def full_screen_main():
    """풀스크린 깊이 맵 표시 버전"""
    
    ENGINE_PATH = "./tensorrt_output/litemono.engine"
    
    # 모델 로드
    try:
        estimator = RealtimeDepthEstimatorTRT10(ENGINE_PATH)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 웹캠
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print("풀스크린 깊이 맵 표시 (ESC로 종료)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        try:
            # 추론
            start_time = time.time()
            input_tensor = estimator.preprocess_frame(frame)
            disparity = estimator.predict(input_tensor)
            depth_img = estimator.postprocess_disparity(disparity)
            inference_time = time.time() - start_time
            
            # 깊이 맵을 더 크게 표시 (2배 확대)
            depth_large = cv2.resize(depth_img, (1280, 384))
            
            # 성능 정보
            cv2.putText(depth_large, f"Depth Map (192x640 -> 1280x384)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(depth_large, f"Inference: {inference_time*1000:.1f}ms", 
                       (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Full Screen Depth', depth_large)
            
        except Exception as e:
            print(f"⚠️ 처리 오류: {e}")
            continue
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 기본 버전 (좌우 분할)
    simple_main()
    
    # 또는 풀스크린 깊이 맵
    # full_screen_main()