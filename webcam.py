"""
젯슨 나노용 실시간 깊이 추정 (웹캠)
TensorRT + OpenCV로 간단하고 빠르게 구현
"""

import cv2
import numpy as np
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import matplotlib.cm as cm

class RealtimeDepthEstimator:
    """실시간 깊이 추정 클래스 (젯슨 나노 최적화)"""
    
    def __init__(self, engine_path):
        """TensorRT 엔진 로드"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # 엔진 로드
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # 바인딩 설정
        self.input_binding = 0
        self.output_binding = 1
        
        # 입출력 크기 (고정)
        self.input_shape = (1, 3, 192, 640)
        self.output_shape = (1, 1, 192, 640)
        
        # GPU 메모리 할당
        input_size = int(np.prod(self.input_shape) * 4)
        output_size = int(np.prod(self.output_shape) * 4)
        
        self.d_input = cuda.mem_alloc(input_size)
        self.d_output = cuda.mem_alloc(output_size)
        self.stream = cuda.Stream()
        
        print(f"✅ TensorRT 엔진 로드 완료")
        print(f"입력 크기: {self.input_shape}")
        print(f"출력 크기: {self.output_shape}")
    
    def preprocess_frame(self, frame):
        """프레임 전처리 (최적화됨)"""
        # 640x192로 리사이즈
        resized = cv2.resize(frame, (640, 192))
        
        # BGR → RGB 변환
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 정규화 및 차원 변경
        normalized = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(normalized, (2, 0, 1))  # (H,W,C) → (C,H,W)
        batch = np.expand_dims(tensor, axis=0)  # 배치 차원 추가
        
        return batch
    
    def predict(self, input_data):
        """TensorRT 추론 (최적화됨)"""
        # GPU로 데이터 복사
        cuda.memcpy_htod_async(self.d_input, input_data.ravel(), self.stream)
        
        # 추론 실행
        bindings = [int(self.d_input), int(self.d_output)]
        self.context.execute_async_v2(bindings, self.stream.handle)
        
        # CPU로 결과 복사
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return output
    
    def postprocess_disparity(self, disparity, target_size):
        """Disparity 후처리 및 시각화"""
        # 첫 번째 배치, 첫 번째 채널 선택
        disp = disparity[0, 0]  # (192, 640)
        
        # 컬러맵 적용
        normalized = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)
        colored = cm.magma(normalized)[:, :, :3]  # RGB만 사용
        colored_uint8 = (colored * 255).astype(np.uint8)
        
        # 원본 크기로 리사이즈
        resized = cv2.resize(colored_uint8, target_size)
        
        # BGR로 변환 (OpenCV 표시용)
        bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
        
        return bgr

def main():
    """메인 함수 - 실시간 깊이 추정"""
    
    # 설정
    ENGINE_PATH = "./tensorrt_output/litemono.engine"
    CAMERA_ID = 0  # 웹캠 ID (보통 0)
    
    # TensorRT 모델 로드
    try:
        depth_estimator = RealtimeDepthEstimator(ENGINE_PATH)
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # 젯슨 나노 최적화 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ 웹캠을 열 수 없습니다")
        return
    
    print("✅ 웹캠 초기화 완료")
    print("ESC 키를 눌러 종료하세요")
    
    # FPS 계산용
    fps_counter = 0
    fps_start_time = time.time()
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다")
            break
        
        try:
            # 추론 시작 시간
            start_time = time.time()
            
            # 전처리
            input_tensor = depth_estimator.preprocess_frame(frame)
            
            # TensorRT 추론
            disparity = depth_estimator.predict(input_tensor)
            
            # 후처리 및 시각화
            depth_colored = depth_estimator.postprocess_disparity(
                disparity, (frame.shape[1], frame.shape[0])
            )
            
            # 추론 시간 계산
            inference_time = time.time() - start_time
            
            # 결과 화면 구성
            # 원본 프레임 크기 조정
            original_resized = cv2.resize(frame, (320, 240))
            depth_resized = cv2.resize(depth_colored, (320, 240))
            
            # 상하로 결합
            combined = np.vstack([original_resized, depth_resized])
            
            # 텍스트 오버레이
            cv2.putText(combined, f"Original", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined, f"Depth", (10, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(combined, f"Inference: {inference_time*1000:.1f}ms", 
                       (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # FPS 계산 및 표시
            fps_counter += 1
            if fps_counter % 30 == 0:  # 30프레임마다 FPS 업데이트
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                
            if fps_counter > 30:
                cv2.putText(combined, f"FPS: {fps:.1f}", 
                           (200, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 화면 표시
            cv2.imshow('Real-time Depth Estimation', combined)
            
        except Exception as e:
            print(f"⚠️ 처리 오류: {e}")
            continue
        
        # ESC 키로 종료
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    # 정리
    cap.release()
    cv2.destroyAllWindows()
    print("프로그램 종료")

# 젯슨 나노 전용 간단 버전
def simple_main():
    """초간단 버전 - 젯슨 나노용"""
    
    ENGINE_PATH = "./tensorrt_output/litemono.engine"
    
    # 모델 로드
    estimator = RealtimeDepthEstimator(ENGINE_PATH)
    
    # 웹캠
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("실시간 깊이 추정 시작 (ESC로 종료)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 추론
        input_tensor = estimator.preprocess_frame(frame)
        disparity = estimator.predict(input_tensor)
        
        # 시각화
        depth_img = estimator.postprocess_disparity(disparity, (640, 480))
        
        # 표시 (원본과 깊이를 좌우로 배치)
        combined = np.hstack([cv2.resize(frame, (320, 240)), 
                             cv2.resize(depth_img, (320, 240))])
        
        cv2.imshow('Depth', combined)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 간단 버전 사용 (젯슨 나노 권장)
    simple_main()
    
    # 또는 전체 버전
    # main()