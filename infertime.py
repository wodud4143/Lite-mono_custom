import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os.path as osp
from glob import glob
from PIL import Image

# 설정
# ENGINE_PATH = "model.engine"
# IMAGE_PATH = "input.jpg"
# INPUT_SHAPE = (3, 224, 224)  # 예시: (C, H, W)
def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_tensor_shape(binding))
        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        # 페이지 잠금된 호스트 메모리
        host_mem = cuda.pagelocked_empty(size, dtype)
        # 디바이스 메모리 할당
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # 바인딩 리스트에 저장 (입력, 출력 모두)
        bindings.append(int(device_mem))
        # 입력, 출력 구분해서 저장
        if engine.get_tensor_mode(binding) ==  trt.TensorIOMode.INPUT:
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def preprocess_image(image_path, input_shape):
    # 예) BGR -> RGB, Resize, Normalize, CHW 변환
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[0], input_shape[1]))
    img = img.astype(np.float32) / 255.0
    img = img.transpose((2, 0, 1))  # HWC -> CHW
    return img.ravel()

def infer(engine, context, bindings, inputs, outputs, stream):
    
    start_event = cuda.Event()
    end_event = cuda.Event()
    
    for _ in range(5):
        # 입력 데이터 디바이스로 복사
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        # # 비동기 실행
        # context.execute_async_v3(bindings=bindings, stream_handle=stream.handle)

        # 바인딩 주소를 텐서 이름으로 설정
        for i in range(engine.num_io_tensors):
            context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

        # 시작 이벤트 기록
        start_event.record(stream)
        # 비동기 실행 (v3)
        context.execute_async_v3(stream_handle=stream.handle)
        # 끝 이벤트 기록
        end_event.record(stream)

        # 출력 데이터를 호스트로 복사
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()
    
    # 추론 시간(ms) 계산
    inference_time_ms = start_event.time_till(end_event)
    
    
    return outputs[0]['host'], inference_time_ms


def main():
    engine = load_engine(engine_path)
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    context = engine.create_execution_context()
    after_warmup = []
    for i in range (1,11) :
        if i == 1 :
            print("웜업중")
        for test_image_path in test_image_paths:
            image = preprocess_image(test_image_path, input_shape=(192, 640))
            np.copyto(inputs[0]['host'], image)
            output, inference_time = infer(engine, context, bindings, inputs, outputs, stream)
            if i > 3 :
                after_warmup.append(inference_time)
                print(f"추론 시간: {inference_time:.3f} ms")
    
    # print(f"평균 추론 시간: {avg_inference_time/num:.3f} ms")
    print(f"{model_type} 평균 추론 시간: {sum(after_warmup)/len(after_warmup):.3f} ms")


if __name__ == "__main__":
    
    model_type = 'lite_v4'
    test_image_dir = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\kitti_data\2011_09_26\2011_09_26_drive_0001_sync\image_02\data"
    test_image_paths = glob(osp.join(test_image_dir, "*.png"))
    
    engine_path = osp.join("onnx_output", f'optimized_{model_type}_fp16.engine')

    
    main()