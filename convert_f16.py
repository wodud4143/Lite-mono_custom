import os.path as osp
import torch
import tensorrt as trt
import time
import onnx

from PIL import Image
import networks 
from glob import glob
from torchvision import transforms
from onnx import numpy_helper
import onnxoptimizer


device = 'cuda'

class FullModel(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.decoder(encoded)
        
        disp = out[('disp', 0)]
        disp_2x_down = out[('disp', 1)]
        disp_4x_down = out[('disp', 2)]
        
        return disp                                                                                                                                                                                                  
    
    
    
    
def custom_load_state_dict(loaded_enc, loaded_dec):
    with torch.no_grad():
        encoder = networks.LiteMono(model="lite-mono",
                                                    drop_path_rate=0.2,
                                                    width=640, height=192)
        decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=[0, 1, 2])

        enc_state_dict, dec_state_dict = encoder.state_dict(), decoder.state_dict()
        
        encoder.load_state_dict({
            k: v for k, v in loaded_enc.items() 
            if k in enc_state_dict and enc_state_dict[k].shape == v.shape})
        decoder.load_state_dict({
            k: v for k, v in loaded_dec.items() 
            if k in dec_state_dict and dec_state_dict[k].shape == v.shape})

        encoder.to(device)
        decoder.to(device)
        
        encoder.eval()
        decoder.eval()
    
    return encoder, decoder


def convert_onnx_and_trt(onnx_dir, models,model_type, device='cpu',):
    dummy_input = torch.randn(1, 3, 192, 640).to(device)  # 입력 사이즈에 맞게 조절
    
    encoder, decoder = models
    model = FullModel(encoder, decoder).eval()
    model = model.to(device)
    
    
    torch.onnx.export(
        model, dummy_input, osp.join(onnx_dir, model_type+'.onnx'),
        input_names=["input"], output_names=["output"],
        dynamic_axes=None,
        export_params=True,
        do_constant_folding=True,
        opset_version=17
    )
    
    # 그래프 최적화
    model = onnx.load(osp.join(onnx_dir, model_type+'.onnx'))
    passes = [
        'eliminate_deadend',
        'eliminate_identity',
        'eliminate_nop_transpose',
        'fuse_consecutive_transposes',
        'fuse_bn_into_conv',
        'fuse_pad_into_conv',  # Depthwise-friendly
        'fuse_add_bias_into_conv'  # TensorRT에서 Conv + Bias로 병합
    ]

    optimized_model = onnxoptimizer.optimize(model, passes)
    for init in optimized_model.graph.initializer:
        if init.data_type == onnx.TensorProto.INT64:
            arr = numpy_helper.to_array(init)
            arr32 = arr.astype('int32')
            init.CopyFrom(numpy_helper.from_array(arr32, init.name))
    
    onnx.save(optimized_model, osp.join(onnx_dir, f'optimized_{model_type}.onnx'))
    engine = build_engine(osp.join(onnx_dir, f'optimized_{model_type}.onnx'))
    
    with open(osp.join(onnx_dir, f"optimized_{model_type}.engine"), "wb") as f:
        f.write(engine.serialize())
    print('done')


# region [build trt]
def build_engine(onnx_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB


    # ✅ FP16 지원 여부 확인
    config.set_flag(trt.BuilderFlag.FP16)
    
    
    # ✅ 네트워크 생성 (explicit batch)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # ✅ ONNX 파싱
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as f:
        parser.parse(f.read())

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build engine")
        return None
    

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    # # ✅ 엔진 빌드
    # engine = builder.build_engine(network, config)
    return engine


def convertTensorRT(model_path,model_type):
    onnx_dir = "onnx_output"
    model_type = model_type
    enc_model_path = model_path + "\encoder.pth"
    dec_model_path = model_path + "\depth.pth"
    enc_state_dict = torch.load(enc_model_path, map_location=device)
    dec_state_dict = torch.load(dec_model_path, map_location=device)
    encoder, decoder = custom_load_state_dict(enc_state_dict, dec_state_dict)
    convert_onnx_and_trt(onnx_dir, models=[encoder, decoder], model_type = model_type, device=device)



def main():
    onnx_dir = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\onnx_output"
    
    enc_state_dict = torch.load(enc_model_path, map_location=device)
    dec_state_dict = torch.load(dec_model_path, map_location=device)
    model_type = 'v4_3_R_aug3_gamma'
    encoder, decoder = custom_load_state_dict(enc_state_dict, dec_state_dict)
    convert_onnx_and_trt(onnx_dir, models=[encoder, decoder], model_type = model_type, device=device)


if __name__ == "__main__":

    exp_dir = osp.join(osp.dirname(__file__), "experiments\logs")
    
    enc_model_path = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\v4_3_R_aug3_gamma\models\weights_19\encoder.pth"
    dec_model_path = r"C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\experiments\logs\v4_3_R_aug3_gamma\models\weights_19\depth.pth"
    main()