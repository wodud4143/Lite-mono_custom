import torch
from thop import profile, clever_format
from networks.depth_encoder import LiteMono
from networks.depth_decoder import DepthDecoder

def calculate_flops(model_name="lite-mono", height=192, width=640, device="cuda"):

    x = torch.randn(1, 3, height, width).to(device)


    encoder = LiteMono(model=model_name, height=height, width=width).to(device)
    decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3)).to(device)

    with torch.no_grad():
        features = encoder(x)
        flops_e, params_e = profile(encoder, inputs=(x,), verbose=False)
        flops_d, params_d = profile(decoder, inputs=(features,), verbose=False)
        total_flops = flops_e + flops_d
        total_params = params_e + params_d

    flops_str, params_str = clever_format([total_flops, total_params], "%.3f")
    flops_e_str, params_e_str = clever_format([flops_e, params_e], "%.3f")
    flops_d_str, params_d_str = clever_format([flops_d, params_d], "%.3f")

    print(f"\nFLOPs")
    print(f"input_size       : {height} x {width}")
    print(f"model            : {model_name}")
    print(f"Encoder FLOPs    : {flops_e_str} | Params: {params_e_str}")
    print(f"Decoder FLOPs    : {flops_d_str} | Params: {params_d_str}")
    print(f"Total FLOPs      : {flops_str} | Total Params: {params_str}\n")


if __name__ == "__main__":
    calculate_flops(model_name="lite-mono", height=192, width=640)
