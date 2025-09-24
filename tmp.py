import torch
import random
import kornia.geometry.transform as KTF
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt

def translate_image(img_tensor, tx, ty):
    M = torch.tensor([
        [1., 0., tx],
        [0., 1., ty]
    ], dtype=torch.float32).unsqueeze(0)
    return KTF.warp_affine(img_tensor.unsqueeze(0), M,
                           dsize=(img_tensor.shape[1], img_tensor.shape[2]),
                           mode='nearest',
                           padding_mode='border',
                           align_corners=True).squeeze(0)

def test_translation(img_path, translate_ratio=0.2):
    # 이미지 로드 및 tensor 변환 (C, H, W), 0~1 범위
    img = ToTensor()(Image.open(img_path).convert('RGB'))

    h, w = img.shape[1], img.shape[2]
    # 무작위 이동 픽셀 수
    tx = int(random.uniform(-translate_ratio, translate_ratio) * w)
    ty = int(random.uniform(-translate_ratio, translate_ratio) * h)

    translated = translate_image(img, tx, ty)

    to_pil = ToPILImage()
    orig_img = to_pil(img)
    trans_img = to_pil(translated)

    # 시각화
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(orig_img)
    axs[0].set_title('원본 이미지')
    axs[0].axis('off')
    axs[1].imshow(trans_img)
    axs[1].set_title(f'Translated Image (tx={tx}, ty={ty})')
    axs[1].axis('off')
    plt.show()

# 아래에 테스트할 이미지 경로 입력
img_path = r'C:\Users\wodud\OneDrive\Desktop\Lite-mono_custom\kitti_data\2011_09_26\2011_09_26_drive_0014_sync\image_02\data\0000000003.png'
test_translation(img_path)
