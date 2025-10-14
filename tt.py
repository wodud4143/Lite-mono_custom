import os
import random
import numpy as np
import PIL.Image as pil
import torch
import kornia.geometry.transform as KTF

# --------------------------------------------------------------------------------
# 제공해주신 KITTIDataset의 get_color 메소드 로직을 그대로 가져온 함수입니다.
# --------------------------------------------------------------------------------
def apply_augmentations(color_img, do_flip, do_crop, crop_params, do_rotate, rot_angle, do_tr_aug, tr_params):
    """주어진 이미지에 모든 증강을 순서대로 적용합니다."""
    
    # 원본 이미지를 복사하여 사용
    color = color_img.copy()

    # 1. 좌우 반전 (Flip)
    if do_flip:
        print("-> Applying Flip")
        color = color.transpose(pil.FLIP_LEFT_RIGHT)
        
    # 2. 크롭 (Crop)
    if do_crop and crop_params is not None:
        print(f"-> Applying Crop with params: {crop_params}")
        crop_x, crop_y, crop_w, crop_h = crop_params
        color = color.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
        
    # 3. 회전 (Rotate)
    if do_rotate:
        print(f"-> Applying Rotation with angle: {rot_angle:.2f} degrees")
        color = color.rotate(rot_angle, resample=pil.BICUBIC, expand=False)

    # 4. 이동 (Translate)
    if do_tr_aug:
        print(f"-> Applying Translation with ratios: ({tr_params[0]:.2f}, {tr_params[1]:.2f})")
        
        # --- Translate에 필요한 헬퍼 함수 ---
        def pil_to_tensor(img):
            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            return tensor

        def tensor_to_pil(tensor):
            arr = (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
            return pil.fromarray(arr)

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
        # ------------------------------------

        color_tensor = pil_to_tensor(color)
        h, w = color_tensor.shape[1], color_tensor.shape[2]

        tx_ratio, ty_ratio = tr_params
        tx = int(tx_ratio * w)
        ty = int(ty_ratio * h)

        translated_tensor = translate_image(color_tensor, tx, ty)
        color = tensor_to_pil(translated_tensor)
            
    return color


# --------------------------------------------------------------------------------
# 메인 실행 부분
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # ========================= 사용자 설정 영역 =========================
    # 여기에 테스트하고 싶은 이미지의 전체 경로를 입력하세요.
    IMAGE_PATH = "test.png"

    # 각 증강을 켜거나 끌 수 있습니다 (True: 적용, False: 미적용).
    APPLY_FLIP = False
    APPLY_CROP = False
    APPLY_ROTATE = True
    
    APPLY_TRANSLATE = True
    # =================================================================

    # --- 이미지 로딩 ---
    if not os.path.exists(IMAGE_PATH):
        print(f"오류: 이미지 파일을 찾을 수 없습니다. 경로를 확인해주세요: {IMAGE_PATH}")
    else:
        original_image = pil.open(IMAGE_PATH).convert('RGB')
        original_w, original_h = original_image.size
        print(f"원본 이미지 로딩 완료: {IMAGE_PATH} (크기: {original_w}x{original_h})")

        # --- 증강 파라미터 랜덤 생성 (MonoDataset __getitem__ 로직과 유사하게) ---
        # 1. 크롭 파라미터
        crop_ratio = random.uniform(0.85, 0.95)
        crop_w = int(original_w * crop_ratio)
        crop_h = int(original_h * crop_ratio)
        crop_x = random.randint(0, original_w - crop_w)
        crop_y = random.randint(0, original_h - crop_h)
        crop_params = (crop_x, crop_y, crop_w, crop_h)

        # 2. 회전 파라미터
        rot_angle = random.uniform(-25, 25)

        # 3. 이동 파라미터
        tr_params = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))

        # --- 증강 적용 ---
        print("\n적용될 증강 목록:")
        augmented_image = apply_augmentations(
            original_image,
            do_flip=APPLY_FLIP,
            do_crop=APPLY_CROP,
            crop_params=crop_params,
            do_rotate=APPLY_ROTATE,
            rot_angle=rot_angle,
            do_tr_aug=APPLY_TRANSLATE,
            tr_params=tr_params
        )
        print("\n증강 적용 완료!")

        # --- 결과 비교 및 출력 ---
        # 원본과 증강된 이미지를 나란히 붙여서 보여주기
        # 증강된 이미지의 크기가 크롭으로 인해 작아질 수 있으므로, 원본 크기에 맞춰서 붙임
        side_by_side = pil.new('RGB', (original_w * 2, original_h))
        side_by_side.paste(original_image, (0, 0))
        
        # 증강된 이미지를 중앙에 위치시키기 위한 좌표 계산
        paste_x = original_w + (original_w - augmented_image.width) // 2
        paste_y = (original_h - augmented_image.height) // 2
        side_by_side.paste(augmented_image, (paste_x, paste_y))
        
        # 결과 이미지 보기
        side_by_side.show(title="Original vs. Augmented")

        # 결과 이미지 저장
        output_path = "augmentation_result.png"
        side_by_side.save(output_path)
        print(f"결과가 '{output_path}' 파일로 저장되었습니다.")