import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

# Dataset import
sys.path.append(os.path.dirname(__file__))
from datasets.kitti_dataset_copy import KITTIRAWDataset
from utils import readlines

# ---- CLAHE와 반대로 동작하는 Shadow Augmentation 함수들 ----

def apply_inverse_clahe_shadow(img_np):
    """CLAHE와 반대로 동작: 대비 감소 + 어둡게 (방법 4: 수학적 역변환)"""
    lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l)
    
    # Inverse: CLAHE의 변화량을 역으로 적용
    # CLAHE는 어두운 부분을 밝게 만듦 → 변화량 = clahe_l - l (양수)
    # 역변환: 원본에서 이 변화량을 빼서 어둡게
    diff = clahe_l.astype(np.float32) - l.astype(np.float32)  # CLAHE가 만든 변화량 (양수)
    
    # 역변환: 원본에서 diff를 빼서 어둡게
    inverse_l = (l.astype(np.float32) - diff * 0.8).astype(np.uint8)
    inverse_l = np.clip(inverse_l, 0, 255)
    
    # 추가로 전체 밝기 감소
    inverse_l = (inverse_l * 0.6).astype(np.uint8)
    
    merged = cv2.merge((inverse_l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_inverse_histogram_equalization(img_np, intensity_range=(0.4, 0.7)):
    """방법 1: Inverse Histogram Equalization - 히스토그램을 역으로 균등화"""
    lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 히스토그램 계산
    hist, bins = np.histogram(l.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    
    # 역 CDF 생성 (어두운 값에 더 많은 픽셀 할당)
    inverse_cdf = 255 - cdf_normalized
    inverse_cdf = (inverse_cdf - inverse_cdf.min()) * 255 / (inverse_cdf.max() - inverse_cdf.min() + 1e-7)
    
    # L 채널에 역 매핑 적용
    l_mapped = np.interp(l.flatten(), bins[:-1], inverse_cdf).reshape(l.shape)
    l_mapped = l_mapped.astype(np.uint8)
    
    # 추가로 전체적으로 어둡게 (랜덤 intensity)
    intensity = np.random.uniform(intensity_range[0], intensity_range[1])
    l_darkened = (l_mapped * intensity).astype(np.uint8)
    
    merged = cv2.merge((l_darkened, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def apply_global_darkening(img_np, intensity_range=(0.4, 0.7)):
    """방법 2: 전체적으로 어둡게 - 단순 밝기 감소"""
    # 랜덤한 intensity 선택
    intensity = 0.4
    
    # 전체 이미지에 일괄 적용
    out = img_np.astype(np.float32) * intensity
    return np.clip(out, 0, 255).astype(np.uint8)

# ---- Dataset에서 샘플 로드 ----
def visualize_dataset_augmentations():
    # Dataset 설정 (trainer.py와 동일한 설정 사용)
    data_path = "kitti_data"  # 필요시 수정
    split = "eigen_zhou"
    height = 192
    width = 640
    frame_ids = [0, -1, 1]
    num_scales = 4
    is_train = True
    img_ext = '.png'
    
    # 파일 경로 설정
    fpath = os.path.join("splits", split, "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))
    
    # Dataset 생성
    dataset = KITTIRAWDataset(
        data_path, 
        train_filenames, 
        height, 
        width,
        frame_ids, 
        num_scales, 
        is_train=is_train, 
        img_ext=img_ext
    )
    
    # 샘플 가져오기
    sample_idx = 0
    inputs = dataset[sample_idx]
    
    # Tensor를 numpy로 변환하고 시각화
    def tensor_to_numpy(tensor_img):
        """Tensor (C, H, W)를 numpy (H, W, C)로 변환"""
        if isinstance(tensor_img, torch.Tensor):
            img = tensor_img.cpu().numpy()
            if img.shape[0] == 3:  # (C, H, W)
                img = img.transpose(1, 2, 0)
            # [0, 1] 범위를 [0, 255]로 변환
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            return img
        return tensor_img
    
    # 원본 이미지
    img_original = tensor_to_numpy(inputs[("color", 0, 0)])
    
    # CLAHE 적용된 이미지
    img_clahe = tensor_to_numpy(inputs[("color_aug", 0, 0)])
    
    # Shadow 적용된 이미지
    img_shadow = tensor_to_numpy(inputs[("color_aug_shadow", 0, 0)])
    
    # 시각화
    plt.figure(figsize=(18, 6))
    
    # (1) Original
    plt.subplot(1, 3, 1)
    plt.title("Original Image", fontsize=14, fontweight='bold')
    plt.imshow(img_original)
    plt.axis("off")
    
    # (2) CLAHE
    plt.subplot(1, 3, 2)
    plt.title("CLAHE Augmentation\n(Brightened)", fontsize=14, fontweight='bold')
    plt.imshow(img_clahe)
    plt.axis("off")
    
    # (3) Shadow
    plt.subplot(1, 3, 3)
    plt.title("Inverse CLAHE Shadow\n(Darkened + Low Contrast)", fontsize=14, fontweight='bold')
    plt.imshow(img_shadow)
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("augmentation_visualization.png", dpi=150, bbox_inches='tight')
    print("시각화 결과가 'augmentation_visualization.png'로 저장되었습니다.")
    plt.show()
    
    # 추가: 여러 샘플 비교
    print("\n=== 여러 샘플 비교 ===")
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    for i in range(3):
        inputs = dataset[i]
        
        img_orig = tensor_to_numpy(inputs[("color", 0, 0)])
        img_clahe = tensor_to_numpy(inputs[("color_aug", 0, 0)])
        img_shadow = tensor_to_numpy(inputs[("color_aug_shadow", 0, 0)])
        
        axes[i, 0].imshow(img_orig)
        axes[i, 0].set_title(f"Sample {i+1}: Original", fontsize=12)
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(img_clahe)
        axes[i, 1].set_title(f"Sample {i+1}: CLAHE", fontsize=12)
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(img_shadow)
        axes[i, 2].set_title(f"Sample {i+1}: Inverse CLAHE Shadow", fontsize=12)
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig("augmentation_comparison.png", dpi=150, bbox_inches='tight')
    print("비교 결과가 'augmentation_comparison.png'로 저장되었습니다.")
    plt.show()


if __name__ == "__main__":
    try:
        visualize_dataset_augmentations()
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: 직접 이미지 파일로 테스트 (3가지 Shadow Augmentation 방법 비교)
        print("\n=== Fallback: 직접 이미지 파일로 테스트 (3가지 Shadow Augmentation 방법) ===")
        if os.path.exists("0000000000.jpg"):
            img = cv2.imread("0000000000.jpg")
            
            # CLAHE 적용
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            img_clahe = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
            
            # 3가지 Shadow Augmentation 적용
            img_shadow_method1 = apply_inverse_histogram_equalization(img)  # 방법 1
            img_shadow_method2 = apply_global_darkening(img)  # 방법 2
            img_shadow_method4 = apply_inverse_clahe_shadow(img)  # 방법 4
            
            # 시각화: 5개 이미지 비교
            plt.figure(figsize=(30, 6))
            
            plt.subplot(1, 5, 1)
            plt.title("Original Image", fontsize=12, fontweight='bold')
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            
            plt.subplot(1, 5, 2)
            plt.title("CLAHE\n(Brightened + High Contrast)", fontsize=12, fontweight='bold')
            plt.imshow(cv2.cvtColor(img_clahe, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            
            plt.subplot(1, 5, 3)
            plt.title("Method 1: Inverse Histogram\nEqualization (Darkened)", fontsize=12, fontweight='bold')
            plt.imshow(cv2.cvtColor(img_shadow_method1, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            
            plt.subplot(1, 5, 4)
            plt.title("Method 2: Global Darkening\n(Simple Brightness Reduction)", fontsize=12, fontweight='bold')
            plt.imshow(cv2.cvtColor(img_shadow_method2, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            
            plt.subplot(1, 5, 5)
            plt.title("Method 4: Inverse CLAHE Shadow\n(Darkened + Low Contrast)", fontsize=12, fontweight='bold')
            plt.imshow(cv2.cvtColor(img_shadow_method4, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            
            plt.tight_layout()
            plt.savefig("all_shadow_methods_comparison.png", dpi=150, bbox_inches='tight')
            print("비교 결과가 'all_shadow_methods_comparison.png'로 저장되었습니다.")
            plt.show()
   
