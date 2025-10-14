import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import os

def apply_clahe_rgb(img, clip_limit=3.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)

def adjust_gamma(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def put_text_with_background(img, text, position, font_scale=0.8, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """텍스트에 배경을 추가하여 가독성을 높입니다."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    
    # 텍스트 크기 계산
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    # 배경 사각형 그리기 (약간의 패딩 추가)
    padding = 5
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding),
                  bg_color, -1)
    
    # 텍스트 그리기
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

def visualize_preprocessing_pipeline(image_path, output_dir="preprocessing_steps"):
    """
    전처리 파이프라인의 각 단계를 시각화하고 저장합니다.
    
    Args:
        image_path: 입력 이미지 경로
        output_dir: 결과를 저장할 디렉토리 (기본값: "preprocessing_steps")
    """
    # 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"출력 디렉토리 생성: {output_dir}")
    
    # 원본 이미지 로드
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"Error: 이미지를 불러올 수 없습니다. 경로를 확인하세요: {image_path}")
        return
    
    img = img_original.copy()
    h, w = img.shape[:2]
    
    # 결과를 저장할 리스트
    stages = []
    stages_bgr = []  # BGR 형식으로 저장 (파일 저장용)
    titles = []
    descriptions = []
    
    # ========== 원본 이미지 ==========
    stages.append(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
    stages_bgr.append(img_original.copy())
    titles.append("0_original")
    global_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    descriptions.append(f"전역 평균 밝기: {global_brightness:.1f}")
    
    # ========== 1단계: 전역 밝기 평가 및 조정 ==========
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global_brightness = np.mean(gray)
    
    stage1_img = img.copy()
    stage1_desc = f"전역 평균 밝기: {global_brightness:.1f}"
    
    if global_brightness > 180:
        # 과노출 억제
        suppress_gamma = 0.7
        stage1_img = adjust_gamma(img, gamma=suppress_gamma)
        stage1_desc += f"\n→ 과노출 감지! 감마 {suppress_gamma} 적용"
        clahe_limit = 3.0
    elif global_brightness < 60:
        # 저조도 환경
        stage1_desc += "\n→ 저조도 감지! 다음 단계에서 CLAHE 강화"
        clahe_limit = 3.0 + 1.5
    else:
        stage1_desc += "\n→ 정상 밝기 범위"
        clahe_limit = 3.0
    
    stages.append(cv2.cvtColor(stage1_img, cv2.COLOR_BGR2RGB))
    stages_bgr.append(stage1_img.copy())
    titles.append("1_global_brightness_adjustment")
    descriptions.append(stage1_desc)
    img = stage1_img.copy()
    
    # ========== 2단계: CLAHE 적용 ==========
    stage2_img = apply_clahe_rgb(img, clip_limit=clahe_limit)
    stage2_desc = f"CLAHE clip_limit: {clahe_limit:.1f}\n8×8 타일 기반 지역 대비 향상"
    
    stages.append(cv2.cvtColor(stage2_img, cv2.COLOR_BGR2RGB))
    stages_bgr.append(stage2_img.copy())
    titles.append("2_clahe_applied")
    descriptions.append(stage2_desc)
    img = stage2_img.copy()
    
    # ========== 3단계: 중앙 슬라이스 및 좌우 영역 분석 (개선된 시각화) ==========
    slice_width = max(w // 10, 1)
    center_slice = img[:, w//2 - slice_width : w//2 + slice_width]
    target_brightness = np.mean(cv2.cvtColor(center_slice, cv2.COLOR_BGR2GRAY))

    left_img = img[:, :w//2]
    right_img = img[:, w//2:]
    left_brightness = np.mean(cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY))
    right_brightness = np.mean(cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY))

    # 시각화를 위한 이미지 생성 (영역 표시)
    stage3_img = img.copy()

    # 반투명 오버레이 생성
    overlay = stage3_img.copy()

    # 좌측 영역 표시 (파란색 반투명)
    cv2.rectangle(overlay, (0, 0), (w//2, h), (255, 100, 100), -1)

    # 우측 영역 표시 (초록색 반투명)
    cv2.rectangle(overlay, (w//2, 0), (w, h), (100, 255, 100), -1)

    # 중앙 슬라이스 표시 (노란색 반투명)
    cv2.rectangle(overlay, (w//2 - slice_width, 0), (w//2 + slice_width, h), (100, 255, 255), -1)

    # 오버레이 합성 (투명도 20%)
    cv2.addWeighted(overlay, 0.2, stage3_img, 0.8, 0, stage3_img)

    # 경계선 그리기
    # 중앙 슬라이스 경계 (노란색)
    cv2.rectangle(stage3_img, (w//2 - slice_width, 0), (w//2 + slice_width, h), (0, 255, 255), 3)
    # 좌우 분할선 (빨간색)
    cv2.line(stage3_img, (w//2, 0), (w//2, h), (0, 0, 255), 3)

    # 텍스트 추가 (배경 포함) - 상단: 밝기 값들
    text_y_top = 50

    # 좌측 밝기 (상단 왼쪽)
    put_text_with_background(stage3_img, f"Left: {left_brightness:.1f}", 
                            (30, text_y_top), 
                            font_scale=0.9, color=(255, 255, 255), bg_color=(0, 0, 0))

    # 중앙(목표) 밝기 (상단 중앙) - 중앙 정렬
    center_text = f"Target: {target_brightness:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, _), _ = cv2.getTextSize(center_text, font, 0.9, 2)
    center_text_x = (w//2) - (text_width // 2)
    put_text_with_background(stage3_img, center_text, 
                            (center_text_x, text_y_top), 
                            font_scale=0.9, color=(0, 255, 255), bg_color=(0, 0, 0))

    # 우측 밝기 (상단 오른쪽)
    put_text_with_background(stage3_img, f"Right: {right_brightness:.1f}", 
                            (w - 200, text_y_top), 
                            font_scale=0.9, color=(255, 255, 255), bg_color=(0, 0, 0))

    # 하단에 레이블 추가
    label_y = h - 30

    # Left Region (하단 왼쪽)
    put_text_with_background(stage3_img, "Left Region", 
                            (30, label_y), 
                            font_scale=0.8, color=(255, 100, 100), bg_color=(0, 0, 0))

    # Center Slice (하단 중앙)
    center_label = "Center Slice"
    (label_width, _), _ = cv2.getTextSize(center_label, font, 0.8, 2)
    center_label_x = (w//2) - (label_width // 2)
    put_text_with_background(stage3_img, center_label, 
                            (center_label_x, label_y), 
                            font_scale=0.8, color=(0, 255, 255), bg_color=(0, 0, 0))

    # Right Region (하단 오른쪽)
    put_text_with_background(stage3_img, "Right Region", 
                            (w - 180, label_y), 
                            font_scale=0.8, color=(100, 255, 100), bg_color=(0, 0, 0))

    stage3_desc = f"중앙(목표): {target_brightness:.1f}\n좌측: {left_brightness:.1f} | 우측: {right_brightness:.1f}"

    stages.append(cv2.cvtColor(stage3_img, cv2.COLOR_BGR2RGB))
    stages_bgr.append(stage3_img.copy())
    titles.append("3_region_analysis")
    descriptions.append(stage3_desc)
    
    # ========== 4단계: 좌우 독립 보정 ==========
    brightness_threshold = 5.0
    
    def enhance_region(region_img, region_brightness):
        diff = abs(target_brightness - region_brightness)
        gamma_info = ""
        
        if global_brightness < 50 or diff > brightness_threshold:
            gamma_boost = 1.5 if global_brightness < 40 else 1.2
            estimated_gamma = np.log(target_brightness + 1e-6) / np.log(region_brightness + 1e-6)
            blended_gamma = np.clip((estimated_gamma + gamma_boost) / 2, 0.6, 2.0)
            region_img = adjust_gamma(region_img, gamma=blended_gamma)
            gamma_info = f"γ={blended_gamma:.2f}"
        elif region_brightness > 200:
            suppress_gamma = 0.7
            region_img = adjust_gamma(region_img, gamma=suppress_gamma)
            gamma_info = f"γ={suppress_gamma}"
        else:
            gamma_info = "보정 없음"
        
        return region_img, gamma_info
    
    left_enhanced, left_gamma_info = enhance_region(left_img.copy(), left_brightness)
    right_enhanced, right_gamma_info = enhance_region(right_img.copy(), right_brightness)
    
    stage4_img = np.hstack((left_enhanced, right_enhanced))
    stage4_desc = f"좌측: {left_gamma_info}\n우측: {right_gamma_info}"
    
    stages.append(cv2.cvtColor(stage4_img, cv2.COLOR_BGR2RGB))
    stages_bgr.append(stage4_img.copy())
    titles.append("4_independent_correction")
    descriptions.append(stage4_desc)
    
    # ========== 5단계: 최종 결과 (경계선 표시) ==========
    stage5_img = stage4_img.copy()
    cv2.line(stage5_img, (w//2, 0), (w//2, h), (0, 255, 0), 3)
    
    put_text_with_background(stage5_img, "Merge Boundary", 
                            (w//2 - 120, h - 30), 
                            font_scale=0.9, color=(0, 255, 0), bg_color=(0, 0, 0))
    
    final_brightness = np.mean(cv2.cvtColor(stage4_img, cv2.COLOR_BGR2GRAY))
    stage5_desc = f"좌우 영역 결합 완료\n최종 평균 밝기: {final_brightness:.1f}"
    
    stages.append(cv2.cvtColor(stage5_img, cv2.COLOR_BGR2RGB))
    stages_bgr.append(stage5_img.copy())
    titles.append("5_final_merged")
    descriptions.append(stage5_desc)
    
    # ========== 개별 이미지 파일 저장 ==========
    print("\n=== 단계별 이미지 저장 중 ===")
    for idx, (bgr_img, title) in enumerate(zip(stages_bgr, titles)):
        filename = os.path.join(output_dir, f"{title}.png")
        cv2.imwrite(filename, bgr_img)
        print(f"저장 완료: {filename}")
    
    # 최종 결과 (경계선 없는 버전) 별도 저장
    final_clean = stage4_img.copy()
    final_clean_path = os.path.join(output_dir, "final_result_clean.png")
    cv2.imwrite(final_clean_path, final_clean)
    print(f"저장 완료: {final_clean_path}")
    
    # ========== 시각화 ==========
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, (stage_img, title, desc) in enumerate(zip(stages, titles, descriptions)):
        if idx < 3:
            ax = fig.add_subplot(gs[0, idx])
        else:
            ax = fig.add_subplot(gs[1, idx - 3])
        
        ax.imshow(stage_img)
        ax.set_title(f"{title}\n{desc}", fontsize=11, pad=10)
        ax.axis('off')
    
    plt.suptitle("전처리 파이프라인 단계별 시각화", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 전체 시각화 저장
    visualization_path = os.path.join(output_dir, "pipeline_visualization.png")
    plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
    print(f"저장 완료: {visualization_path}")
    plt.show()
    
    # ========== 원본 vs 최종 비교 ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].imshow(stages[0])
    axes[0].set_title(f"원본 이미지\n평균 밝기: {np.mean(cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)):.1f}", 
                      fontsize=13)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(stage4_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"전처리 완료\n평균 밝기: {final_brightness:.1f}", fontsize=13)
    axes[1].axis('off')
    
    plt.suptitle("원본 vs 전처리 결과 비교", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 비교 이미지 저장
    comparison_path = os.path.join(output_dir, "comparison_original_vs_processed.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"저장 완료: {comparison_path}")
    plt.show()
    
    print(f"\n=== 모든 결과가 '{output_dir}' 디렉토리에 저장되었습니다 ===")
    print(f"총 {len(stages_bgr) + 2}개의 이미지 파일이 생성되었습니다.")

# ========== 사용 예시 ==========
if __name__ == "__main__":
    # 여기에 이미지 경로를 설정하세요
    image_path = "test.png"  # 예: "test_image.jpg"
    
    # 출력 디렉토리 이름도 변경 가능 (선택사항)
    output_directory = "preprocessing_steps"  # 원하는 폴더명으로 변경 가능
    
    visualize_preprocessing_pipeline(image_path, output_dir=output_directory)