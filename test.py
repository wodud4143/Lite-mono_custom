import cv2

# === 입력 파일 경로 ===
video_path1 = r'C:\Users\wodud\OneDrive\Desktop\도로주행 데이터\새 폴더 (2)\test3.mp4'
video_path2 = r'C:\Users\wodud\OneDrive\Desktop\도로주행 데이터\새 폴더 (2)\litemono3.mp4'
output_path = r'C:\Users\wodud\OneDrive\Desktop\도로주행 데이터\새 폴더 (2)\video3.mp4'

# === 비디오 캡처 열기 ===
cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)

# === 최소 프레임 수로 동기화 ===
frame_count = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

# === 공통 프레임 크기 맞추기 (가로 크기 기준으로 통일) ===
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

target_width = min(width1, width2)
resize1 = (target_width, int(height1 * target_width / width1))
resize2 = (target_width, int(height2 * target_width / width2))

# === 병합 후 프레임 크기 설정 ===
out_height = resize1[1] + resize2[1]
out_width = target_width

fps = cap1.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

# === 병합 실행 ===
for _ in range(frame_count):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # 리사이징
    frame1 = cv2.resize(frame1, resize1)
    frame2 = cv2.resize(frame2, resize2)

    # 수직 스택
    combined = cv2.vconcat([frame1, frame2])

    # 저장
    out.write(combined)

# === 정리 ===
cap1.release()
cap2.release()
out.release()
print("✅ 저장 완료:", output_path)
