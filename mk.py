import os
import glob
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import json
from datetime import datetime

class VideoMergerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("도로주행 데이터 비디오 생성 프로그램")
        self.root.geometry("900x800")
        
        # 설정 파일 경로
        self.config_file = "video_merger_config.json"
        
        # 폴더 경로 변수들
        self.original_folder = tk.StringVar()
        self.model_folders = []  # 동적으로 관리될 모델 폴더 리스트
        
        # 기타 설정 변수들
        self.subfolder_name = tk.StringVar(value="2011_09_26_drive_0009_sync_학습함")
        self.frame_rate = tk.IntVar(value=15)
        self.output_path = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Desktop", "output.mp4"))
        self.text_size = tk.DoubleVar(value=0.7)
        self.text_height = tk.IntVar(value=40)
        self.image_width = tk.IntVar(value=640)
        self.image_height = tk.IntVar(value=480)
        self.use_custom_size = tk.BooleanVar(value=False)
        
        # 진행 상태 변수
        self.progress = tk.DoubleVar()
        self.status_text = tk.StringVar(value="준비됨")
        
        # 모델 폴더 위젯들을 저장할 리스트
        self.model_widgets = []
        
        # 이전 설정 불러오기
        self.load_config()
        
        # UI 생성
        self.create_ui()
        
    def create_ui(self):
        # 메인 프레임 (스크롤 가능)
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        main_frame = ttk.Frame(scrollable_frame, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # 제목
        title_label = ttk.Label(main_frame, text="도로주행 데이터 비디오 생성기", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # 원본 영상 폴더 선택
        original_frame = ttk.LabelFrame(main_frame, text="원본 영상 폴더", padding="10")
        original_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(original_frame, text="원본 영상 (RGB):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(original_frame, textvariable=self.original_folder, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(original_frame, text="찾아보기", command=self.browse_original_folder).grid(row=0, column=2, pady=5)
        
        # 모델 폴더 선택 섹션
        self.model_frame = ttk.LabelFrame(main_frame, text="모델 폴더 선택", padding="10")
        self.model_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # 모델 추가/제거 버튼
        button_frame = ttk.Frame(self.model_frame)
        button_frame.grid(row=0, column=0, columnspan=3, pady=5)
        
        ttk.Button(button_frame, text="모델 추가", command=self.add_model_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="모델 제거", command=self.remove_model_folder).pack(side=tk.LEFT, padx=5)
        ttk.Label(button_frame, text="(최소 1개, 최대 10개)").pack(side=tk.LEFT, padx=10)
        
        # 모델 폴더 리스트 프레임
        self.model_list_frame = ttk.Frame(self.model_frame)
        self.model_list_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # 초기 모델 폴더 추가 (기본 4개)
        if not self.model_folders:  # 설정에서 불러오지 않았다면
            for i in range(4):
                self.add_model_folder()
        
        # 설정 섹션
        settings_frame = ttk.LabelFrame(main_frame, text="설정", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # 하위 폴더명
        ttk.Label(settings_frame, text="하위 폴더명:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(settings_frame, textvariable=self.subfolder_name, width=40).grid(row=0, column=1, columnspan=2, pady=5)
        
        # 프레임 레이트
        ttk.Label(settings_frame, text="프레임 레이트:").grid(row=1, column=0, sticky=tk.W, pady=5)
        frame_rate_spinbox = ttk.Spinbox(settings_frame, from_=1, to=60, textvariable=self.frame_rate, width=10)
        frame_rate_spinbox.grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(settings_frame, text="fps").grid(row=1, column=2, sticky=tk.W, pady=5)
        
        # 텍스트 크기
        ttk.Label(settings_frame, text="텍스트 크기:").grid(row=2, column=0, sticky=tk.W, pady=5)
        text_size_scale = ttk.Scale(settings_frame, from_=0.3, to=1.5, variable=self.text_size, 
                                   orient=tk.HORIZONTAL, length=200)
        text_size_scale.grid(row=2, column=1, pady=5)
        text_size_label = ttk.Label(settings_frame, text=f"{self.text_size.get():.1f}")
        text_size_label.grid(row=2, column=2, pady=5)
        
        # 텍스트 크기 라벨 업데이트
        def update_text_size_label(value):
            text_size_label.config(text=f"{float(value):.1f}")
        text_size_scale.config(command=update_text_size_label)
        
        # 텍스트 영역 높이
        ttk.Label(settings_frame, text="텍스트 영역 높이:").grid(row=3, column=0, sticky=tk.W, pady=5)
        text_height_spinbox = ttk.Spinbox(settings_frame, from_=20, to=100, textvariable=self.text_height, width=10)
        text_height_spinbox.grid(row=3, column=1, sticky=tk.W, pady=5)
        ttk.Label(settings_frame, text="pixels").grid(row=3, column=2, sticky=tk.W, pady=5)
        
        # 구분선
        ttk.Separator(settings_frame, orient='horizontal').grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # 이미지 크기 설정
        size_frame = ttk.Frame(settings_frame)
        size_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # 커스텀 크기 사용 체크박스
        self.size_checkbox = ttk.Checkbutton(size_frame, text="커스텀 이미지 크기 사용", 
                                           variable=self.use_custom_size,
                                           command=self.toggle_size_inputs)
        self.size_checkbox.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # 너비 입력
        ttk.Label(size_frame, text="너비:").grid(row=1, column=0, sticky=tk.W, padx=(20, 5), pady=5)
        self.width_spinbox = ttk.Spinbox(size_frame, from_=320, to=3840, textvariable=self.image_width, 
                                        width=10, increment=10)
        self.width_spinbox.grid(row=1, column=1, sticky=tk.W, pady=5)
        ttk.Label(size_frame, text="pixels").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        
        # 높이 입력
        ttk.Label(size_frame, text="높이:").grid(row=2, column=0, sticky=tk.W, padx=(20, 5), pady=5)
        self.height_spinbox = ttk.Spinbox(size_frame, from_=240, to=2160, textvariable=self.image_height, 
                                         width=10, increment=10)
        self.height_spinbox.grid(row=2, column=1, sticky=tk.W, pady=5)
        ttk.Label(size_frame, text="pixels").grid(row=2, column=2, sticky=tk.W, padx=5, pady=5)
        
        # 초기 상태 설정
        self.toggle_size_inputs()
        
        # 출력 파일 섹션
        output_frame = ttk.LabelFrame(main_frame, text="출력 파일", padding="10")
        output_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(output_frame, text="저장 위치:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(output_frame, text="찾아보기", command=self.browse_output).grid(row=0, column=2, pady=5)
        
        # 버튼 섹션
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="비디오 생성 시작", 
                                      command=self.start_processing, style='Accent.TButton')
        self.start_button.grid(row=0, column=0, padx=5)
        
        ttk.Button(button_frame, text="설정 저장", command=self.save_config).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="설정 초기화", command=self.reset_config).grid(row=0, column=2, padx=5)
        
        # 진행 상태 섹션
        progress_frame = ttk.LabelFrame(main_frame, text="진행 상태", padding="10")
        progress_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress, length=500)
        self.progress_bar.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Label(progress_frame, textvariable=self.status_text).grid(row=1, column=0, columnspan=2, pady=5)
        
        # 로그 섹션
        log_frame = ttk.LabelFrame(main_frame, text="로그", padding="10")
        log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        self.log_text = tk.Text(log_frame, height=8, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        
    def add_model_folder(self):
        if len(self.model_folders) >= 10:
            messagebox.showwarning("경고", "최대 10개의 모델까지만 추가할 수 있습니다.")
            return
            
        # 새 모델 폴더 변수 추가
        folder_var = tk.StringVar()
        self.model_folders.append(folder_var)
        
        # UI 위젯 생성
        row = len(self.model_folders)
        frame = ttk.Frame(self.model_list_frame)
        frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=2)
        
        label = ttk.Label(frame, text=f"모델 {row}:")
        label.pack(side=tk.LEFT, padx=5)
        
        entry = ttk.Entry(frame, textvariable=folder_var, width=50)
        entry.pack(side=tk.LEFT, padx=5)
        
        button = ttk.Button(frame, text="찾아보기", 
                           command=lambda var=folder_var, idx=row: self.browse_model_folder(var, idx))
        button.pack(side=tk.LEFT, padx=5)
        
        self.model_widgets.append((frame, label, entry, button))
        self.log(f"모델 {row} 추가됨")
        
    def remove_model_folder(self):
        if len(self.model_folders) <= 1:
            messagebox.showwarning("경고", "최소 1개의 모델은 필요합니다.")
            return
            
        # 마지막 모델 제거
        self.model_folders.pop()
        
        # UI 위젯 제거
        if self.model_widgets:
            frame, label, entry, button = self.model_widgets.pop()
            frame.destroy()
            
        self.log(f"모델 제거됨. 현재 모델 수: {len(self.model_folders)}")
        
    def browse_original_folder(self):
        folder_path = filedialog.askdirectory(title="원본 영상 폴더 선택")
        if folder_path:
            self.original_folder.set(folder_path)
            self.log(f"원본 영상 폴더 선택됨: {folder_path}")
            
    def browse_model_folder(self, folder_var, index):
        folder_path = filedialog.askdirectory(title=f"모델 {index} 폴더 선택")
        if folder_path:
            folder_var.set(folder_path)
            self.log(f"모델 {index} 폴더 선택됨: {folder_path}")
            
    def browse_output(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")],
            title="출력 파일 위치 선택"
        )
        if file_path:
            self.output_path.set(file_path)
            self.log(f"출력 파일 위치: {file_path}")
            
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def save_config(self):
        config = {
            'original_folder': self.original_folder.get(),
            'model_folders': [folder.get() for folder in self.model_folders],
            'subfolder_name': self.subfolder_name.get(),
            'frame_rate': self.frame_rate.get(),
            'output_path': self.output_path.get(),
            'text_size': self.text_size.get(),
            'text_height': self.text_height.get()
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            self.log("설정이 저장되었습니다.")
            messagebox.showinfo("저장 완료", "설정이 저장되었습니다.")
        except Exception as e:
            self.log(f"설정 저장 실패: {e}")
            messagebox.showerror("저장 실패", f"설정 저장 중 오류가 발생했습니다: {e}")
            
    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                self.original_folder.set(config.get('original_folder', ''))
                
                # 모델 폴더 불러오기
                model_folders = config.get('model_folders', [])
                for folder_path in model_folders:
                    folder_var = tk.StringVar(value=folder_path)
                    self.model_folders.append(folder_var)
                    
                self.subfolder_name.set(config.get('subfolder_name', ''))
                self.frame_rate.set(config.get('frame_rate', 15))
                self.output_path.set(config.get('output_path', ''))
                self.text_size.set(config.get('text_size', 0.7))
                self.text_height.set(config.get('text_height', 40))
                
                self.log("이전 설정을 불러왔습니다.")
            except Exception as e:
                self.log(f"설정 불러오기 실패: {e}")
                
    def reset_config(self):
        if messagebox.askyesno("설정 초기화", "모든 설정을 초기화하시겠습니까?"):
            self.original_folder.set("")
            
            # 모든 모델 위젯 제거
            for frame, _, _, _ in self.model_widgets:
                frame.destroy()
            self.model_widgets.clear()
            self.model_folders.clear()
            
            # 기본 4개 모델 추가
            for i in range(4):
                self.add_model_folder()
                
            self.subfolder_name.set("2011_09_26_drive_0009_sync_학습함")
            self.frame_rate.set(15)
            self.output_path.set(os.path.join(os.path.expanduser("~"), "Desktop", "output.mp4"))
            self.text_size.set(0.7)
            self.text_height.set(40)
            self.log("설정이 초기화되었습니다.")
            
    def start_processing(self):
        # 입력 검증
        if not self.original_folder.get():
            messagebox.showerror("오류", "원본 영상 폴더를 선택해주세요.")
            return
            
        if not any(folder.get() for folder in self.model_folders):
            messagebox.showerror("오류", "최소 하나의 모델 폴더를 선택해주세요.")
            return
            
        if not self.subfolder_name.get():
            messagebox.showerror("오류", "하위 폴더명을 입력해주세요.")
            return
            
        # 버튼 비활성화
        self.start_button.config(state='disabled')
        
        # 별도 스레드에서 처리
        thread = threading.Thread(target=self.process_video)
        thread.start()
        
    def get_subfolder_after_target(self, path, target="도로주행 데이터"):
        parts = os.path.normpath(path).split(os.sep)
        if target in parts:
            idx = parts.index(target)
            if idx + 1 < len(parts):
                return parts[idx + 1]
        return os.path.basename(path)
        
    def load_image(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            self.log(f"이미지 로드 실패: {img_path}, 오류: {e}")
            return None
            
    def add_text_to_image(self, img, text):
        """이미지에 텍스트 추가"""
        height, width = img.shape[:2]
        
        # 텍스트 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = self.text_size.get()
        thickness = 2
        color = (255, 255, 255)  # 흰색
        
        # 텍스트 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 텍스트 위치 계산 (하단 중앙)
        text_x = (width - text_width) // 2
        text_y = height - 10  # 하단에서 10픽셀 위
        
        # 텍스트 배경 추가 (가독성 향상)
        bg_rect_start = (text_x - 5, text_y - text_height - 5)
        bg_rect_end = (text_x + text_width + 5, text_y + baseline + 5)
        cv2.rectangle(img, bg_rect_start, bg_rect_end, (0, 0, 0), -1)  # 검은색 배경
        
        # 텍스트 추가
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return img
        
    def process_video(self):
        try:
            self.status_text.set("비디오 생성 중...")
            self.progress.set(0)
            
            subfolder = self.subfolder_name.get()
            
            # 실제로 선택된 모델 폴더만 필터링
            active_model_folders = [(i, folder) for i, folder in enumerate(self.model_folders) if folder.get()]
            
            if not active_model_folders:
                raise Exception("선택된 모델 폴더가 없습니다.")
                
            self.log(f"선택된 모델 수: {len(active_model_folders)}")
            
            # 폴더 경로 설정
            original_folder_path = os.path.join(self.original_folder.get(), subfolder)
            model_folder_paths = []
            model_names = []
            
            for idx, folder_var in active_model_folders:
                folder_path = os.path.join(folder_var.get(), subfolder)
                model_folder_paths.append(folder_path)
                model_name = self.get_subfolder_after_target(folder_var.get())
                model_names.append(model_name)
                self.log(f"모델 {idx + 1}: {model_name}")
            
            # 이미지 파일 수집
            original_images = sorted(glob.glob(os.path.join(original_folder_path, "*.jpg")) + 
                                   glob.glob(os.path.join(original_folder_path, "*.jpeg")) + 
                                   glob.glob(os.path.join(original_folder_path, "*.png")))
            
            model_images_list = []
            for folder_path in model_folder_paths:
                images = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) + 
                              glob.glob(os.path.join(folder_path, "*.jpeg")) + 
                              glob.glob(os.path.join(folder_path, "*.png")))
                model_images_list.append(images)
            
            # 프레임 개수 확인
            all_lengths = [len(original_images)] + [len(images) for images in model_images_list]
            num_frames = min(all_lengths)
            
            if num_frames == 0:
                raise Exception("폴더 중 하나 이상에 이미지가 없습니다.")
                
            self.log(f"총 프레임 수: {num_frames}")
            
            # 첫 이미지로 해상도 확인
            original_img = self.load_image(original_images[0])
            if original_img is None:
                raise Exception("원본 이미지 로드 실패")
                
            model_first_images = []
            for images in model_images_list:
                img = self.load_image(images[0])
                if img is None:
                    raise Exception("모델 이미지 로드 실패")
                model_first_images.append(img)
            
            # 최소 크기 찾기
            all_images = [original_img] + model_first_images
            height = min(img.shape[0] for img in all_images)
            width = min(img.shape[1] for img in all_images)
            
            self.log(f"이미지 크기: {width}x{height}")
            
            # 비디오 설정
            text_height = self.text_height.get()
            num_models = len(active_model_folders)
            
            # 프레임 크기 계산 (모델 수에 따라 동적으로)
            frame_width = width * num_models
            frame_height = height + (height + text_height)
            frame_size = (frame_width, frame_height)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(self.output_path.get(), fourcc, 
                                         self.frame_rate.get(), frame_size)
            
            # 프레임 처리
            for i in range(num_frames):
                # 진행률 업데이트
                progress_percent = (i / num_frames) * 100
                self.progress.set(progress_percent)
                self.status_text.set(f"처리 중... {i+1}/{num_frames} 프레임")
                
                # 원본 이미지 로드
                original_img = self.load_image(original_images[i])
                if original_img is None:
                    self.log(f"원본 프레임 {i} 로드 실패, 건너뜀")
                    continue
                
                # 모델 이미지들 로드
                model_imgs = []
                for j, images in enumerate(model_images_list):
                    img = self.load_image(images[i])
                    if img is None:
                        self.log(f"모델 {j+1} 프레임 {i} 로드 실패")
                        break
                    model_imgs.append(img)
                
                if len(model_imgs) != num_models:
                    continue
                    
                # 이미지 크기 조정
                original_img = cv2.resize(original_img, (width, height))
                model_imgs = [cv2.resize(img, (width, height)) for img in model_imgs]
                
                # 모델 이미지들에 텍스트 공간 추가 및 텍스트 추가
                model_imgs_with_text = []
                for j, (img, name) in enumerate(zip(model_imgs, model_names)):
                    # 텍스트 공간 추가
                    img_padded = cv2.copyMakeBorder(img, 0, text_height, 0, 0, 
                                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    # 텍스트 추가
                    img_with_text = self.add_text_to_image(img_padded, name)
                    model_imgs_with_text.append(img_with_text)
                
                # 프레임 결합
                # 상단: 원본 이미지 (중앙 정렬)
                # 하단: 모델 이미지들
                
                # 하단 행 생성
                bottom_row = np.hstack(model_imgs_with_text)
                
                # 원본 이미지 중앙 정렬을 위한 패딩
                pad_left = (bottom_row.shape[1] - original_img.shape[1]) // 2
                pad_right = bottom_row.shape[1] - original_img.shape[1] - pad_left
                
                original_padded = cv2.copyMakeBorder(
                    original_img, 0, 0, pad_left, pad_right, 
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
                
                # 상하 결합
                combined = np.vstack((original_padded, bottom_row))
                video_writer.write(combined)
                
            video_writer.release()
            
            self.progress.set(100)
            self.status_text.set("완료!")
            self.log(f"비디오 생성 완료: {self.output_path.get()}")
            
            messagebox.showinfo("완료", f"비디오가 성공적으로 생성되었습니다!\n{self.output_path.get()}")
            
        except Exception as e:
            self.log(f"오류 발생: {e}")
            messagebox.showerror("오류", f"비디오 생성 중 오류가 발생했습니다:\n{e}")
            
        finally:
            self.start_button.config(state='normal')
            self.progress.set(0)
            self.status_text.set("준비됨")

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoMergerApp(root)
    root.mainloop()