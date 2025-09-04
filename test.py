import os
import re
import shutil
import networks


def save_network(modelname, dir):
    # networks 패키지 경로
    NETWORKS_DIR = os.path.dirname(networks.__file__)
    INIT_FILE = os.path.join(NETWORKS_DIR, "__init__.py")
    # 목적지 경로
    TARGET_DIR = dir
    os.makedirs(TARGET_DIR, exist_ok=True)
    name = modelname
    
    # 복사할 클래스와 새 파일명 매핑
    targets = {
        "LiteMono": f"{name}_encoder.py",
        "DepthDecoder": f"{name}_decoder.py"
    }
    
    # 항상 복사할 추가 파일들
    additional_files = ["core_layer.py", "custom_layers.py"]
    
    # __init__.py 읽기
    with open(INIT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 주석 아닌 줄만 필터링
    active_lines = [line for line in lines if not line.strip().startswith("#")]
    
    # 정규식: from .파일 import 클래스
    pattern = re.compile(r"from \.(\w+) import (\w+)")
    matches = [pattern.search(line).groups() for line in active_lines if pattern.search(line)]
    
    # 대상 클래스만 찾아서 복사
    for fname, cls in matches:
        if cls in targets:
            src_path = os.path.join(NETWORKS_DIR, f"{fname}.py")
            dst_path = os.path.join(TARGET_DIR, targets[cls])
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"{cls} 정의된 {fname}.py → {dst_path} 로 복사 완료")
            else:
                print(f"⚠ {fname}.py 없음 (건너뜀)")
    
    # 추가 파일들 복사
    for additional_file in additional_files:
        src_path = os.path.join(NETWORKS_DIR, additional_file)
        dst_path = os.path.join(TARGET_DIR, additional_file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"{additional_file} → {dst_path} 로 복사 완료")
        else:
            print(f"⚠ {additional_file} 없음 (건너뜀)")