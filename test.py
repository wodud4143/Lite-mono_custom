# import os
# import re
# import shutil
# import networks


# def save_network(modelname, dir):
#     # networks 패키지 경로
#     NETWORKS_DIR = os.path.dirname(networks.__file__)
#     INIT_FILE = os.path.join(NETWORKS_DIR, "__init__.py")
#     # 목적지 경로
#     TARGET_DIR = dir
#     os.makedirs(TARGET_DIR, exist_ok=True)
#     name = modelname
    
#     # 복사할 클래스와 새 파일명 매핑
#     targets = {
#         "LiteMono": f"{name}_encoder.py",
#         "DepthDecoder": f"{name}_decoder.py"
#     }
    
#     # 항상 복사할 추가 파일들
#     additional_files = ["core_layer.py", "custom_layers.py"]
    
#     # __init__.py 읽기
#     with open(INIT_FILE, "r", encoding="utf-8") as f:
#         lines = f.readlines()
    
#     # 주석 아닌 줄만 필터링
#     active_lines = [line for line in lines if not line.strip().startswith("#")]
    
#     # 정규식: from .파일 import 클래스
#     pattern = re.compile(r"from \.(\w+) import (\w+)")
#     matches = [pattern.search(line).groups() for line in active_lines if pattern.search(line)]
    
#     # 대상 클래스만 찾아서 복사
#     for fname, cls in matches:
#         if cls in targets:
#             src_path = os.path.join(NETWORKS_DIR, f"{fname}.py")
#             dst_path = os.path.join(TARGET_DIR, targets[cls])
#             if os.path.exists(src_path):
#                 shutil.copy2(src_path, dst_path)
#                 print(f"{cls} 정의된 {fname}.py → {dst_path} 로 복사 완료")
#             else:
#                 print(f"⚠ {fname}.py 없음 (건너뜀)")
    
#     # 추가 파일들 복사
#     for additional_file in additional_files:
#         src_path = os.path.join(NETWORKS_DIR, additional_file)
#         dst_path = os.path.join(TARGET_DIR, additional_file)
#         if os.path.exists(src_path):
#             shutil.copy2(src_path, dst_path)
#             print(f"{additional_file} → {dst_path} 로 복사 완료")
#         else:
#             print(f"⚠ {additional_file} 없음 (건너뜀)")


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
    
    def modify_imports(content, modelname):
        """파일 내용에서 from networks 구문을 from experiments.logs.modelname으로 변경"""
        # from networks import 패턴들을 찾아서 변경
        patterns_to_replace = [
            (r'from networks import', f'from experiments.logs.{modelname} import'),
            (r'from networks\.', f'from experiments.logs.{modelname}.'),
        ]
        
        modified_content = content
        for old_pattern, new_pattern in patterns_to_replace:
            modified_content = re.sub(old_pattern, new_pattern, modified_content)
        
        return modified_content
    
    def copy_and_modify_file(src_path, dst_path, modelname):
        """파일을 복사하면서 import 구문 수정"""
        with open(src_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # import 구문 수정
        modified_content = modify_imports(content, modelname)
        
        # 수정된 내용으로 새 파일 저장
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(modified_content)
    
    # 대상 클래스만 찾아서 복사 (import 구문 수정 포함)
    for fname, cls in matches:
        if cls in targets:
            src_path = os.path.join(NETWORKS_DIR, f"{fname}.py")
            dst_path = os.path.join(TARGET_DIR, targets[cls])
            if os.path.exists(src_path):
                copy_and_modify_file(src_path, dst_path, name)
                print(f"{cls} 정의된 {fname}.py → {dst_path} 로 복사 완료 (import 구문 수정됨)")
            else:
                print(f"⚠ {fname}.py 없음 (건너뜀)")
    
    # 추가 파일들 복사 (import 구문 수정 포함)
    for additional_file in additional_files:
        src_path = os.path.join(NETWORKS_DIR, additional_file)
        dst_path = os.path.join(TARGET_DIR, additional_file)
        if os.path.exists(src_path):
            copy_and_modify_file(src_path, dst_path, name)
            print(f"{additional_file} → {dst_path} 로 복사 완료 (import 구문 수정됨)")
        else:
            print(f"⚠ {additional_file} 없음 (건너뜀)")

# 사용 예시
# save_network("v4_1", "experiments/logs/v4_1")