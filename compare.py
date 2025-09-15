import time
import torch
import numpy as np
import itertools
import os
from torch.utils.data import DataLoader
from trainer import Trainer

class DataLoaderOptimizer:
    def __init__(self, options):
        self.opt = options
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")
        
        # 테스트할 파라미터 범위 정의
        self.param_grid = {
            'num_workers': [6, 7,8,9,10,11,12],
            'pin_memory': [True, False],
            'prefetch_factor': [1, 2,3, 4],
            'persistent_workers': [True, False]
        }
        
        
        # CPU 코어 수에 따라 num_workers 범위 조정
        cpu_count = os.cpu_count()
        self.param_grid['num_workers'] = [i for i in self.param_grid['num_workers'] if i <= cpu_count]
        
        print(f"CPU 코어 수: {cpu_count}")
        print(f"테스트할 num_workers 범위: {self.param_grid['num_workers']}")
        
        self.results = []
        
    def create_custom_trainer(self, num_workers, pin_memory, prefetch_factor, persistent_workers):
        """커스텀 DataLoader 설정으로 Trainer 생성"""
        
        # 기본 Trainer 초기화 (DataLoader 제외)
        trainer = Trainer.__new__(Trainer)
        trainer.opt = self.opt
        trainer.log_path = os.path.join(trainer.opt.log_dir, trainer.opt.model_name)
        
        # 모델 설정
        trainer.models = {}
        trainer.models_pose = {}
        trainer.parameters_to_train = []
        trainer.parameters_to_train_pose = []
        trainer.device = self.device
        
        # 기본 설정들 복사
        trainer.num_scales = len(trainer.opt.scales)
        trainer.frame_ids = len(trainer.opt.frame_ids)
        trainer.use_pose_net = not (trainer.opt.use_stereo and trainer.opt.frame_ids == [0])
        
        if trainer.opt.use_stereo:
            trainer.opt.frame_ids.append("s")
        
        # 모델 생성 (간단하게)
        import networks
        trainer.models["encoder"] = networks.LiteMono(model=trainer.opt.model,
                                                     drop_path_rate=trainer.opt.drop_path,
                                                     width=trainer.opt.width, 
                                                     height=trainer.opt.height)
        trainer.models["encoder"].to(trainer.device)
        
        trainer.models["depth"] = networks.DepthDecoder(trainer.models["encoder"].num_ch_enc,
                                                       trainer.opt.scales)
        trainer.models["depth"].to(trainer.device)
        
        # 데이터셋 설정
        import datasets
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                        "kitti_odom": datasets.KITTIOdomDataset}
        dataset_class = datasets_dict[trainer.opt.dataset]
        
        # 파일 경로
        from utils import readlines
        fpath = os.path.join(os.path.dirname(__file__), "splits", trainer.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        img_ext = '.png' if trainer.opt.png else '.jpg'
        
        # 데이터셋 생성
        train_dataset = dataset_class(trainer.opt.data_path, train_filenames,
                                     trainer.opt.height, trainer.opt.width,
                                     trainer.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        
        # 커스텀 DataLoader 생성
        # persistent_workers는 num_workers > 0일 때만 적용
        use_persistent = persistent_workers if num_workers > 0 else False
        # prefetch_factor도 num_workers > 0일 때만 적용
        use_prefetch = prefetch_factor if num_workers > 0 else 2
        
        trainer.train_loader = DataLoader(
            train_dataset, 
            batch_size=trainer.opt.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            prefetch_factor=use_prefetch,
            persistent_workers=use_persistent
        )
        
        return trainer
    
    def benchmark_single_config(self, config, num_batches=20, warmup_batches=3):
        """단일 설정에 대한 성능 측정"""
        num_workers = config['num_workers']
        pin_memory = config['pin_memory']
        prefetch_factor = config['prefetch_factor']
        persistent_workers = config['persistent_workers']
        
        print(f"\n테스트 중: workers={num_workers}, pin_memory={pin_memory}, "
              f"prefetch={prefetch_factor}, persistent={persistent_workers}")
        
        try:
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 커스텀 trainer 생성 (정상적인 Trainer 초기화 후 DataLoader만 교체)
            trainer = Trainer(self.opt)
            
            # 데이터셋 다시 생성
            import datasets
            from utils import readlines
            
            datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                            "kitti_odom": datasets.KITTIOdomDataset}
            dataset_class = datasets_dict[trainer.opt.dataset]
            
            fpath = os.path.join(os.path.dirname(__file__), "splits", trainer.opt.split, "{}_files.txt")
            train_filenames = readlines(fpath.format("train"))
            img_ext = '.png' if trainer.opt.png else '.jpg'
            
            train_dataset = dataset_class(trainer.opt.data_path, train_filenames,
                                         trainer.opt.height, trainer.opt.width,
                                         trainer.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
            
            # 새로운 DataLoader 설정으로 교체
            if num_workers == 0:
                trainer.train_loader = DataLoader(
                    train_dataset, 
                    batch_size=trainer.opt.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=True
                )
            else:
                trainer.train_loader = DataLoader(
                    train_dataset, 
                    batch_size=trainer.opt.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=True,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers
                )
            trainer.set_train()
            
            data_iter = iter(trainer.train_loader)
            
            # 워밍업
            for _ in range(warmup_batches):
                try:
                    inputs = next(data_iter)
                    with torch.no_grad():
                        outputs, losses = trainer.process_batch(inputs)
                    del inputs, outputs, losses
                except StopIteration:
                    data_iter = iter(trainer.train_loader)
                    break
            
            # 실제 측정
            batch_times = []
            data_loading_times = []
            
            total_start = time.time()
            
            for i in range(num_batches):
                try:
                    # 데이터 로딩 시간
                    data_start = time.time()
                    inputs = next(data_iter)
                    data_end = time.time()
                    data_loading_times.append(data_end - data_start)
                    
                    # 전체 배치 처리 시간
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    batch_start = time.time()
                    
                    with torch.no_grad():
                        outputs, losses = trainer.process_batch(inputs)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    batch_end = time.time()
                    
                    batch_time = batch_end - batch_start
                    batch_times.append(batch_time)
                    
                    del inputs, outputs, losses
                    
                except StopIteration:
                    print(f"  데이터 부족으로 {i} 배치에서 중단")
                    break
                except Exception as e:
                    print(f"  배치 처리 중 오류: {e}")
                    break
            
            total_end = time.time()
            total_time = total_end - total_start
            
            if len(batch_times) > 0:
                avg_batch_time = np.mean(batch_times)
                avg_data_loading = np.mean(data_loading_times)
                throughput = len(batch_times) / total_time
                
                result = {
                    'config': config.copy(),
                    'avg_batch_time': avg_batch_time,
                    'avg_data_loading': avg_data_loading,
                    'throughput': throughput,
                    'total_time': total_time,
                    'num_batches': len(batch_times),
                    'std_batch_time': np.std(batch_times),
                    'min_batch_time': np.min(batch_times),
                    'max_batch_time': np.max(batch_times)
                }
                
                print(f"  결과: 배치시간={avg_batch_time:.4f}초, 처리량={throughput:.2f} batch/s, "
                      f"데이터로딩={avg_data_loading:.4f}초")
                
                del trainer
                return result
            else:
                print("  측정 실패: 유효한 배치가 없음")
                del trainer
                return None
                
        except Exception as e:
            print(f"  설정 테스트 실패: {e}")
            return None
    
    def optimize(self, num_batches=15):
        """모든 파라미터 조합에 대해 최적화 수행"""
        print("=" * 70)
        print("DataLoader 파라미터 최적화 시작")
        print("=" * 70)
        
        # 모든 파라미터 조합 생성
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        print(f"총 {total_combinations}개 조합 테스트 예정")
        print(f"예상 소요 시간: 약 {total_combinations * 30 / 60:.1f}분")
        
        valid_results = []
        
        for i, combination in enumerate(itertools.product(*param_values)):
            config = dict(zip(param_names, combination))
            
            print(f"\n[{i+1}/{total_combinations}] ", end="")
            result = self.benchmark_single_config(config, num_batches)
            
            if result:
                valid_results.append(result)
                self.results.append(result)
        
        print(f"\n\n성공한 테스트: {len(valid_results)}/{total_combinations}")
        
        if valid_results:
            self.analyze_results(valid_results)
            return self.get_best_config(valid_results)
        else:
            print("모든 테스트가 실패했습니다.")
            return None
    
    def analyze_results(self, results):
        """결과 분석 및 출력"""
        print("\n" + "=" * 70)
        print("최적화 결과 분석")
        print("=" * 70)
        
        # 처리량 기준 상위 5개
        sorted_by_throughput = sorted(results, key=lambda x: x['throughput'], reverse=True)
        
        print("\n🏆 처리량 기준 상위 5개 설정:")
        print("-" * 70)
        for i, result in enumerate(sorted_by_throughput[:5]):
            config = result['config']
            print(f"{i+1}. 처리량: {result['throughput']:.2f} batch/s, "
                  f"배치시간: {result['avg_batch_time']:.4f}초")
            print(f"   workers={config['num_workers']}, pin_memory={config['pin_memory']}, "
                  f"prefetch={config['prefetch_factor']}, persistent={config['persistent_workers']}")
        
        # 배치 처리 시간 기준 상위 5개
        sorted_by_batch_time = sorted(results, key=lambda x: x['avg_batch_time'])
        
        print("\n⚡ 배치 처리 시간 기준 상위 5개 설정:")
        print("-" * 70)
        for i, result in enumerate(sorted_by_batch_time[:5]):
            config = result['config']
            print(f"{i+1}. 배치시간: {result['avg_batch_time']:.4f}초, "
                  f"처리량: {result['throughput']:.2f} batch/s")
            print(f"   workers={config['num_workers']}, pin_memory={config['pin_memory']}, "
                  f"prefetch={config['prefetch_factor']}, persistent={config['persistent_workers']}")
        
        # 파라미터별 평균 성능 분석
        print("\n📊 파라미터별 평균 성능:")
        print("-" * 50)
        
        # num_workers별 분석
        workers_performance = {}
        for result in results:
            workers = result['config']['num_workers']
            if workers not in workers_performance:
                workers_performance[workers] = []
            workers_performance[workers].append(result['throughput'])
        
        print(f"\nnum_workers:")
        for workers, throughputs in workers_performance.items():
            avg_throughput = np.mean(throughputs)
            print(f"  {workers}: {avg_throughput:.2f} batch/s (평균)")
        
        # pin_memory별 분석 (num_workers > 0인 경우만)
        pin_performance = {'True': [], 'False': []}
        for result in results:
            if result['config']['num_workers'] > 0:
                pin_value = str(result['config']['pin_memory'])
                pin_performance[pin_value].append(result['throughput'])
        
        print(f"\npin_memory (workers>0):")
        for pin_value, throughputs in pin_performance.items():
            if throughputs:
                avg_throughput = np.mean(throughputs)
                print(f"  {pin_value}: {avg_throughput:.2f} batch/s (평균)")
        
        # prefetch_factor별 분석 (num_workers > 0인 경우만)
        prefetch_performance = {}
        for result in results:
            if result['config']['num_workers'] > 0 and result['config']['prefetch_factor']:
                prefetch = result['config']['prefetch_factor']
                if prefetch not in prefetch_performance:
                    prefetch_performance[prefetch] = []
                prefetch_performance[prefetch].append(result['throughput'])
        
        print(f"\nprefetch_factor (workers>0):")
        for prefetch, throughputs in prefetch_performance.items():
            avg_throughput = np.mean(throughputs)
            print(f"  {prefetch}: {avg_throughput:.2f} batch/s (평균)")
    
    def get_best_config(self, results):
        """최적 설정 반환"""
        if not results:
            return None
        
        # 처리량 기준 최적 설정
        best_result = max(results, key=lambda x: x['throughput'])
        
        print("\n🎯 최적 설정 (처리량 기준):")
        print("=" * 50)
        config = best_result['config']
        print(f"num_workers: {config['num_workers']}")
        print(f"pin_memory: {config['pin_memory']}")
        print(f"prefetch_factor: {config['prefetch_factor']}")
        print(f"persistent_workers: {config['persistent_workers']}")
        print(f"\n성능:")
        print(f"  처리량: {best_result['throughput']:.2f} batch/s")
        print(f"  평균 배치 시간: {best_result['avg_batch_time']:.4f}초")
        print(f"  데이터 로딩 시간: {best_result['avg_data_loading']:.4f}초")
        
        return best_result
    
    def quick_test(self, configs_to_test=None):
        """빠른 테스트 (주요 설정들만)"""
        if configs_to_test is None:
            # 추천 설정들만 테스트
            configs_to_test = [
                {'num_workers': 0, 'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': False},
                {'num_workers': 2, 'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True},
                {'num_workers': 4, 'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True},
                {'num_workers': 4, 'pin_memory': False, 'prefetch_factor': 2, 'persistent_workers': True},
                {'num_workers': 6, 'pin_memory': True, 'prefetch_factor': 2, 'persistent_workers': True},
            ]
        
        print("빠른 테스트 모드 - 주요 설정들만 테스트")
        print(f"총 {len(configs_to_test)}개 설정 테스트")
        
        results = []
        for i, config in enumerate(configs_to_test):
            print(f"\n[{i+1}/{len(configs_to_test)}] ", end="")
            result = self.benchmark_single_config(config, num_batches=10)
            if result:
                results.append(result)
        
        if results:
            self.analyze_results(results)
            return self.get_best_config(results)
        return None

# 사용 예시
if __name__ == "__main__":
    # options 설정 (실제 사용 시 적절히 수정)
    from options import LiteMonoOptions
    
    opts = LiteMonoOptions()
    options = opts.parse()
    
    # 최적화 실행
    optimizer = DataLoaderOptimizer(options)
    
    print("1. 빠른 테스트 (추천)")
    print("2. 전체 최적화 (시간 오래 걸림)")
    choice = input("선택하세요 (1 or 2): ")
    
    if choice == "1":
        best_config = optimizer.quick_test()
    else:
        best_config = optimizer.optimize()
    
    if best_config:
        print("\n최적화 완료! 위의 설정을 trainer.py에 적용하세요.")
    else:
        print("\n최적화 실패")