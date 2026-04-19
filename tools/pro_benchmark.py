import torch
import time
import builtins          # 🚀 新增这一行
import numpy as np

builtins.np = np
from pointcept.engines.defaults import default_config_parser
from pointcept.models import build_model
from pointcept.datasets import build_dataset
# 🚀 修改 1：删除了这里报错的 collate_fn 导入

def profile_model():
    # 1. 解析你的 Config
    config_path = "configs/sensaturban/semseg-pt-v3m1-0-base.py"
    cfg = default_config_parser(config_path, options=None)
    
    print(f"========== Profiling Config: {config_path} ==========")

    # 2. 构建模型
    model = build_model(cfg.model).cuda()
    model.eval()

    # 计算 Params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Params] Total: {total_params / 1e6:.2f} M | Trainable: {trainable_params / 1e6:.2f} M")

    # ==========================================
    # 🚀 修改 2：彻底重写了数据加载部分，绕开 DataLoader 和 collate_fn
    # ==========================================
    print("Loading a real scan from the dataset...")
    val_dataset = build_dataset(cfg.data.val)
    raw_data = val_dataset[0] # 直接抽出验证集的第一帧
    
    data_dict = {}
    for k, v in raw_data.items():
        if isinstance(v, np.ndarray):
            data_dict[k] = torch.from_numpy(v).cuda(non_blocking=True)
        elif isinstance(v, torch.Tensor):
            data_dict[k] = v.cuda(non_blocking=True)
        else:
            data_dict[k] = v
            
    # 手动添加 Pointcept 稀疏运算必需的 batch 和 offset 信息
    num_points = data_dict['coord'].shape[0]
    data_dict['offset'] = torch.tensor([num_points], dtype=torch.int32).cuda()
    data_dict['batch'] = torch.zeros(num_points, dtype=torch.long).cuda()
    
    print(f"[Data] Successfully loaded and batched a scan with {num_points} points.")
    # ==========================================

    # ==========================================
    # 4. 测算显存 (Peak Memory)
    # ==========================================
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        _ = model(data_dict)
        
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[Memory] Peak Inference Memory: {peak_mem:.3f} GB")

    # ==========================================
    # 5. 测算推理延迟 (Latency) - 极其严谨的做法
    # ==========================================
    # 预热 (Warm-up)：让 GPU 频率拉满，cudnn 寻找最优算法
    print("Warming up GPU...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(data_dict)
            
    torch.cuda.synchronize() # 强制同步，等待所有预热任务完成

    num_iters = 100
    start_time = time.perf_counter() # 使用精度更高的计时器
    
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(data_dict)
            
    torch.cuda.synchronize() # 再次强制同步
    end_time = time.perf_counter()
    
    latency = (end_time - start_time) / num_iters * 1000 # 转为 ms
    fps = 1000.0 / latency
    print(f"[Latency] Average Inference Latency over {num_iters} runs: {latency:.2f} ms")
    print(f"[FPS] Throughput: {fps:.2f} scans/sec")

    # ==========================================
    # 6. 测算 FLOPs (使用 fvcore, 可选)
    # ==========================================
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, data_dict)
        flops.unsupported_ops_warnings(False) # 忽略不支持的算子警告
        total_flops = flops.total() / 1e9 # 转为 GFLOPs
        print(f"[FLOPs] Total FLOPs: {total_flops:.2f} G")
    except ImportError:
        print("[FLOPs] Install fvcore (pip install fvcore) to calculate FLOPs.")

if __name__ == "__main__":
    profile_model()