import torch
import time
import argparse
from torch.utils.tensorboard import SummaryWriter  # 新增引用
from timm.models import create_model
from dynamic_tanh import convert_ln_to_dyt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vit_base_patch16_224')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dynamic_tanh', action='store_true', help='Enable DyT')
    parser.add_argument('--dyt_variant', default='scalar', type=str, help='DyT variant')
    parser.add_argument('--runs', type=int, default=200, help='Number of runs for averaging')
    parser.add_argument('--nb_classes', type=int, default=200)
    # [新增] Log 路径参数
    parser.add_argument('--log_dir', default=None, help='Path to save TensorBoard logs')
    return parser.parse_args()

def benchmark(args):
    device = torch.device('cuda')
    
    # 初始化 TensorBoard Writer
    writer = None
    if args.log_dir:
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"TensorBoard logging enabled: {args.log_dir}")

    print(f"--------------------------------------------------")
    print(f"Model: {args.model}")
    print(f"Mode: {'DyT (' + args.dyt_variant + ')' if args.dynamic_tanh else 'LayerNorm (Baseline)'}")
    print(f"Batch Size: {args.batch_size}")
    
    # 记录配置信息到 TensorBoard Text
    if writer:
        config_text = f"Model: {args.model}  \nMode: {'DyT-'+args.dyt_variant if args.dynamic_tanh else 'LayerNorm'}  \nBatch: {args.batch_size}"
        writer.add_text('Config', config_text, 0)

    model = create_model(args.model, pretrained=False, num_classes=args.nb_classes)
    if args.dynamic_tanh:
        model = convert_ln_to_dyt(model, variant=args.dyt_variant)
    
    model.to(device)
    
    input_tensor = torch.randn(args.batch_size, 3, 224, 224, device=device)
    target = torch.randint(0, args.nb_classes, (args.batch_size,), device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())

    # ==========================
    # Phase 1: 训练速度 (Train Speed)
    # ==========================
    model.train()
    print("\n[Phase 1] Benchmarking Training Step (Forward + Backward)...")
    
    for _ in range(20): # Warmup
        optimizer.zero_grad()
        loss = criterion(model(input_tensor), target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(args.runs):
        optimizer.zero_grad()
        loss = criterion(model(input_tensor), target)
        loss.backward()
        optimizer.step()
    end_event.record()
    torch.cuda.synchronize()
    
    total_train_time = start_event.elapsed_time(end_event) # ms
    avg_step_time = total_train_time / args.runs
    print(f" -> Avg Step Time: {avg_step_time:.2f} ms")
    
    if writer:
        writer.add_scalar('Benchmark/Step_Time_ms', avg_step_time, 0)

    # ==========================
    # Phase 2: 显存占用 (Peak Memory)
    # ==========================
    print("\n[Phase 2] Measuring Peak VRAM Usage...")
    torch.cuda.reset_peak_memory_stats()
    
    optimizer.zero_grad()
    loss = criterion(model(input_tensor), target)
    loss.backward()
    optimizer.step()
    
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
    print(f" -> Peak Memory: {max_mem:.2f} MB")
    
    if writer:
        writer.add_scalar('Benchmark/Peak_Memory_MB', max_mem, 0)

    # ==========================
    # Phase 3: 推论吞吐量 (Inference Throughput)
    # ==========================
    model.eval()
    print("\n[Phase 3] Benchmarking Inference Throughput (Forward only)...")
    
    with torch.no_grad(): # Warmup
        for _ in range(20):
            _ = model(input_tensor)
    torch.cuda.synchronize()

    start_event.record()
    with torch.no_grad():
        for _ in range(args.runs):
            _ = model(input_tensor)
    end_event.record()
    torch.cuda.synchronize()
    
    total_infer_time = start_event.elapsed_time(end_event)
    avg_infer_batch_time = total_infer_time / args.runs
    throughput = (args.batch_size * 1000) / avg_infer_batch_time
    
    print(f" -> Inference Throughput: {throughput:.2f} images/sec")
    print(f"--------------------------------------------------\n")

    if writer:
        writer.add_scalar('Benchmark/Throughput_imgs_sec', throughput, 0)
        writer.close()
        print("Results saved to TensorBoard.")

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args = get_args()
    benchmark(args)