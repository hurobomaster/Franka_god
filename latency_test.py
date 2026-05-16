import torch
import time
import numpy as np
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs import PreTrainedConfig

def test_latency():
    device = torch.device("cpu")
    model_path = "./model/act_franka_insert_baseline/checkpoints/060000/pretrained_model"
    
    config = PreTrainedConfig.from_pretrained(model_path)
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(model_path, config=config)
    policy.to(device)
    policy.eval()

    # Create dummy images and state
    state = torch.randn(1, 8, device=device)
    front = torch.randn(1, 3, 480, 640, device=device)
    wrist = torch.randn(1, 3, 480, 640, device=device)
    
    batch = {
        "observation.state": state,
        "observation.images.front": front,
        "observation.images.wrist": wrist
    }

    print(f"Policy type: {type(policy)}")
    
    # Warmup
    for _ in range(5):
        with torch.no_grad():
            # Use forward or specific model components if select_action is too high level
            # But ACTPolicy should use images. Check how it processes images.
            _ = policy.select_action(batch)

    # Benchmark
    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        with torch.no_grad():
            output = policy.select_action(batch)
        latencies.append((time.perf_counter() - start) * 1000)

    avg = np.mean(latencies)
    median = np.median(latencies)
    p95 = np.percentile(latencies, 95)
    fps = 1000 / avg

    print(f"Action shape: {output.shape}")
    print(f"Avg: {avg:.2f} ms")
    print(f"Median: {median:.2f} ms")
    print(f"P95: {p95:.2f} ms")
    print(f"Max FPS: {fps:.2f}")

if __name__ == "__main__":
    test_latency()
