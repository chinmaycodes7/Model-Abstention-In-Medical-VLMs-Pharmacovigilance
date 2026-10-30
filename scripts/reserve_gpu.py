"""Reserve GPU memory by holding a dummy tensor in VRAM."""
import argparse
import sys
import time
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Device ID")
    parser.add_argument("--gb", type=float, default=40.0, help="Gigabytes to reserve")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device(f"cuda:{args.device}")
    bytes_to_alloc = int(args.gb * (1024 ** 3))
    num_elements = bytes_to_alloc // 4  # float32 is 4 bytes

    print(f"Allocating {args.gb:.2f} GB on {torch.cuda.get_device_name(device)} (cuda:{args.device})...")
    sys.stdout.flush()

    try:
        reserved_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
        # Touch tensor so memory is physically committed
        reserved_tensor.fill_(1.0)
        print(f"Successfully locked {args.gb:.2f} GB on cuda:{args.device}. Holding memory...")
        sys.stdout.flush()
    except Exception as e:
        print(f"Failed to allocate {args.gb} GB: {e}")
        sys.stdout.flush()
        return

    try:
        while True:
            time.sleep(10)
    except (KeyboardInterrupt, SystemExit):
        print("Releasing GPU memory...")

if __name__ == "__main__":
    main()
