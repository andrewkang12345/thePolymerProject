#!/usr/bin/env python3
"""DDP training launcher. Wraps train_one.py to run on multiple GPUs."""
from __future__ import annotations

import os
import subprocess
import sys

def main():
    # Find how many GPUs
    import torch
    n_gpus = torch.cuda.device_count()

    # All args after this script name get forwarded to train_one.py
    train_args = sys.argv[1:]

    script = os.path.join(os.path.dirname(__file__), "train_one.py")

    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={n_gpus}",
        "--master_port=29500",
        script,
    ] + train_args

    print(f"Launching DDP with {n_gpus} GPUs: {' '.join(cmd)}")
    os.execvp(cmd[0], cmd)

if __name__ == "__main__":
    main()
