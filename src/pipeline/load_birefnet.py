import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForImageSegmentation

import contextlib
import io

import warnings
warnings.filterwarnings("ignore")

def load_birefnet(device="cuda"):
    # Silence stdout/stderr during model loading
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            trust_remote_code=True
        )

    torch.set_float32_matmul_precision("high")
    birefnet.to(device)
    birefnet.eval()
    birefnet.half()
    return birefnet



