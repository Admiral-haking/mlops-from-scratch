import json, logging, numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)
DATA_DIR = Path("data/raw")

def check():
    files = sorted(DATA_DIR.glob("batch_*.json"))
    if len(files) < 2:
        return False
    mid = len(files) // 2
    bl, re = [], []
    for f in files[:mid]:
        with open(f) as fh:
            bl.extend([i["text"] for i in json.load(fh)])
    for f in files[mid:]:
        with open(f) as fh:
            re.extend([i["text"] for i in json.load(fh)])
    r = abs(np.mean([len(t) for t in re]) - np.mean([len(t) for t in bl])) / max(np.mean([len(t) for t in bl]), 1)
    return r > 0.2

if __name__ == "__main__":
    print(f"DRIFT={'YES' if check() else 'NO'}")
