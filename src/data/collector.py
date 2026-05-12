import json, logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
DATA_DIR = Path("data/raw")

def collect_samples() -> list[dict]:
    return [
        {"text": "این محصول عالی بود!", "label": 1, "source": "twitter"},
        {"text": "کیفیت افتضاح بود", "label": 0, "source": "twitter"},
        {"text": "خدمات خوبی دارن", "label": 1, "source": "telegram"},
        {"text": "دیر تحویل دادن", "label": 0, "source": "telegram"},
    ]

def save_samples(samples):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fp = DATA_DIR / f"batch_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(samples)} to {fp}")

if __name__ == "__main__":
    save_samples(collect_samples())
