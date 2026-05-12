import json, logging
from pathlib import Path
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)
DATA_DIR = Path("data/raw")

def train():
    texts, labels = [], []
    for f in sorted(DATA_DIR.glob("batch_*.json")):
        with open(f) as fh:
            for i in json.load(fh):
                texts.append(i["text"]); labels.append(i["label"])
    split = int(len(texts) * 0.8)
    pipe = Pipeline([("vec", TfidfVectorizer(max_features=5000)), ("clf", LogisticRegression())])
    pipe.fit(texts[:split], labels[:split])
    acc = accuracy_score(labels[split:], pipe.predict(texts[split:]))
    mlflow.set_experiment("persian-sentiment-analysis")
    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipe, "model")
    logger.info(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train()
