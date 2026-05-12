import random, mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MLOps Sentiment API")
_model_a, _model_b, _split = None, None, 0.9

class Req(BaseModel):
    text: str
class Resp(BaseModel):
    sentiment: str
    confidence: float
    version: str

def _load():
    global _model_a, _model_b
    try:
        exp = mlflow.get_experiment_by_name("persian-sentiment-analysis")
        if exp:
            runs = mlflow.search_runs([exp.experiment_id], order_by=["start_time desc"], max_results=2)
            if len(runs) > 0:
                _model_a = mlflow.sklearn.load_model(f"{runs.iloc[0].artifact_uri}/model")
            if len(runs) > 1:
                _model_b = mlflow.sklearn.load_model(f"{runs.iloc[1].artifact_uri}/model")
    except: pass

@app.on_event("startup")
async def startup():
    _load()

@app.get("/health")
async def health():
    return {"a": _model_a is not None, "b": _model_b is not None}

@app.post("/predict", response_model=Resp)
async def predict(r: Req):
    if _model_a is None:
        raise HTTPException(500, "No model")
    use_b = _model_b and random.random() > _split
    m = _model_b if use_b else _model_a
    v = "B" if use_b else "A"
    proba = m.predict_proba([r.text])[0]
    pred = m.predict([r.text])[0]
    return Resp(sentiment="pos" if pred else "neg", confidence=round(float(max(proba)), 4), version=v)
