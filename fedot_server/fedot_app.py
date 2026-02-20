from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional, Any, Dict
import pandas as pd
import logging
from fedot_ind.api.main import FedotIndustrial
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fedot.Industrial API", version="1.0.0")


class TrainPredictParams(BaseModel):
    datapath: str
    target: str
    problem: Literal["regression", "classification"] = "classification"


class GetMetricsResponse(BaseModel):
    score: float
    predictions: list
    shape: tuple
    problem: str
    status: str = "success"
    error: Optional[str] = None


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "fedot-industrial",
        "version": "1.0.0"
    }


@app.post("/mcp", response_model=GetMetricsResponse)
async def train_and_predict(params: TrainPredictParams) -> Dict[str, Any]:
    try:
        logger.info(f"Training Fedot.Industrial model...")
        logger.info(f"Data path: {params.datapath}")
        logger.info(f"Target: {params.target}")
        logger.info(f"Problem: {params.problem}")
        
        if not Path(params.datapath).exists():
            raise FileNotFoundError(f"File not found: {params.datapath}")
        
        logger.info("Loading data...")
        data = pd.read_csv(params.datapath)
        logger.info(f"Data shape: {data.shape}")
        
        if params.target not in data.columns:
            raise ValueError(f"Target column '{params.target}' not found in data")
        
        y = data[params.target]
        X = data.drop(columns=[params.target])
        
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        logger.info("Initializing Fedot.Industrial...")
        
        industrial = FedotIndustrial(
            problem=params.problem,
                timeout=5,  
                n_jobs=2,
                logging_level=20
            )
        
        logger.info("Training model...")
        model = industrial.fit(input_data=(X, y))

        
        logger.info("Making predictions...")
        preds = industrial.predict(X)
        
        logger.info("Calculating metrics...")
        
        if params.problem == "classification":
            try:
                from sklearn.metrics import roc_auc_score
                if hasattr(industrial, 'predict_proba'):
                    preds_proba = industrial.predict_proba(features=X)
                    if preds_proba.ndim > 1:
                        preds_proba = preds_proba[:, 1]
                    score = float(roc_auc_score(y, preds_proba))
                else:
                    from sklearn.metrics import accuracy_score
                    score = float(accuracy_score(y, preds))
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}, using accuracy")
                from sklearn.metrics import accuracy_score
                score = float(accuracy_score(y, preds))
        else:
            from sklearn.metrics import r2_score
            score = float(r2_score(y, preds))

        
        logger.info(f"Model score: {score:.4f}")
        
        result = {
            "score": score,
            "predictions": preds.tolist() if hasattr(preds, 'tolist') else list(preds),
            "shape": data.shape,
            "problem": params.problem,
            "status": "success"
        }
        
        logger.info("Model training completed successfully")
        
        return result
    
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        raise HTTPException(status_code=404, detail=f"File not found: {e}")
    
    except ValueError as e:
        logger.error(f"Data error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid data: {e}")
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Model training failed: {str(e)}"
        )


@app.post("/status")
async def get_status() -> Dict[str, Any]:
    return {
        "status": "ready",
        "service": "fedot-industrial",
        "endpoints": {
            "/health": "GET - Health check",
            "/mcp": "POST - Train and predict",
            "/status": "POST - Get status"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
