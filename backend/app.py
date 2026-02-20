from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import logging
import tempfile
import os
from typing import Optional

from orchestrator import run_orchestration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Architect System", version="1.0.0")


class OrchestrationRequest(BaseModel):
    csv_path: str
    target_column: str
    task_type: str = "classification"
    fedot_url: str = "http://fedot-server:8000"
    iterations: int = 2


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ml-architect-backend"}


@app.get("/config")
async def get_config():
    """Get system configuration"""
    return {
        "agents": ["Architect", "Engineer", "Critic", "Describe"],
        "llm_model": "llama2",
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "fedot_host": os.getenv("FEDOT_HOST", "http://fedot-server:8000")
    }


@app.post("/orchestrate")
async def orchestrate(request: OrchestrationRequest):    
    try:
        logger.info(f"Starting orchestration: {request.csv_path}")
        
        result = await run_orchestration(
            csv_path=request.csv_path,
            target_column=request.target_column,
            task_type=request.task_type,
            fedot_url=request.fedot_url,
            iterations=request.iterations
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        logger.error(f"Orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orchestrate/file")
async def orchestrate_file(
    file: UploadFile = File(...),
    target_column: str = None,
    task_type: str = "classification"
):    
    if not target_column:
        raise HTTPException(status_code=400, detail="target_column required")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            result = await run_orchestration(
                csv_path=tmp_path,
                target_column=target_column,
                task_type=task_type,
                fedot_url=os.getenv("FEDOT_HOST", "http://fedot-server:8000"),
                iterations=2
            )
            
            return JSONResponse(content=result)
        
        finally:
            os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"File orchestration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
