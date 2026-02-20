import asyncio
import aiohttp
import logging
from typing import Dict, Any
from .base_agent import BaseAgent
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

logger = logging.getLogger("Engineer")


class Engineer(BaseAgent):
    """Engineer trains pipelines and compares with Fedot.Industrial"""
    
    def __init__(self, name: str, fedot_url: str = "http://fedot-server:8000"):
        super().__init__(name=name)
        self.fedot_url = fedot_url

    async def execute(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        pipelines: Dict[str, Any],
        csv_path: str,
        target_column: str,
        task_type: str = "classification"
    ) -> Dict[str, Any]:
        """
        Train pipelines and call Fedot.Industrial
        
        Args:
            pipelines: Dict of pipeline objects (from architect)
            csv_path: Path to CSV file
            target_column: Target column name
            task_type: "regression" or "classification"
        
        Returns:
            Dict with best score and comparison
        """
        try:
            self.logger.info(f"[{self.name}] Training {len(pipelines)} pipelines...")
            
            architect_results = {
                "best_score": 0,
                "best_pipeline": None,
                "pipeline_results": [],
                "failed_pipelines": {}
            }
            
            # Train each pipeline
            for pipeline_name, pipeline_obj in pipelines.items():
                result = self._train_pipeline(
                    pipeline_name, pipeline_obj, X_train, y_train, X_val, y_val, task_type
                )
                architect_results["pipeline_results"].append(result)
                
                if "error" in result:
                    architect_results["failed_pipelines"][pipeline_name] = result["error"]
                elif result["score"] > architect_results["best_score"]:
                    architect_results["best_score"] = result["score"]
                    architect_results["best_pipeline"] = result["name"]
            
            self.logger.info(f"[{self.name}] Best Architect score: {architect_results['best_score']:.4f}")
            
            self.logger.info(f"[{self.name}] Calling Fedot.Industrial API...")
            fedot_result = await self._call_fedot(csv_path, target_column, task_type)
            
            comparison = self._compare_results(architect_results, fedot_result)
            
            return {
                "best_score": architect_results["best_score"],
                "best_model": architect_results["best_pipeline"],
                "pipeline_results": architect_results["pipeline_results"],
                "failed_pipelines": architect_results["failed_pipelines"],
                "fedot_score": fedot_result.get("score", 0),
                "fedot_result": fedot_result,
                "comparison": comparison,
                "status": "success"
            }
        
        except Exception as e:
            self.logger.error(f"[{self.name}] Error: {str(e)}")
            return {
                "best_score": 0,
                "status": "error",
                "error": str(e)
            }

    def _train_pipeline(self, name, pipeline_obj, X_train, y_train, X_val, y_val, task_type):
        """Train a single pipeline"""
        try:
            pipeline_obj.fit(X_train, y_train)
            y_pred = pipeline_obj.predict(X_val)
            
            # Get probabilities if available
            if hasattr(pipeline_obj, "predict_proba"):
                try:
                    y_pred_proba = pipeline_obj.predict_proba(X_val)[:, 1]
                except:
                    y_pred_proba = y_pred
            else:
                y_pred_proba = y_pred
            
            # Calculate metrics
            if task_type == "classification":
                score = roc_auc_score(y_val, y_pred_proba)
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
            else:
                score = r2_score(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)
                precision = recall = f1 = None
            
            return {
                "name": name,
                "score": float(score),
                "precision": float(precision) if precision else None,
                "recall": float(recall) if recall else None,
                "f1": float(f1) if f1 else None
            }
        
        except Exception as e:
            self.logger.error(f"[{self.name}] Failed to train {name}: {e}")
            return {
                "name": name,
                "score": 0,
                "error": str(e)
            }

    async def _call_fedot(self, csv_path: str, target_column: str, task_type: str) -> Dict[str, Any]:
        """Call Fedot.Industrial API"""
        try:
            url = f"{self.fedot_url}/mcp"
            payload = {
                "datapath": csv_path,
                "target": target_column,
                "problem": task_type
            }
            
            self.logger.info(f"[{self.name}] POST {url}")
            self.logger.info(f"[{self.name}] Payload: {payload}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"[{self.name}] Fedot success: {result.get('score', 0):.4f}")
                        return {
                            "score": float(result.get("score", 0)),
                            "predictions": result.get("predictions", []),
                            "shape": result.get("shape", None),
                            "status": "success"
                        }
                    else:
                        error = await response.text()
                        self.logger.error(f"[{self.name}] Fedot error: {response.status}")
                        self.logger.error(f"[{self.name}] Response: {error}")
                        return {
                            "score": 0,
                            "status": "error",
                            "error": f"HTTP {response.status}: {error}"
                        }
        
        except asyncio.TimeoutError:
            self.logger.error(f"[{self.name}] Fedot timeout after 300s")
            return {"score": 0, "status": "error", "error": "Timeout after 300s"}
        
        except Exception as e:
            self.logger.error(f"[{self.name}] Fedot call failed: {e}")
            return {"score": 0, "status": "error", "error": str(e)}

    def _compare_results(self, architect_results, fedot_result):
        """Compare architect vs fedot results"""
        arch_score = architect_results.get("best_score", 0)
        fedot_score = fedot_result.get("score", 0)
        
        return {
            "architect_score": float(arch_score),
            "fedot_score": float(fedot_score),
            "difference": float(arch_score - fedot_score),
            "architect_wins": arch_score > fedot_score,
            "comparison_text": f"Architect: {arch_score:.4f} vs Fedot: {fedot_score:.4f}"
        }
