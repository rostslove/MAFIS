import logging
import asyncio
from .base_agent import BaseAgent
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

logger = logging.getLogger(__name__)


class Architect(BaseAgent):
    """Architect analyzes data and generates ML pipelines"""
    
    SYSTEM_PROMPT = """You are an ML Architect analyzing data for machine learning.

AVAILABLE PIPELINES:

For Regression:
- baseline_lr: LinearRegression with StandardScaler (simple, fast, good baseline)
- ridge: Ridge regression with StandardScaler (regularized linear, handles multicollinearity)
- svr: Support Vector Regression with StandardScaler (non-linear, good for complex patterns)
- xgb_reg: XGBoost regression with StandardScaler (powerful ensemble, handles non-linearity well)

For Classification:
- baseline_lr: LogisticRegression with StandardScaler (simple, fast, interpretable)
- svc: Support Vector Classifier with StandardScaler (non-linear decision boundaries)
- xgb_clf: XGBoost classifier with StandardScaler (powerful ensemble, handles imbalance)
- rf: RandomForest classifier with StandardScaler (robust, good feature importance)

Analyze the data and provide:

ANALYSIS: What issues or patterns you see (missing values, class imbalance, data size, etc.)

TASK: Is this regression or classification?

SELECTED_PIPELINES: Which 2-3 pipelines you recommend (comma-separated: e.g., baseline_lr, xgb_reg)

REASONING: Why these specific pipelines? Consider the data profile and issues.
"""

    async def execute(self, profile, iteration, prev_feedback, task_type):
        """
        Execute architect analysis and generate pipelines
        
        Args:
            X_train: Training features
            y_train: Training target
            profile: Data profile dict
            iteration: Current iteration number
            prev_feedback: Feedback from previous iteration
            task_type: "regression" or "classification"
        
        Returns:
            tuple: (pipelines dict, analysis dict with full_response)
        """
        try:
            logger.info(f"[Architect] Iteration {iteration} starting")
            
            context = self._prepare_context(profile, iteration, prev_feedback, task_type)
            response = await self.call_ollama(context)
            
            if not response.get("success"):
                logger.warning("[Architect] LLM call failed, using fallback")
                return self._get_pipelines(task_type), {
                    "analysis": "Data analysis fallback",
                    "task": task_type,
                    "reasoning": "Fallback mode",
                    "selected_pipelines": list(self._get_pipelines(task_type).keys())[:2],
                    "expected": "Unknown",
                    "full_response": "LLM failed - using fallback"
                }
            
            text = response.get("full_response", "")
            
            parsed = {
                "analysis": self.extract_field(text, "ANALYSIS"),
                "task": self.extract_field(text, "TASK"),
                "selected_pipelines": self.extract_pipelines(text, "SELECTED_PIPELINES", task_type),
                "reasoning": self.extract_field(text, "REASONING"),
                "expected": self.extract_field(text, "EXPECTED"),
                "full_response": text  # Save full LLM response for display
            }
            
            # Get only selected pipelines
            all_pipelines = self._get_pipelines(task_type)
            pipelines = {k: v for k, v in all_pipelines.items() if k in parsed["selected_pipelines"]}
            
            # If no valid pipelines selected, use all
            if not pipelines:
                pipelines = all_pipelines
                parsed["selected_pipelines"] = list(all_pipelines.keys())
            
            logger.info(f"[Architect] Analysis complete. Selected {len(pipelines)} pipelines: {parsed['selected_pipelines']}")
            
            return pipelines, parsed
        
        except Exception as e:
            logger.error(f"[Architect] Error: {str(e)}")
            return self._get_pipelines(task_type), {
                "analysis": f"Error: {str(e)}",
                "error": str(e),
                "full_response": f"Error: {str(e)}"
            }

    def _prepare_context(self, profile, iteration, prev_feedback, task_type):
        """Prepare context for LLM"""
        context = f"""Iteration: {iteration}
Task Type: {task_type}

Data Profile:
- Samples: {profile.get('n_samples', 'Unknown')}
- Features: {profile.get('n_features', 'Unknown')}
- Issues: {', '.join(profile.get('issues', []))}
- Feature list: {', '.join(profile.get('features', [])[:5])}...
"""
        
        if prev_feedback:
            context += f"\nPrevious iteration feedback: {prev_feedback}\n"
        
        context += "\nBased on this data profile, which pipelines would you select?"
        return context

    def _get_pipelines(self, task_type):
        """Get ML pipelines based on task type"""
        
        if task_type == "regression":
            return {
                "baseline_lr": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression())
                ]),
                "ridge": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0))
                ]),
                "svr": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel='rbf'))
                ]),
                "xgb_reg": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", XGBRegressor(n_estimators=100, random_state=42))
                ])
            }
        
        else:  # classification
            return {
                "baseline_lr": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=1000))
                ]),
                "svc": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", SVC(probability=True))
                ]),
                "xgb_clf": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", XGBClassifier(n_estimators=100, random_state=42))
                ]),
                "rf": Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
                ])
            }

    def extract_field(self, text, field_name):
        """Extract field from LLM response"""
        try:
            if field_name not in text:
                return "N/A"
            
            start = text.find(field_name) + len(field_name)
            # Skip colon and whitespace
            while start < len(text) and text[start] in ":\n \t":
                start += 1
            
            # Find end (next double newline or field name)
            end = text.find("\n\n", start)
            if end == -1:
                end = text.find("\n", start + 1)
            if end == -1:
                end = len(text)
            
            result = text[start:end].strip()
            return result[:500] if result else "N/A"
        except Exception as e:
            logger.warning(f"[Architect] Failed to extract {field_name}: {e}")
            return "N/A"

    def extract_pipelines(self, text, field_name, task_type):
        """Extract pipeline names from LLM response"""
        try:
            if field_name not in text:
                return []
            
            start = text.find(field_name) + len(field_name)
            # Skip colon and whitespace
            while start < len(text) and text[start] in ":\n \t":
                start += 1
            
            # Find end
            end = text.find("\n\n", start)
            if end == -1:
                end = text.find("\n", start + 1)
            if end == -1:
                end = len(text)
            
            section = text[start:end].strip()
            
            # Get valid pipeline names for this task
            all_pipelines = self._get_pipelines(task_type)
            valid_names = set(all_pipelines.keys())
            
            # Parse comma-separated or line-separated names
            items = []
            for part in section.split(","):
                name = part.strip().lower()
                if name in valid_names:
                    items.append(name)
            
            return items if items else []
        
        except Exception as e:
            logger.warning(f"[Architect] Failed to extract pipelines: {e}")
            return []