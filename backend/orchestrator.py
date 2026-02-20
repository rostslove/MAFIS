import logging
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from agents.architect import Architect
from agents.engineer import Engineer
from agents.critic import Critic
from agents.describe import Describe

logger = logging.getLogger("Orchestrator")


async def run_orchestration(
    csv_path: str,
    target_column: str,
    task_type: str = "classification",
    fedot_url: str = "http://fedot-server:8000",
    iterations: int = 2
) -> Dict[str, Any]:
    
    logger.info(f"{'='*60}")
    logger.info("STARTING ORCHESTRATION")
    logger.info(f"{'='*60}")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Target: {target_column}")
    logger.info(f"Task: {task_type}")
    logger.info(f"Fedot URL: {fedot_url}")
        
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded data: {df.shape} samples, {df.shape} features")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return {"error": str(e), "status": "failed"}
    
    if target_column not in df.columns:
        error = f"Target column '{target_column}' not found"
        logger.error(error)
        return {"error": error, "status": "failed"}
    
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    label_encoder = None
    if task_type == "classification" and y.dtype == "object":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        logger.info(f"Encoded target variable: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task_type == "classification" else None
    )
    
    logger.info(f"Split data: {X_train.shape} train, {X_val.shape} val")
    
    profile = {
        "n_samples": len(df),
        "n_features": len(X.columns),
        "task_type": task_type,
        "target_column": target_column,
        "features": list(X.columns),
        "issues": _detect_issues(df, target_column, task_type)
    }
    
    logger.info(f"Data issues detected: {profile['issues']}")
    
    
    architect = Architect(name="Architect")
    engineer = Engineer(name="Engineer", fedot_url=fedot_url)
    critic = Critic(name="Critic")
    describe = Describe(name="Describe")
        
    all_results = []
    prev_feedback = None
    
    for iteration in range(1, iterations + 1):
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}/{iterations}")
        logger.info(f"{'='*60}\n")
        
        try:
            # 1. ARCHITECT
            logger.info("[Step 1/4] Architect analyzing...")
            pipelines, architect_data = await architect.execute(
                profile, iteration, prev_feedback, task_type
            )
            logger.info(f"Architect suggested {len(pipelines)} pipelines")
            
            # 2. ENGINEER
            logger.info("[Step 2/4] Engineer training...")
            engineer_results = await engineer.execute(
                X_train, y_train, X_val, y_val, pipelines,
                csv_path, target_column, task_type
            )
            logger.info(f"Architect best: {engineer_results['best_score']:.4f}")
            logger.info(f"Fedot best: {engineer_results['fedot_result'].get('score', 0):.4f}")
            logger.info(f"Comparison: {engineer_results['comparison']['comparison_text']}")
            
            # 3. CRITIC
            logger.info("[Step 3/4] Critic analyzing...")
            critic_results = await critic.execute(
                engineer_results, engineer_results['fedot_result'],
                engineer_results['comparison'],
                profile, iteration
            )
            logger.info(f"Winner: {critic_results.get('winner', 'unknown').upper()}")
            logger.info(f"Feedback: {critic_results.get('feedback', 'N/A')[:100]}...")
            
            # 4. DESCRIBE
            logger.info("[Step 4/4] Describe explaining...")
            describe_results = await describe.execute(
                engineer_results, engineer_results['fedot_result'],
                critic_results, profile
            )
            logger.info(f"Solution: {describe_results.get('title', 'N/A')}")
            
            all_results.append({
                "iteration": iteration,
                "architect": architect_data,
                "engineer": engineer_results,
                "critic": critic_results,
                "describe": describe_results
            })
            
            prev_feedback = critic_results.get("feedback")
            
            logger.info(f"\nIteration {iteration} completed successfully\n")
        
        except Exception as e:
            logger.error(f"Iteration {iteration} failed: {e}")
            all_results.append({
                "iteration": iteration,
                "error": str(e),
                "status": "failed"
            })
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info("ORCHESTRATION COMPLETED")
    logger.info(f"{'='*60}\n")
    
    return {
        "status": "success",
        "profile": profile,
        "iterations": all_results,
        "best_iteration": _get_best_iteration(all_results),
        "summary": _create_summary(all_results)
    }


def _detect_issues(df: pd.DataFrame, target_column: str, task_type: str) -> List[str]:    
    issues = []
    
    missing_pct = (df.isnull().sum() / len(df) * 100).max()
    if missing_pct > 10:
        issues.append(f"Missing values: {missing_pct:.1f}%")
    
    if task_type == "classification":
        y = df[target_column]
        if len(y.unique()) <= 10:
            value_counts = y.value_counts()
            ratio = value_counts.min() / value_counts.max()
            if ratio < 0.3:
                issues.append(f"Class imbalance: {ratio:.2f}")
    
    if len(df) < 100:
        issues.append("Small dataset")
    elif len(df) > 1000000:
        issues.append("Large dataset")
    
    return issues if issues else ["None"]


def _get_best_iteration(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    best = None
    best_score = -1
    
    for result in all_results:
        if "error" not in result:
            score = result.get("engineer", {}).get("best_score", -1)
            if score > best_score:
                best_score = score
                best = result
    
    return best if best else (all_results[-1] if all_results else {})


def _create_summary(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    
    successful = [r for r in all_results if "error" not in r]
    
    if not successful:
        return {"status": "no successful iterations"}
    
    architect_scores = [
        r.get("engineer", {}).get("best_score", 0) for r in successful
    ]
    fedot_scores = [
        r.get("engineer", {}).get("fedot_result", {}).get("score", 0) for r in successful
    ]
    
    return {
        "total_iterations": len(all_results),
        "successful_iterations": len(successful),
        "avg_architect_score": sum(architect_scores) / len(architect_scores) if architect_scores else 0,
        "avg_fedot_score": sum(fedot_scores) / len(fedot_scores) if fedot_scores else 0,
        "max_architect_score": max(architect_scores) if architect_scores else 0,
        "max_fedot_score": max(fedot_scores) if fedot_scores else 0,
        "architect_wins_count": sum(1 for r in successful if r.get("engineer", {}).get("comparison", {}).get("architect_wins", False))
    }
