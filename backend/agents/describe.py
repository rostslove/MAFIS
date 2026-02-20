import logging

from .base_agent import BaseAgent

logger = logging.getLogger("Describe")

class Describe(BaseAgent):
    """Describe explains the ML solution"""

    SYSTEM_PROMPT = """You are a Solution Architect explaining ML solutions.

Explain the solution and provide:

TITLE: Solution name

ARCHITECTURE: Describe the pipeline

COMPONENTS:
- Component 1
- Component 2

PERFORMANCE: Analyze the metrics

VS_FEDOT: Compare with Fedot

RECOMMENDATIONS:
- Recommendation 1
- Recommendation 2"""

    async def execute(self, architect_results, fedot_results, critic_analysis, profile):
        """Explain the complete solution"""
        try:
            logger.info("[Describe] Starting explanation")
            
            context = f"""
Best pipeline: {architect_results.get('best_model', 'unknown')}
Architect Score: {architect_results.get('best_score', 0):.4f}
Fedot Score: {fedot_results.get('score', 0):.4f}
Winner: {critic_analysis.get('winner', 'architect')}
Critic feedback: {critic_analysis.get('feedback', '')}

Please explain this solution comprehensively.
"""
            
            response = await self.call_ollama(context)
            
            if not response.get("success"):
                logger.warning("[Describe] LLM failed, using fallback")
                return self._fallback_explanation()
            
            text = response.get("full_response", "")
            
            parsed = {
                "title": self.extract_field(text, "TITLE"),
                "architecture": self.extract_field(text, "ARCHITECTURE"),
                "components": self.extract_list(text, "COMPONENTS"),
                "performance": self.extract_field(text, "PERFORMANCE"),
                "vs_fedot": self.extract_field(text, "VS_FEDOT"),
                "recommendations": self.extract_list(text, "RECOMMENDATIONS"),
                "description": text[:500],
                "full_response": text  # Save full LLM response
            }
            
            logger.info("[Describe] Explanation complete")
            return parsed
            
        except Exception as e:
            logger.error(f"[Describe] Error: {str(e)}")
            return self._fallback_explanation()

    def _fallback_explanation(self):
        """Fallback when LLM fails"""
        return {
            "title": "ML Solution",
            "architecture": "Trained pipeline with preprocessing and model",
            "components": ["Data Preprocessing", "Feature Scaling", "Model Training"],
            "performance": "Model trained successfully",
            "vs_fedot": "Performance comparison completed",
            "recommendations": ["Deploy model", "Monitor performance", "Retrain periodically"],
            "description": "ML Solution",
            "full_response": "Fallback explanation - LLM was unavailable"
        }