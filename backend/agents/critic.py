import logging

from .base_agent import BaseAgent

logger = logging.getLogger("Critic")

class Critic(BaseAgent):
    """Critic analyzes and compares ML results"""

    SYSTEM_PROMPT = """You are an ML Critic comparing two ML approaches.

Compare the results and provide:

ANALYSIS: Compare the two approaches

WINNER: architect or fedot

ARCHITECT_STRENGTHS:
- Strength 1
- Strength 2

ARCHITECT_WEAKNESSES:
- Weakness 1
- Weakness 2

FEEDBACK: What to improve next iteration"""

    async def execute(self, architect_results, fedot_results, comparison, profile, iteration):
        """Analyze and compare results"""
        try:
            logger.info(f"[Critic] Iteration {iteration} starting")
            
            context = f"""
Architect best score: {architect_results.get('best_score', 0):.4f}
Architect best model: {architect_results.get('best_model', 'unknown')}
Fedot score: {fedot_results.get('score', 0):.4f}
Comparison: {comparison.get('comparison_text', '')}
Iteration: {iteration}

Please compare these results and provide detailed feedback.
"""
            
            response = await self.call_ollama(context)
            
            if not response.get("success"):
                logger.warning("[Critic] LLM failed, using fallback")
                return self._fallback_analysis(architect_results, fedot_results)
            
            text = response.get("full_response", "")
            
            parsed = {
                "analysis": self.extract_field(text, "ANALYSIS"),
                "winner": self.extract_field(text, "WINNER").lower(),
                "strengths": self.extract_list(text, "ARCHITECT_STRENGTHS"),
                "weaknesses": self.extract_list(text, "ARCHITECT_WEAKNESSES"),
                "feedback": self.extract_field(text, "FEEDBACK"),
                "full_response": text  # Save full LLM response
            }
            
            logger.info(f"[Critic] Analysis complete")
            return parsed
            
        except Exception as e:
            logger.error(f"[Critic] Error: {str(e)}")
            return self._fallback_analysis(architect_results, fedot_results)

    def _fallback_analysis(self, architect_results, fedot_results):
        """Fallback when LLM fails"""
        arch_score = architect_results.get("best_score", 0)
        fedot_score = fedot_results.get("score", 0)
        
        return {
            "analysis": f"Architect: {arch_score:.4f}, Fedot: {fedot_score:.4f}",
            "winner": "architect" if arch_score > fedot_score else "fedot",
            "strengths": ["Successfully trained", "Reliable performance"],
            "weaknesses": ["No LLM analysis"],
            "feedback": "Continue optimization",
            "full_response": "Fallback analysis - LLM was unavailable"
        }