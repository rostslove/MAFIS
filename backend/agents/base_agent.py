import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from ollama import chat

logger = logging.getLogger("BaseAgent")


class BaseAgent(ABC):    
    SYSTEM_PROMPT: str = ""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
    
    # ============== LLM CALL ==============
    async def call_ollama(
        self,
        user_message: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            self.logger.info(f"[{self.name}] Calling llama2...")
            
            system = system_prompt or self.SYSTEM_PROMPT
            
            response = chat(
                model="llama2",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message}
                ],
                stream=False
            )
            
            text = response.message.content
            self.logger.info(f"[{self.name}] Got response ({len(text)} chars)")
            
            return {
                "full_response": text,
                "success": True
            }
        
        except Exception as e:
            error = str(e)
            self.logger.error(f"[{self.name}] Error: {error}")
            return {
                "full_response": "",
                "success": False,
                "error": error
            }
        
    @staticmethod
    def extract_field(text: str, field_name: str) -> str:
        try:
            start = text.find(field_name + ":")
            if start == -1:
                return ""
            
            start += len(field_name) + 1
            while start < len(text) and text[start] in " \t":
                start += 1
            
            end = text.find("\n", start)
            if end == -1:
                end = len(text)
            
            return text[start:end].strip()
        except:
            return ""
    
    @staticmethod
    def extract_section(text: str, section_name: str) -> str:
        try:
            start = text.find(section_name + ":")
            if start == -1:
                return ""
            
            start += len(section_name) + 1
            while start < len(text) and text[start] in " \t\n":
                start += 1
            
            end = len(text)
            for line in text[start:].split("\n"):
                if ":" in line and line.split(":").isupper() and len(line.split(":")) > 1:
                    break
                end += len(line) + 1
            
            return text[start:end].strip()
        except:
            return ""
    
    @staticmethod
    def extract_list(text: str, section_name: str) -> List[str]:
        try:
            section = BaseAgent.extract_section(text, section_name)
            if not section:
                return []
            
            items = []
            for line in section.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    items.append(line[2:].strip())
                elif line.startswith("* "):
                    items.append(line[2:].strip())
            
            return items
        except:
            return []
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        pass
