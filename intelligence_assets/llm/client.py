# client.py for LLM clients

import os
import logging
from typing import List, Dict, Any, Optional, Callable

# import client libraries

try:
    import anthropic
except:
    anthropic = None

try: 
    from openai import OpenAI
except:
    OpenAI = None

try: 
    import google.generativeai as genai
    from google.generativeai import types as genai_types
except:
    genai = None
    genai_types = None

from env import getenv

logger = logging.getLogger(__name__)

class LLMClientManager:
    def __init__(self):
        self._clients = {}
        self.initialize_clients()

    def _initialive_clients(self):
        # OpenAI
        if OpenAI:
            try:
                openai_key = getenv("OPEN_API_KEY",None)
            except Exception as e:
                logger.warning(f"Could not initailize OpenAI client: {e}")
        
        # Anthropic
        if anthropic:
            try:
                anthropic_key = getenv("ANTHROPIC_API_KEY", None)
                if anthropic_key:
                    self._clients["anthropic"] = anthropic.Anthropic(api_key = anthropic_key )
                    logger.info("Anthropic client initialized")
            except Exception as e:
                logger.warning(f{"Could not initialize Anthropic client: {e} "})

        
        # Google Gemini
        if genai:
            try:
                google_key = getenv("GOOGLE_API_KEY", None)
                if google_key:
                    genai.configure(api_key = google_key)
                    self._clients["google"] = genai
                    logger.info("Google Gemini API configured")
            except Exception as e:
                logger.warning(f"Could not initialize Google Gemini client: {e}")


        def get_client(self, provider: str):
            client = self._clients.get(provider)
            if not client:
                raise ValueError(f"Client for provider {provider} not initialized.")
            return client
        
        def call_model(self, model_name: str, prompt: str, system_prompt: Optional[str] = None, response_format: str = "text", **kwargs) -> str:
            """
            A unified interface to call different LLM models.
            """
            # Simple routing based on model name prefixes

            if "gpt" in model_name:
                return self._call_openai(model_name, prompt, system_prompt, response_format, **kwargs)
            elif "claude" in model_name:
                return self._call_anthropic(model_name, prompt, system_prompt, response_format, **kwargs)
            elif "gemini" in model_name:
                return self.__call_gemini(model_name, prompt, system_prompt, response_format, **kwargs)
            else:
                raise ValueError(f"Uknown or unsupported model name: {model_name}")
            
            




