# intelligence_assets/llm/client.py

import os
import logging
from typing import List, Dict, Any, Optional, Callable

# import client libraries
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    # We rely on the OpenAI library for both OpenAI and Ollama (via compatible API)
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
    from google.generativeai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

# Assuming env.py is correctly implemented in the root
from env import getenv

logger = logging.getLogger(__name__)

class LLMClientManager:
    def __init__(self):
        self._clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        # OpenAI and Ollama Initialization
        if OpenAI:
            # 1. Standard OpenAI (Cloud)
            try:
                openai_key = getenv("OPENAI_API_KEY", None)
                if openai_key:
                    self._clients["openai"] = OpenAI(api_key=openai_key)
                    logger.info("OpenAI client initialized.")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

            # 2. NEW: Ollama (Local, OpenAI Compatible API)
            self._initialize_ollama_client()

        # Anthropic
        if anthropic:
            try:
                anthropic_key = getenv("ANTHROPIC_API_KEY", None)
                if anthropic_key:
                    self._clients["anthropic"] = anthropic.Anthropic(api_key=anthropic_key)
                    logger.info("Anthropic client initialized")
            except Exception as e:
                logger.warning(f"Could not initialize Anthropic client: {e} ")

        # Google Gemini
        if genai:
            try:
                google_key = getenv("GOOGLE_API_KEY", None)
                if google_key:
                    genai.configure(api_key=google_key)
                    self._clients["google"] = genai
                    logger.info("Google Gemini API configured")
            except Exception as e:
                logger.warning(f"Could not initialize Google Gemini client: {e}")

    def _initialize_ollama_client(self):
        """Initializes the Ollama client using the OpenAI library."""
        try:
            # Default Ollama endpoint. Can be overridden by setting OLLAMA_BASE_URL in .env
            # The URL must end with /v1 for compatibility with the OpenAI client SDK.
            ollama_base_url = getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            if not ollama_base_url.endswith("/v1"):
                 logger.info(f"Ollama base URL detected without /v1 suffix. Adding /v1 for OpenAI compatibility.")
                 ollama_base_url = ollama_base_url.rstrip('/') + "/v1"

            # The OpenAI client expects an API key, but Ollama doesn't require one locally.
            self._clients["ollama"] = OpenAI(api_key="ollama", base_url=ollama_base_url)
            logger.info(f"Ollama client initialized (Base URL: {ollama_base_url}).")
        except Exception as e:
            logger.warning(f"Could not initialize Ollama client: {e}")

    def get_client(self, provider: str):
        client = self._clients.get(provider)
        if not client:
            if provider == "ollama":
                 raise ValueError("Ollama client requested but not initialized. Check connectivity to the Ollama host.")
            raise ValueError(f"Client for provider {provider} not initialized or supported.")
        return client

    def call_model(self, model_name: str, prompt: str, system_prompt: Optional[str] = None, response_format: str = "text", **kwargs) -> str:
        """
        A unified interface to call different LLM models.
        """
        # NEW: Routing logic for Ollama. We use the explicit prefix "ollama/".
        if model_name.startswith("ollama/"):
            # Remove the prefix before sending the request to the Ollama server
            actual_model_name = model_name.replace("ollama/", "", 1)
            return self._call_ollama(actual_model_name, prompt, system_prompt, response_format, **kwargs)
        
        if "gpt" in model_name:
            return self._call_openai(model_name, prompt, system_prompt, response_format, **kwargs)
        elif "claude" in model_name:
            return self._call_anthropic(model_name, prompt, system_prompt, response_format, **kwargs)
        elif "gemini" in model_name:
            return self._call_gemini(model_name, prompt, system_prompt, response_format, **kwargs)
        else:
            raise ValueError(f"Unknown or unsupported model name: {model_name}. To use Ollama models, prefix the name with 'ollama/' (e.g., 'ollama/qwen2.5:7b-instruct').")

    # NEW: Dedicated method for Ollama calls
    def _call_ollama(self, model_name: str, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs) -> str:
        """Calls a local Ollama model using the OpenAI compatible client."""
        client = self.get_client("ollama")
        # Local models often handle longer contexts well; increase default if not provided.
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096
        return self._execute_openai_compatible_call(client, model_name, prompt, system_prompt, response_format, **kwargs)

    def _call_openai(self, model_name: str, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs) -> str:
        client = self.get_client("openai")
        return self._execute_openai_compatible_call(client, model_name, prompt, system_prompt, response_format, **kwargs)

    # NEW: Reusable method for both OpenAI and Ollama calls (Refactored)
    def _execute_openai_compatible_call(self, client, model_name: str, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response_format_arg = None
        # Note: JSON mode support varies by model, especially in Ollama.
        if response_format == "json":
            response_format_arg = {"type": "json_object"}
            # Defensive Prompting: Ensure the prompt asks for JSON if JSON mode is requested, 
            # crucial for reliability, especially with local models.
            if not any("JSON" in msg['content'].upper() for msg in messages):
                 messages[-1]['content'] += "\n\nCRITICAL: Respond ONLY in the requested JSON format."

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format=response_format_arg,
                temperature=1,
                max_completion_tokens=kwargs.get("max_tokens", 4096) # Increased default max tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            # Handle potential connection errors if the server (like Ollama) isn't running
            logger.error(f"Error calling model {model_name}. Is the server running? Error: {e}")
            # Re-raise as a connection error for better handling upstream if needed
            raise ConnectionError(f"Failed to connect to the LLM server for model {model_name}: {e}")

    def _call_anthropic(self, model_name: str, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs) -> str:
        # (Anthropic implementation remains the same)
        client = self.get_client("anthropic")

        if response_format == "json":
            if system_prompt and "JSON" not in system_prompt:
                system_prompt += "\n\nCRITICAL: Respond ONLY in the requested JSON format."
            elif not system_prompt:
                system_prompt = "CRITICAL: Respond ONLY in the requested JSON format."

        response = client.messages.create(
            model=model_name,
            system=system_prompt if system_prompt else None,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 0.0)
        )

        if response.content and hasattr(response.content[0], 'text'):
            return response.content[0].text.strip()

        return ""

    def _call_gemini(self, model_name: str, prompt: str, system_prompt: Optional[str], response_format: str, **kwargs) -> str:
        # (Gemini implementation remains the same)
        if not genai_types:
             raise RuntimeError("Gemini types module missing for configuration.")
             
        client = self.get_client("google")

        model = client.GenerativeModel(model_name, system_instruction=system_prompt)

        config_kwargs = {
            "temperature": kwargs.get("temperature", 0),
            "max_output_tokens": kwargs.get("max_tokens", 4096)
        }

        if response_format == "json":
             config_kwargs["response_mime_type"] = "application/json"

        generation_config = genai_types.GenerationConfig(**config_kwargs)

        response = model.generate_content(
            contents=prompt,
            generation_config=generation_config
        )

        try:
            return response.text
        except ValueError as e:
            logger.warning(f"Gemini response empty or blocked. Error: {e}. Feedback: {response.prompt_feedback}")
            return ""
        
    def get_embeddings(self, texts: List[str], model="text-embedding-3-small", provider: str = "openai") -> List[List[float]]:
        """
        Get embeddings for a list of texts from the specified provider.
        
        Supports:
        - "openai": OpenAI's embedding models (paid)
        - "local": Local sentence-transformers models (free)
        """
        # Process texts consistently regardless of provider
        texts = [text.replace("\n", " ") for text in texts]
        
        # Use sentence-transformers if provider is "local"
        if provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
                
                # Get or initialize the model (lazy loading)
                if not hasattr(self, '_sentence_transformer'):
                    # all-MiniLM-L6-v2 is a good balance of quality and speed
                    # all-mpnet-base-v2 is higher quality but slower
                    self._sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                    
                # Generate embeddings and convert to list format
                embeddings = self._sentence_transformer.encode(texts, convert_to_tensor=False)
                return embeddings.tolist() if hasattr(embeddings, 'tolist') else [e.tolist() for e in embeddings]
            except ImportError:
                logger.warning("sentence-transformers not installed. Falling back to OpenAI embeddings.")
                provider = "openai"
        
        # OpenAI embeddings path (existing implementation)
        if provider != "openai":
            logger.info(f"Defaulting to OpenAI for embeddings to ensure consistency in novelty checks.")
            provider = "openai"

        if provider not in self._clients:
            raise ValueError(f"Provider {provider} not initialized for embeddings.")

        client = self.get_client(provider)

        # Basic check for standard embeddings interface
        if not hasattr(client, 'embeddings') or not hasattr(client.embeddings, 'create'):
            raise ValueError(f"Embeddings not supported by the configured client for {provider}.")

        response = client.embeddings.create(input=texts, model=model)
        # ensure embeddings are returned in the same order as input texts
        return [data.embedding for data in sorted(response.data, key=lambda x: x.index)]
# Renamed global instance for clarity and consistency
llm_manager = LLMClientManager()