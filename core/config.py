"""
Secure Configuration & Model Initialization Module for PaperPilot-AI.
Safely manages environment secrets and initializes LangChain Google GenAI clients.
Supports dynamic model configuration via GEMINI_MODEL in .env.
Default production model: gemini-3.8-flash
Alternative lightweight model: gemini-3.5-flash-lite
"""

import os
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load variables from .env file into os.environ
load_dotenv()

# Active, supported production Gemini model (default: gemini-3.8-flash)
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"


def get_gemini_model_name() -> str:
    """Retrieves the configured Gemini model name from GEMINI_MODEL env var or default."""
    configured_model = os.getenv("GEMINI_MODEL")
    if configured_model and configured_model.strip():
        return configured_model.strip()
    return DEFAULT_GEMINI_MODEL


def get_google_api_key(override_key: Optional[str] = None) -> Optional[str]:
    """Retrieves Google API key from explicit override or environment variables.
    
    Checks in order:
    1. Direct user override (e.g. from UI sidebar input)
    2. GOOGLE_API_KEY environment variable
    3. GEMINI_API_KEY environment variable (fallback alias)
    """
    candidate_key = None
    if override_key and override_key.strip():
        candidate_key = override_key.strip()
    else:
        env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if env_key and env_key.strip():
            candidate_key = env_key.strip()

    if candidate_key and candidate_key != "your_gemini_api_key_here":
        return candidate_key
    
    return None


def get_llm(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = 2048,
):
    """Initializes and returns a ChatGoogleGenerativeAI client if an API key is available.
    
    Returns:
        (llm_instance, error_message): A tuple containing the initialized model
        (or None) and a descriptive, safe error message if unavailable.
    """
    resolved_key = get_google_api_key(api_key)
    
    if not resolved_key:
        error_msg = (
            "Google API Key is not configured. Please set GOOGLE_API_KEY in your .env file "
            "or enter it directly in the application sidebar to activate LLM features."
        )
        return None, error_msg

    target_model = model_name or get_gemini_model_name()

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Initialize LangChain Google GenAI model
        llm = ChatGoogleGenerativeAI(
            model=target_model,
            google_api_key=resolved_key,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return llm, None
    except Exception as e:
        # Avoid leaking key or raw secret material in exceptions
        error_msg = (
            f"Failed to initialize ChatGoogleGenerativeAI with model '{target_model}': "
            f"{type(e).__name__} - {str(e)}. "
            "Please verify that GEMINI_MODEL is set to a supported model (e.g. gemini-3.8-flash or gemini-3.5-flash-lite)."
        )
        return None, error_msg
