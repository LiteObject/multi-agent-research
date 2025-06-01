# llm_factory.py
import os
from enum import Enum
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


class LLMType(Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


DEFAULTS = {
    "ollama_model": "qwen2.5:7b-instruct-q8_0",
    "openai_model": "gpt-4.1",
    "temperature": 0.1,
    "timeout": 360.0,
    "ollama_embed_model": "nomic-embed-text:latest",
    "openai_embed_model": "text-embedding-3-large"
}


def get_llm(llm_type: LLMType | str = None, **kwargs):
    """
    Factory method to return an LLM instance based on user preference.
    llm_type: LLMType enum or string ("ollama" or "openai") (default: from env LLM_TYPE)
    kwargs: parameters for the LLM constructor
    """
    if isinstance(llm_type, LLMType):
        llm_type_str = llm_type.value
    else:
        llm_type_str = (llm_type or os.getenv("LLM_TYPE", "ollama")).lower()

    if llm_type_str == "ollama":
        model_name = kwargs.get("model", DEFAULTS["ollama_model"])
        return Ollama(
            model=model_name,
            temperature=kwargs.get("temperature", DEFAULTS["temperature"]),
            request_timeout=kwargs.get("request_timeout", DEFAULTS["timeout"])
        )
    elif llm_type_str == "openai":
        model_name = kwargs.get("model", DEFAULTS["openai_model"])
        return OpenAI(
            api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY")),
            model=model_name,
            temperature=kwargs.get("temperature", DEFAULTS["temperature"]),
            request_timeout=kwargs.get("request_timeout", DEFAULTS["timeout"])
        )
    else:
        raise ValueError(
            f"Unsupported LLM type: {llm_type}. Supported types are: {[t.value for t in LLMType]}")


def get_embedding_model(llm_type: LLMType | str = None, **kwargs):
    """
    Factory method to return an embedding model instance based on user preference.
    llm_type: LLMType enum or string ("ollama" or "openai") (default: from env LLM_TYPE)
    kwargs: parameters for the embedding model constructor
    """
    if isinstance(llm_type, LLMType):
        llm_type_str = llm_type.value
    else:
        llm_type_str = (llm_type or os.getenv("LLM_TYPE", "ollama")).lower()

    if llm_type_str == "ollama":
        model_name = kwargs.get("embed_model", DEFAULTS["ollama_embed_model"])
        return OllamaEmbedding(model_name=model_name)
    elif llm_type_str == "openai":
        model_name = kwargs.get("embed_model", DEFAULTS["openai_embed_model"])
        return OpenAIEmbedding(
            api_key=kwargs.get("api_key", os.getenv("OPENAI_API_KEY")),
            model=model_name
        )
    else:
        raise ValueError(
            f"Unsupported LLM type for embedding: {llm_type}. Supported types are: {[t.value for t in LLMType]}")
