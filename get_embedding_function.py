from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    # Use local Ollama embeddings instead of AWS Bedrock
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings

