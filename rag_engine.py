import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from openai import APIConnectionError, APIStatusError, AuthenticationError
from google import genai
from google.genai import types

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredPowerPointLoader,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Original RAG architecture — unchanged
# Loader -> Chunk -> MiniLM -> Chroma -> Similarity Retrieval
# ---------------------------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "doc_intel"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".csv"}

# ASCII-safe internal model IDs. Display names belong in app.py.
MODEL_AZURE = "azure_gpt41_mini"
MODEL_GEMINI = "gemini_free"
MODEL_OPENROUTER = "openrouter_free"

SYSTEM_PROMPT = """You are DocuChat, a document-grounded AI assistant.

Answer the user's question using ONLY the supplied document context.
Do not use outside knowledge to fill missing information.

If the answer is not contained in the context, say exactly:
"I couldn't find that information in the uploaded document."

Be clear and concise. Use bullets or short sections when helpful.

DOCUMENT CONTEXT:
{context}
"""


def load_document(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".pptx":
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        loader = CSVLoader(file_path, encoding="utf-8")

    docs = loader.load()

    if not docs or not "".join(d.page_content for d in docs).strip():
        raise RuntimeError(
            "The document contains no extractable text. A scanned PDF may require OCR."
        )

    for doc in docs:
        doc.metadata.setdefault("source", file_path)

    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise RuntimeError("Chunking produced zero chunks.")

    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _delete_existing_collection() -> None:
    if not Path(CHROMA_DIR).exists():
        return

    try:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    except Exception:
        import shutil
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)


def index_document(file_path: str) -> Chroma:
    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    embeddings = get_embeddings()

    _delete_existing_collection()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )


def retrieve_documents(vectorstore: Chroma, question: str) -> List[Document]:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    return retriever.invoke(question)


def format_docs(docs: List[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Provider layer
# ---------------------------------------------------------------------------

def _azure_client() -> Tuple[OpenAI, str]:
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini").strip()

    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is missing from .env")
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is missing from .env")
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is missing from .env")

    if endpoint.endswith("/openai/v1"):
        base_url = endpoint
    else:
        base_url = endpoint + "/openai/v1"

    return OpenAI(api_key=api_key, base_url=base_url), deployment


def _gemini_client() -> Tuple[genai.Client, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from .env. Create a key in Google AI Studio."
        )

    if not model:
        raise RuntimeError("GEMINI_MODEL is missing from .env")

    return genai.Client(api_key=api_key), model


def _openrouter_client() -> Tuple[OpenAI, str]:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing from .env. Create a key in OpenRouter."
        )

    if not model:
        raise RuntimeError("OPENROUTER_MODEL is missing from .env")

    return (
        OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "DocuChat",
            },
        ),
        model,
    )


def _client_for_model(model_label: str):
    if model_label == MODEL_AZURE:
        client, model = _azure_client()
        return "openai", client, model, "Microsoft Foundry"

    if model_label == MODEL_GEMINI:
        client, model = _gemini_client()
        return "gemini", client, model, "Google Gemini"

    if model_label == MODEL_OPENROUTER:
        client, model = _openrouter_client()
        return "openai", client, model, "OpenRouter"

    raise ValueError(f"Unknown model selection: {model_label}")


def _messages(question: str, docs: List[Document]) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=format_docs(docs)),
        },
        {
            "role": "user",
            "content": question,
        },
    ]


def stream_question(
    vectorstore: Chroma,
    question: str,
    model_label: str,
) -> Tuple[Iterator[str], List[Document], str]:
    """Retrieve document context once and stream from Azure, Gemini, or OpenRouter."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    source_docs = retrieve_documents(vectorstore, question)
    provider_type, client, model, provider = _client_for_model(model_label)
    messages = _messages(question, source_docs)

    if provider_type == "gemini":
        system_instruction = messages[0]["content"]
        user_content = messages[1]["content"]

        def gemini_stream() -> Iterator[str]:
            try:
                response_stream = client.models.generate_content_stream(
                    model=model,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.0,
                        max_output_tokens=1500,
                    ),
                )
                for chunk in response_stream:
                    if getattr(chunk, "text", None):
                        yield chunk.text
            except Exception as exc:
                raise RuntimeError(
                    f"{provider} request failed for model '{model}': {exc}"
                ) from exc

        return gemini_stream(), source_docs, f"{provider} · {model}"

    try:
        response_stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=1500,
            stream=True,
        )
    except AuthenticationError as exc:
        if provider == "OpenRouter":
            raise RuntimeError(
                "OpenRouter authentication failed. Check OPENROUTER_API_KEY in .env."
            ) from exc
        raise RuntimeError(
            "Microsoft Foundry authentication failed. Check the Azure key, "
            "project endpoint, and deployment name."
        ) from exc
    except APIConnectionError as exc:
        raise RuntimeError(f"{provider} connection failed: {exc}") from exc
    except APIStatusError as exc:
        raise RuntimeError(f"{provider} returned HTTP {exc.status_code}: {exc}") from exc

    def token_stream() -> Iterator[str]:
        try:
            for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as exc:
            raise RuntimeError(
                f"{provider} request failed for model '{model}': {exc}"
            ) from exc

    return token_stream(), source_docs, f"{provider} · {model}"


def ask_question(
    vectorstore: Chroma,
    question: str,
    model_label: str,
) -> Dict[str, Any]:
    stream, sources, provider_model = stream_question(
        vectorstore,
        question,
        model_label,
    )

    answer = "".join(stream).strip()

    return {
        "answer": answer,
        "sources": sources,
        "model": provider_model,
    }
