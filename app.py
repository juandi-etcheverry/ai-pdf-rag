import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import requests
from dotenv import load_dotenv


@dataclass
class Config:
    openrouter_api_key: str
    embedding_model: str = "intfloat/multilingual-e5-small"
    openrouter_model: str = "x-ai/grok-4-fast:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"


def load_config() -> Config:
    load_dotenv()
    embedding_model = os.getenv("EMBEDDING_MODEL") or "intfloat/multilingual-e5-small"
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_model = os.getenv("OPENROUTER_MODEL") or "x-ai/grok-4-fast:free"
    
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    return Config(
        embedding_model=embedding_model,
        openrouter_api_key=openrouter_api_key,
        openrouter_model=openrouter_model,
    )


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_paragraphs(text: str) -> List[str]:
    # Divide por renglones en blanco; filtra parrafos vacios
    raw_parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in raw_parts if p]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def sha256_of(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def ensure_cache_dir() -> str:
    cache_dir = os.path.join(".cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def cache_path_for(doc_path: str, embedding_model: str, doc_hash: str) -> str:
    safe_name = hashlib.sha1(f"{doc_path}|{embedding_model}|{doc_hash}".encode("utf-8")).hexdigest()
    return os.path.join(ensure_cache_dir(), f"embeddings_{safe_name}.json")


_LOCAL_EMBEDDER = None

def get_local_embedder(model_id: str):
    global _LOCAL_EMBEDDER
    if _LOCAL_EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _LOCAL_EMBEDDER = SentenceTransformer(model_id)
    return _LOCAL_EMBEDDER


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def embed_texts(texts: List[str], cfg: Config, is_query: bool) -> np.ndarray:
    # e5 requiere prefijos 'query: ' / 'passage: '
    prefix = "query: " if is_query else "passage: "
    processed = [prefix + t for t in texts]
    model = get_local_embedder(cfg.embedding_model)
    vectors = model.encode(processed, normalize_embeddings=True, convert_to_numpy=True)
    return vectors.astype(np.float32)


def most_similar_paragraphs(query_vec: np.ndarray, para_vecs: np.ndarray, top_k: int = 3) -> List[Tuple[int, float]]:
    # Calcula similitud coseno con todas las filas
    sims = []
    for i in range(para_vecs.shape[0]):
        sims.append((i, cosine_similarity(query_vec, para_vecs[i])))
    
    # Ordena por similitud descendente y toma los top_k
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:min(top_k, len(sims))]


def call_openrouter_api(query: str, context_paragraphs: List[Tuple[str, float]], cfg: Config) -> str:
    context_parts = []
    for i, (paragraph, similarity) in enumerate(context_paragraphs, 1):
        context_parts.append(f"Context {i} (similarity: {similarity:.4f}):\n{paragraph}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""Eres un asistente que responde preguntas basado en el contexto proporcionado. Usa los párrafos de contexto abajo para responder la pregunta del usuario. Prioriza información de los párrafos con mayores similitudes, pero considera todo el contexto proporcionado.

Contexto:
{context}

Pregunta del usuario: {query}

Proporciona una respuesta completa basado en el contexto. Si el contexto no contiene suficiente información para responder la pregunta, menciona qué información está disponible y qué podría estar faltando."""
    
    headers = {
        "Authorization": f"Bearer {cfg.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/user/ai-pdf-rag",
        "X-Title": "AI PDF RAG System"
    }
    
    payload = {
        "model": cfg.openrouter_model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }
    
    try:
        response = requests.post(
            f"{cfg.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        return f"Error calling OpenRouter API: {str(e)}"
    except (KeyError, IndexError) as e:
        return f"Error parsing OpenRouter response: {str(e)}"


def load_or_build_embeddings(doc_path: str, paragraphs: List[str], cfg: Config, use_cache: bool) -> Tuple[np.ndarray, List[str]]:
    text = "\n\n".join(paragraphs)
    doc_hash = sha256_of(text)
    cache_path = cache_path_for(doc_path, cfg.embedding_model, doc_hash)

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if (
            data.get("model") == cfg.embedding_model
            and data.get("sha256") == doc_hash
            and len(data.get("paragraphs", [])) == len(paragraphs)
        ):
            vecs = np.array(data["embeddings"], dtype=np.float32)
            paras = data["paragraphs"]
            return vecs, paras

    # Recalcular
    para_vecs = embed_texts(paragraphs, cfg, is_query=False)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": cfg.embedding_model,
                "sha256": doc_hash,
                "paragraphs": paragraphs,
                "embeddings": para_vecs.tolist(),
            },
            f,
            ensure_ascii=False,
        )
    return para_vecs, paragraphs


def main():
    parser = argparse.ArgumentParser(description="RAG con embeddings locales + OpenRouter API")
    parser.add_argument("--doc", required=True, help="Ruta al archivo de texto")
    parser.add_argument("--query", required=True, help="Consulta del usuario")
    parser.add_argument("--no-cache", action="store_true", help="Desactivar caché de embeddings")
    parser.add_argument("--show-paragraphs", action="store_true", help="Imprime los párrafos top-3 con similitudes")
    parser.add_argument("--top-k", type=int, default=3, help="Número de párrafos más similares a usar (default: 3)")
    args = parser.parse_args()

    cfg = load_config()

    doc_text = read_text_file(args.doc)
    paragraphs = split_paragraphs(doc_text)
    if not paragraphs:
        raise RuntimeError("El documento no contiene párrafos.")

    para_vecs, paragraphs = load_or_build_embeddings(
        doc_path=args.doc,
        paragraphs=paragraphs,
        cfg=cfg,
        use_cache=not args.no_cache,
    )

    query_vec = embed_texts([args.query], cfg, is_query=True)[0]
    top_similar = most_similar_paragraphs(query_vec, para_vecs, top_k=args.top_k)
    
    context_paragraphs = [(paragraphs[idx], similarity) for idx, similarity in top_similar]

    if args.show_paragraphs:
        print("=== TOP PARRAFOS RECUPERADOS ===")
        for i, (paragraph, similarity) in enumerate(context_paragraphs, 1):
            print(f"--- Parrafo {i} (similitud: {similarity:.4f}) ---")
            print(paragraph)
            print("=" * 50)
        print()

    # Generate response using OpenRouter API
    print("Generando respuesta con OpenRouter API...")
    response = call_openrouter_api(args.query, context_paragraphs, cfg)
    
    print("\n=== RESPUESTA ===")
    print(response)
    
    # Show similarity scores summary
    print(f"\n=== INFORMACIÓN DE SIMILITUD ===")
    for i, (_, similarity) in enumerate(context_paragraphs, 1):
        print(f"Párrafo {i}: {similarity:.4f}")
    
    if context_paragraphs:
        highest_sim = context_paragraphs[0][1]
        print(f"Similitud más alta: {highest_sim:.4f} (párrafo 1 - prioridad máxima)")


if __name__ == "__main__":
    main()