import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from dotenv import load_dotenv


# -----------------------------
# Config y utilidades
# -----------------------------


@dataclass
class Config:
    embedding_model: str


def load_config() -> Config:
    load_dotenv()
    embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    return Config(
        embedding_model=embedding_model,
    )


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_paragraphs(text: str) -> List[str]:
    # Divide por líneas en blanco; filtra párrafos vacíos
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


_LOCAL_EMBEDDER = None  # cache del modelo local


def get_local_embedder(model_id: str):
    global _LOCAL_EMBEDDER
    if _LOCAL_EMBEDDER is None:
        # Lazy import
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


def most_similar_paragraph(query_vec: np.ndarray, para_vecs: np.ndarray) -> Tuple[int, float]:
    # Calcula similitud coseno con todas las filas
    sims = []
    for i in range(para_vecs.shape[0]):
        sims.append(cosine_similarity(query_vec, para_vecs[i]))
    best_idx = int(np.argmax(sims)) if sims else -1
    best_sim = float(sims[best_idx]) if sims else 0.0
    return best_idx, best_sim


def chat_answer(_: str, __: str) -> str:
    # Placeholder sin LLM. Se deja para futura extensión local si se desea.
    return "[Sin LLM configurado]"


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
    parser = argparse.ArgumentParser(description="RAG mínimo por párrafos (embeddings locales, sin LLM)")
    parser.add_argument("--doc", required=True, help="Ruta al archivo de texto")
    parser.add_argument("--query", required=True, help="Consulta del usuario")
    parser.add_argument("--no-cache", action="store_true", help="Desactivar caché de embeddings")
    parser.add_argument("--show-paragraph", action="store_true", help="Imprime el párrafo elegido antes de la respuesta")
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
    idx, sim = most_similar_paragraph(query_vec, para_vecs)
    chosen_paragraph = paragraphs[idx]

    if args.show_paragraph:
        print("--- Párrafo recuperado (similitud {:.4f}) ---".format(sim))
        print(chosen_paragraph)
        print("---------------------------------------------\n")

    print(chat_answer(args.query, chosen_paragraph))


if __name__ == "__main__":
    main()



