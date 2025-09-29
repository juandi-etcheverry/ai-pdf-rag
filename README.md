# RAG mínimo por párrafos (embeddings locales)

Aplicación mínima en Python que:
- **Lee un archivo de texto**, separa por párrafos (doble salto de línea)
- **Genera embeddings** de cada párrafo
- **Recupera** el párrafo más similar a la consulta
- **Responde** con un LLM usando ese párrafo como contexto

No requiere API externas. Usa `sentence-transformers` (modelo E5 multilingüe) localmente.

## Requisitos

- Python 3.10+
--

## Instalación

```bash
pip install -r requirements.txt
cp .env.example .env
# Edita .env si quieres cambiar el modelo local (opcional)
```

## Uso

```bash
python app.py --doc ruta/al/archivo.txt --query "¿Cuál es el objetivo principal?" --show-paragraph
```

Flags:
- `--no-cache`: fuerza recalcular embeddings
- `--show-paragraph`: imprime el párrafo elegido

Variables de entorno (vía `.env`):
- `EMBEDDING_MODEL`: por defecto `intfloat/multilingual-e5-small`

Notas:
- La primera ejecución descargará el modelo (~80MB). Funciona en CPU.

## Notas

- El caché se guarda en `.cache/embeddings_*.json` y se invalida si cambian el texto, el modelo o el número de párrafos.
- El documento debe ser texto plano. Para PDFs, convierte a texto previamente (p.ej., `pdftotext`).
