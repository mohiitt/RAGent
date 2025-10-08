import os
import logging
from pathlib import Path
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai
from llama_index.core.node_parser import SentenceSplitter

try:
    from llama_index.readers.file import PDFReader
except ImportError:
    try:
        from llama_index.readers.pdf import PDFReader
    except ImportError:
        raise ImportError("Could not import PDFReader. Install with: pip install llama-index-readers-file")

logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=api_key)

EMBED_MODEL = "models/text-embedding-004"  # Updated model
EMBED_DIM = 768

reader = PDFReader()
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str) -> List[str]:
    try:
        pdf_path = Path(path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File must be a PDF: {path}")

        logger.info(f"Loading PDF: {path}")
        docs = reader.load_data(file=pdf_path)

        if not docs:
            raise ValueError(f"No content extracted from PDF: {path}")

        # Extract text from documents
        texts = []
        for d in docs:
            text = getattr(d, "text", None)
            if text and text.strip():
                texts.append(text)

        if not texts:
            raise ValueError(f"No text content found in PDF: {path}")

        logger.info(f"Extracted text from {len(docs)} pages")

        # Split into chunks
        chunks = []
        for t in texts:
            chunks.extend(splitter.split_text(t))

        if not chunks:
            raise ValueError(f"No chunks generated from PDF: {path}")

        logger.info(f"Generated {len(chunks)} chunks from PDF")
        return chunks

    except Exception as e:
        logger.error(f"Error loading PDF {path}: {str(e)}")
        raise


def embed_texts(texts: List[str]) -> List[List[float]]:
    try:
        if not texts:
            raise ValueError("texts list cannot be empty")

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("No valid non-empty texts to embed")

        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")

        logger.info(f"Embedding {len(valid_texts)} texts")
        embeddings = []

        for i, text in enumerate(valid_texts):
            try:
                # Truncate very long texts
                if len(text) > 10000:
                    logger.warning(f"Truncating text {i} from {len(text)} to 10000 chars")
                    text = text[:10000]

                response = genai.embed_content(
                    model=EMBED_MODEL,
                    content=text,
                    task_type="retrieval_document"  # Specify task type for better embeddings
                )

                embedding = response["embedding"]

                # Validate embedding dimension
                if len(embedding) != EMBED_DIM:
                    raise ValueError(
                        f"Expected embedding dimension {EMBED_DIM}, got {len(embedding)}"
                    )

                embeddings.append(embedding)

                # Log progress for large batches
                if (i + 1) % 10 == 0:
                    logger.info(f"Embedded {i + 1}/{len(valid_texts)} texts")

            except Exception as e:
                logger.error(f"Error embedding text {i}: {str(e)}")
                raise

        logger.info(f"Successfully embedded {len(embeddings)} texts")
        return embeddings

    except Exception as e:
        logger.error(f"Error in embed_texts: {str(e)}")
        raise


def embed_query(query: str) -> List[float]:
    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info("Embedding query")
        response = genai.embed_content(
            model=EMBED_MODEL,
            content=query,
            task_type="retrieval_query"  # Use query task type for queries
        )

        embedding = response["embedding"]

        if len(embedding) != EMBED_DIM:
            raise ValueError(
                f"Expected embedding dimension {EMBED_DIM}, got {len(embedding)}"
            )

        return embedding

    except Exception as e:
        logger.error(f"Error embedding query: {str(e)}")
        raise