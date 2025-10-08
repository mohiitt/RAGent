import os
import uuid
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import google.generativeai as genai

# Custom modules
from data_loader import load_and_chunk_pdf, embed_texts
from vector_database import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

# Load environment variables
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash"

# Setup logging
logger = logging.getLogger("uvicorn")

# Initialize Inngest
inngest_client = inngest.Inngest(
    app_id="ragent_app",
    logger=logger,
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


# Helper functions for steps
def load_pdf_chunks(pdf_path: str, source_id: Optional[str] = None) -> RAGChunkAndSrc:
    """Load and chunk a PDF file."""
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not source_id:
            source_id = pdf_path

        logger.info(f"Loading PDF: {pdf_path}")
        chunks = load_and_chunk_pdf(pdf_path)

        if not chunks:
            raise ValueError(f"No chunks extracted from PDF: {pdf_path}")

        logger.info(f"Extracted {len(chunks)} chunks from {pdf_path}")
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)
    except Exception as e:
        logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
        raise


def upsert_embeddings(chunks: list[str], source_id: str) -> RAGUpsertResult:
    """Embed chunks and upsert into vector database."""
    try:
        if not chunks:
            raise ValueError("No chunks provided for embedding")

        logger.info(f"Embedding {len(chunks)} chunks")
        vecs = embed_texts(chunks)

        if not vecs or len(vecs) != len(chunks):
            raise ValueError(f"Embedding failed: expected {len(chunks)} vectors, got {len(vecs)}")

        # Generate deterministic IDs
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]

        logger.info(f"Upserting {len(vecs)} vectors to Qdrant")
        store = QdrantStorage()
        store.upsert(vecs, ids, payloads)

        logger.info(f"Successfully ingested {len(chunks)} chunks")
        return RAGUpsertResult(ingested=len(chunks))
    except Exception as e:
        logger.error(f"Error during embedding/upsert: {str(e)}")
        raise


def search_contexts(question: str, top_k: int = 5) -> RAGSearchResult:
    """Search for relevant contexts using vector similarity."""
    try:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        logger.info(f"Searching for: {question[:100]}...")
        query_vecs = embed_texts([question])

        if not query_vecs or len(query_vecs) == 0:
            raise ValueError("Failed to embed question")

        store = QdrantStorage()
        found = store.search(query_vecs[0], top_k)

        if not found.get("contexts"):
            logger.warning("No contexts found for query")
            return RAGSearchResult(contexts=[], sources=[])

        logger.info(f"Found {len(found['contexts'])} relevant contexts")
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise


async def generate_answer_with_gemini(question: str, contexts: list[str]) -> str:
    """Generate answer using Gemini with provided contexts."""
    try:
        if not contexts:
            return "I couldn't find any relevant information to answer your question."

        context_block = "\n\n".join(f"- {c}" for c in contexts)
        user_content = (
            "You are a helpful assistant. Use ONLY the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely and factually. If the context doesn't contain enough information, say so."
        )

        logger.info("Generating answer with Gemini")
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Use async method
        response = await model.generate_content_async(user_content)

        # Handle response safely
        if not response or not response.candidates:
            raise ValueError("Empty response from Gemini")

        answer = response.text.strip()

        if not answer:
            raise ValueError("Generated answer is empty")

        logger.info("Successfully generated answer")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer with Gemini: {str(e)}")
        raise


# PDF ingestion function
@inngest_client.create_function(
    fn_id="RAGent: Inngest PDF",
    trigger=inngest.TriggerEvent(event="ragent/inngest_pdf")
)
async def ragent_inngest_pdf(ctx: inngest.Context):
    """Ingest a PDF into the RAG system."""

    # Validate input
    pdf_path = ctx.event.data.get("pdf_path")
    if not pdf_path:
        raise ValueError("pdf_path is required in event data")

    source_id = ctx.event.data.get("source_id", pdf_path)

    chunks_and_src: RAGChunkAndSrc = await ctx.step.run(
        "load-and-chunk",
        lambda: load_pdf_chunks(pdf_path, source_id),
        output_type=RAGChunkAndSrc
    )

    ingested: RAGUpsertResult = await ctx.step.run(
        "embed-and-upsert",
        lambda: upsert_embeddings(chunks_and_src.chunks, chunks_and_src.source_id),
        output_type=RAGUpsertResult
    )

    return {"ingested": ingested.ingested}


# Query PDF using Gemini
@inngest_client.create_function(
    fn_id="RAGent: Query PDF",
    trigger=inngest.TriggerEvent(event="ragent/query_pdf_ai")
)
async def ragent_query_pdf_ai(ctx: inngest.Context):
    """Query the RAG system and generate an answer using Gemini."""

    # Validate input
    question = ctx.event.data.get("question")
    if not question or not question.strip():
        raise ValueError("question is required and cannot be empty")

    top_k = int(ctx.event.data.get("top_k", 5))
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")

    found: RAGSearchResult = await ctx.step.run(
        "embed-and-search",
        lambda: search_contexts(question, top_k),
        output_type=RAGSearchResult
    )

    answer: str = await ctx.step.run(
        "generate-answer",
        lambda: generate_answer_with_gemini(question, found.contexts)
    )

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }
app = FastAPI(title="RAGent API")
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ragent"}

inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[ragent_inngest_pdf, ragent_query_pdf_ai]
)