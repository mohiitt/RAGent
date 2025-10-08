import os
import uuid
import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import google.generativeai as genai

# Custom modules
from data_loader import load_and_chunk_pdf, embed_texts
from vector_database import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult, RAGQueryResult

# --- Load environment variables ---
load_dotenv()

# --- Configure Gemini ---
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-1.5-flash"  # or "gemini-1.5-pro"

# --- Initialize Inngest ---
inngest_client = inngest.Inngest(
    app_id="ragent_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

# --- PDF ingestion function ---
@inngest_client.create_function(
    fn_id="RAGent: Inngest PDF",
    trigger=inngest.TriggerEvent(event="ragent/inngest_pdf")
)
async def ragent_inngest_pdf(ctx: inngest.Context):

    # Step 1: Load PDF and chunk it
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    chunks_and_src: RAGChunkAndSrc = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc
    )

    # Step 2: Embed and upsert into vector DB
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        store = QdrantStorage()
        store.upsert(vecs, ids, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    ingested: RAGUpsertResult = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult
    )

    return ingested

# --- Query PDF using Gemini ---
@inngest_client.create_function(
    fn_id="RAGent: Query PDF",
    trigger=inngest.TriggerEvent(event="ragent/query_pdf_ai")
)
async def ragent_query_pdf_ai(ctx: inngest.Context):

    # Step 1: Search relevant chunks
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vecs = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vecs, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found: RAGSearchResult = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult
    )

    # Step 2: Construct prompt for Gemini
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "You are a helpful assistant. Use ONLY the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer concisely and factually."
    )

    # Step 3: Call Gemini LLM
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = await ctx.step.run(
        "generate-answer",
        lambda: model.generate_content(user_content),
    )

    answer = response.text.strip()

    # Step 4: Return using Pydantic model
    return RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts)
    )

# --- FastAPI app ---
app = FastAPI()
inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[ragent_inngest_pdf, ragent_query_pdf_ai]
)
