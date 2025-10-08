import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import os
import uuid
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_database import QdrantStorage
from custom_types import RAGQueryResult,RAGSearchResult, RAGUpsertResult,RAGChunkAndSrc
load_dotenv()

inngest_client=inngest.Inngest(
    app_id="ragent_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()

)
@inngest_client.create_function(
    fn_id="RAGent: Inngest PDF",
    trigger=inngest.TriggerEvent(event="ragent/inngest_pdf")
)
async def ragent_inngest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context)->RAGChunkAndSrc:
        pdf_path=ctx.event.data["pdf_path"]
        source_id=ctx.event.data.get("source_id",pdf_path)
        chunks=load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc)->RAGUpsertResult:
        chunks=chunks_and_src.chunks
        source_id=chunks_and_src.source_id
        vecs=embed_texts(chunks)
        ids=[str(uuid.uuid5(uuid.NAMESPACE_URL,name=f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads=[{"source":source_id, "text": chunks[i]} for i in range(len(chunks))]
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src= await ctx.step.run("load_and_chunk",lambda: _load(ctx),output_type=RAGChunkAndSrc)
    ingested=await ctx.step.run("embed-and-upsert", lambda: _load(ctx),output_type=RAGChunkAndSrc)
    return ingested.model_dump()
app=FastAPI()
inngest.fast_api.serve(app, inngest_client, functions=[ragent_inngest_pdf])