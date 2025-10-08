import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import os
import uuid
import datetime

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
    return {"hello": "world"}
app=FastAPI()
inngest.fast_api.serve(app, inngest_client, functions=[ragent_inngest_pdf])