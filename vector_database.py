import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)


class QdrantStorage:
    def __init__(
            self,
            url: str = "http://localhost:6333",
            collection: str = "docs",
            dim: int = 768,
            timeout: int = 60
    ):
        try:
            self.client = QdrantClient(url=url, timeout=timeout)
            self.collection = collection
            self.dim = dim

            # Test connection
            self.client.get_collections()
            logger.info(f"Connected to Qdrant at {url}")

            # Create collection if it doesn't exist
            if not self.client.collection_exists(self.collection):
                logger.info(f"Creating collection '{self.collection}' with dimension {dim}")
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            else:
                logger.info(f"Using existing collection '{self.collection}'")

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise ConnectionError(f"Cannot connect to Qdrant at {url}: {str(e)}")

    def upsert(self, vectors: list[list[float]], ids: list[str], payloads: list[dict]) -> None:
        try:
            if not vectors or not ids or not payloads:
                raise ValueError("vectors, ids, and payloads cannot be empty")

            if len(vectors) != len(ids) or len(vectors) != len(payloads):
                raise ValueError(
                    f"Length mismatch: vectors({len(vectors)}), ids({len(ids)}), payloads({len(payloads)})"
                )

            # Validate vector dimensions
            for i, vec in enumerate(vectors):
                if len(vec) != self.dim:
                    raise ValueError(
                        f"Vector {i} has dimension {len(vec)}, expected {self.dim}"
                    )

            points = [
                PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
                for i in range(len(ids))
            ]

            logger.info(f"Upserting {len(points)} points to collection '{self.collection}'")
            self.client.upsert(collection_name=self.collection, points=points)
            logger.info(f"Successfully upserted {len(points)} points")

        except UnexpectedResponse as e:
            logger.error(f"Qdrant error during upsert: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            raise

    def search(self, query_vector: list[float], top_k: int = 5) -> dict:
        try:
            if not query_vector:
                raise ValueError("query_vector cannot be empty")

            if len(query_vector) != self.dim:
                raise ValueError(
                    f"Query vector has dimension {len(query_vector)}, expected {self.dim}"
                )

            if top_k < 1:
                raise ValueError("top_k must be at least 1")

            logger.info(f"Searching collection '{self.collection}' for top {top_k} results")
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )

            contexts = []
            sources = set()

            for r in results:
                payload = getattr(r, "payload", None) or {}
                text = payload.get("text", "")
                source = payload.get("source", "")

                if text:
                    contexts.append(text)
                    if source:
                        sources.add(source)

            logger.info(f"Found {len(contexts)} contexts from {len(sources)} sources")
            return {"contexts": contexts, "sources": list(sources)}

        except UnexpectedResponse as e:
            logger.error(f"Qdrant error during search: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def delete_collection(self) -> None:
        try:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(collection_name=self.collection)
                logger.info(f"Deleted collection '{self.collection}'")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    def get_collection_info(self) -> dict:
        try:
            info = self.client.get_collection(collection_name=self.collection)
            return {
                "name": self.collection,
                "vectors_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise
