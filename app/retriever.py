from app.embedding_model import EmbeddingModel
from app.vector_store import VectorStore


class Retriever:

    def __init__(self):

        print("Initializing Retriever...")

        self.embedding_model = (
            EmbeddingModel()
        )

        self.vector_store = (
            VectorStore()
        )

        print("Retriever initialized.")


    def retrieve(
        self,
        query,
        n_results=10,
        filename=None
    ):

        # ====================================================
        # GENERATE QUERY EMBEDDING
        # ====================================================

        query_embedding = (
            self.embedding_model
            .generate_embeddings(
                [query]
            )[0]
        )


        # ====================================================
        # SEARCH CHROMADB
        # ====================================================

        results = (
            self.vector_store.search(
                query_embedding,
                n_results=n_results,
                filename=filename
            )
        )


        return results