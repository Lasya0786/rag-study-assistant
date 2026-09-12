from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        print("Loading reranker model...")

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2"
        )

        print("Reranker loaded successfully.")

    def rerank(
        self,
        query,
        documents,
        top_k=3
    ):

        # Create query-document pairs
        pairs = [
            [query, document]
            for document in documents
        ]

        # Generate relevance scores
        scores = self.model.predict(
            pairs
        )

        # Keep the ORIGINAL document index
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True
        )

        # Return:
        # (document_index, relevance_score)
        return ranked[:top_k]