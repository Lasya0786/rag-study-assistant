from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Embedding model loaded successfully.")

    def generate_embedding(self, text):
        embedding = self.model.encode(text)
        return embedding.tolist()

    def generate_embeddings(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings.tolist()