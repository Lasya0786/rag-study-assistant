from app.embedding_model import EmbeddingModel


embedding_model = EmbeddingModel()

texts = [
    "Dynamic programming is an algorithmic technique.",
    "Dynamic programming solves problems using smaller subproblems.",
    "A computer network connects multiple devices."
]

embeddings = embedding_model.generate_embeddings(texts)

print("\nNumber of texts:", len(texts))
print("Number of embeddings:", len(embeddings))
print("Embedding dimensions:", len(embeddings[0]))
print("First embedding:", embeddings[0][:10])