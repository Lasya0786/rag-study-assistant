from app.embedding_model import EmbeddingModel
from app.vector_store import VectorStore


# ============================================================
# TEST DATA
# ============================================================

chunks = [

    {
        "text": "Dynamic programming solves problems by breaking them into overlapping subproblems.",
        "page_number": 1
    },

    {
        "text": "An operating system manages processes, memory, files, and devices.",
        "page_number": 2
    },

    {
        "text": "A mathematical statement is a sentence that is either true or false.",
        "page_number": 3
    }

]


# ============================================================
# INITIALIZE MODELS
# ============================================================

embedding_model = EmbeddingModel()

vector_store = VectorStore()


# ============================================================
# CREATE TEXT LIST
# ============================================================

texts = [
    chunk["text"]
    for chunk in chunks
]


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

embeddings = (
    embedding_model
    .generate_embeddings(texts)
)


print(
    "\nNumber of embeddings:",
    len(embeddings)
)


print(
    "Embedding dimension:",
    len(embeddings[0])
)


# ============================================================
# ADD DOCUMENTS TO CHROMADB
# ============================================================

filename = "test_document.pdf"


added = vector_store.add_documents(
    chunks,
    embeddings,
    filename
)


print(
    f"\nNew chunks added: {added}"
)


# ============================================================
# TEST SEARCH
# ============================================================

query = "What does an operating system do?"


query_embedding = (
    embedding_model
    .generate_embeddings(
        [query]
    )[0]
)


results = vector_store.search(
    query_embedding,
    n_results=3
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("CHROMADB SEARCH RESULTS")
print("=" * 60)


for index, document in enumerate(
    results["documents"][0],
    start=1
):

    metadata = (
        results["metadatas"][0][index - 1]
    )


    print(
        f"\nResult {index}"
    )


    print(
        "Filename:",
        metadata["filename"]
    )


    print(
        "Page:",
        metadata["page_number"]
    )


    print(
        "Text:",
        document
    )


print("\n")
print("=" * 60)
print("CHROMADB TEST COMPLETED")
print("=" * 60)