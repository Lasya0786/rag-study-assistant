from app.retriever import Retriever


retriever = Retriever()


query = "What does the document say about advertisements?"


results = retriever.retrieve(
    query,
    n_results=3
)


print("\n")
print("=" * 60)
print("SEMANTIC RETRIEVAL RESULTS")
print("=" * 60)


documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]


for i in range(len(documents)):

    print(f"\nResult {i + 1}")
    print("-" * 60)

    print("Page:", metadatas[i]["page_number"])

    print("Distance:", distances[i])

    print("\nText:")
    print(documents[i])


print("\n")
print("=" * 60)
print("RETRIEVAL COMPLETED")
print("=" * 60)