from app.rag_pipeline import RAGPipeline


rag = RAGPipeline()

question = input(
    "\nAsk a question about your PDFs: "
)

result = rag.ask(
    question,
    n_results=10,
    top_k=3
)

print("\n")
print("=" * 60)
print("AI ANSWER")
print("=" * 60)

print(result["answer"])


print("\n")
print("=" * 60)
print("SOURCES")
print("=" * 60)

for source in result["sources"]:

    print(
        f"\n📄 File: {source['filename']}"
    )

    print(
        f"📖 Page: {source['page_number']}"
    )

    print("\nRelevant passage:")
    print("-" * 60)

    print(source["text"])

    print("-" * 60)