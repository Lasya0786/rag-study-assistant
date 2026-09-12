import os

from app.pdf_processor import PDFProcessor
from app.embedding_model import EmbeddingModel
from app.vector_store import VectorStore


PDF_FOLDER = "data/pdfs"


embedding_model = EmbeddingModel()
vector_store = VectorStore()


for filename in os.listdir(PDF_FOLDER):

    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(
        PDF_FOLDER,
        filename
    )

    print("\n" + "=" * 60)
    print(f"Processing: {filename}")
    print("=" * 60)

    # 1. Extract PDF pages
    processor = PDFProcessor(pdf_path)

    pages = processor.extract_pages()

    # 2. Create chunks
    chunks = processor.create_chunks(pages)

    print("Pages:", len(pages))
    print("Chunks:", len(chunks))

    # 3. Generate embeddings
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.generate_embeddings(
        texts
    )

    print(
        "Embeddings:",
        len(embeddings)
    )

    # 4. Store in ChromaDB
    vector_store.add_documents(
        chunks,
        embeddings,
        filename
    )

    print(
        f"{filename} successfully added!"
    )


print("\n")
print("=" * 60)
print("ALL PDFs INGESTED SUCCESSFULLY")
print("=" * 60)