from app.pdf_processor import PDFProcessor


pdf_path = "data/pdfs/jemh1a1.pdf"


processor = PDFProcessor(pdf_path)


# Extract pages
pages = processor.extract_pages()

print("\nPDF Processing Results")
print("----------------------")

print("Number of pages with text:", len(pages))


# Create chunks
chunks = processor.create_chunks(pages)

print("Number of chunks:", len(chunks))


# Display first chunk
if chunks:
    print("\nFirst chunk:")
    print(chunks[0]["text"])

    print("\nPage number:")
    print(chunks[0]["page_number"])