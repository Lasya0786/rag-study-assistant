from app.pdf_processor import PDFProcessor
from app.summarizer import Summarizer


pdf_path = "data/pdfs/OS NOTES.pdf"

processor = PDFProcessor(pdf_path)

pages = processor.extract_pages()

full_text = "\n\n".join(
    page["text"]
    for page in pages
)

print("\nExtracted pages:", len(pages))

summarizer = Summarizer()

summary = summarizer.summarize(full_text)

print("\n")
print("=" * 60)
print("PDF SUMMARY")
print("=" * 60)

print(summary)