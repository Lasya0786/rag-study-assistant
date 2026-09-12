import os

from app.pdf_processor import PDFProcessor
from app.flashcard_generator import FlashcardGenerator


PDF_FOLDER = "data/pdfs"


pdf_files = [
    file
    for file in os.listdir(PDF_FOLDER)
    if file.lower().endswith(".pdf")
]


if not pdf_files:

    print("No PDF files found.")
    exit()


print("\nAvailable PDFs:")
print("=" * 60)

for i, filename in enumerate(pdf_files, start=1):

    print(f"{i}. {filename}")


choice = int(
    input("\nChoose a PDF number: ")
)

selected_file = pdf_files[choice - 1]

pdf_path = os.path.join(
    PDF_FOLDER,
    selected_file
)


number = int(
    input("How many flashcards do you want? ")
)


print("\nProcessing:", selected_file)


processor = PDFProcessor(pdf_path)

pages = processor.extract_pages()

full_text = "\n\n".join(
    page["text"]
    for page in pages
)


print("Pages extracted:", len(pages))


flashcard_generator = FlashcardGenerator()

flashcards = flashcard_generator.generate_flashcards(
    full_text,
    number=number
)


print("\n")
print("=" * 60)
print("GENERATED FLASHCARDS")
print("=" * 60)

print(flashcards)


print("\n")
print("=" * 60)
print("SOURCE")
print("=" * 60)

print(selected_file)