import os

from app.pdf_processor import PDFProcessor
from app.mcq_generator import MCQGenerator


PDF_FOLDER = "data/pdfs"


# Find all PDFs
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


# Select PDF
choice = int(
    input("\nChoose a PDF number: ")
)

selected_file = pdf_files[choice - 1]

pdf_path = os.path.join(
    PDF_FOLDER,
    selected_file
)


# Ask number of MCQs
number = int(
    input("How many MCQs do you want? ")
)


print("\nProcessing:", selected_file)


# Extract PDF text
processor = PDFProcessor(pdf_path)

pages = processor.extract_pages()

full_text = "\n\n".join(
    page["text"]
    for page in pages
)


print("Pages extracted:", len(pages))


# Generate MCQs
mcq_generator = MCQGenerator()

mcqs = mcq_generator.generate_mcqs(
    full_text,
    number=number
)


# Display result
print("\n")
print("=" * 60)
print("GENERATED MCQs")
print("=" * 60)

print(mcqs)

print("\n")
print("=" * 60)
print("SOURCE")
print("=" * 60)

print(selected_file)