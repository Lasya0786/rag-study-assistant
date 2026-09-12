import pymupdf
import re


class PDFProcessor:

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path


    # ========================================================
    # EXTRACT PAGES
    # ========================================================

    def extract_pages(self):

        document = pymupdf.open(
            self.pdf_path
        )

        pages = []


        for page_number, page in enumerate(
            document,
            start=1
        ):

            text = page.get_text()

            text = text.strip()


            if text:

                pages.append({

                    "text": text,

                    "page_number":
                        page_number

                })


        document.close()


        return pages


    # ========================================================
    # CLEAN TEXT
    # ========================================================

    def clean_text(self, text):

        # Replace multiple spaces
        text = re.sub(
            r"[ \t]+",
            " ",
            text
        )


        # Replace excessive newlines
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )


        return text.strip()


    # ========================================================
    # CREATE BETTER CHUNKS
    # ========================================================

    def create_chunks(
        self,
        pages,
        chunk_size=900,
        overlap=150
    ):

        chunks = []


        for page in pages:

            text = self.clean_text(
                page["text"]
            )

            page_number = (
                page["page_number"]
            )


            # Split text into paragraphs

            paragraphs = re.split(
                r"\n\s*\n",
                text
            )


            current_chunk = ""


            for paragraph in paragraphs:

                paragraph = paragraph.strip()


                if not paragraph:
                    continue


                # If adding the paragraph keeps
                # the chunk within the target size

                if (
                    len(current_chunk)
                    + len(paragraph)
                    + 1
                    <= chunk_size
                ):

                    if current_chunk:

                        current_chunk += "\n\n"

                    current_chunk += paragraph


                else:

                    # Save current chunk

                    if current_chunk:

                        chunks.append({

                            "text":
                                current_chunk,

                            "page_number":
                                page_number

                        })


                    # Create overlap

                    overlap_text = (
                        current_chunk[-overlap:]
                        if current_chunk
                        else ""
                    )


                    current_chunk = (
                        overlap_text
                        + "\n\n"
                        + paragraph
                    )


                    # If a single paragraph is
                    # larger than chunk size,
                    # split it safely

                    while (
                        len(current_chunk)
                        > chunk_size
                    ):

                        chunk_text = (
                            current_chunk[
                                :chunk_size
                            ]
                        )


                        chunks.append({

                            "text":
                                chunk_text.strip(),

                            "page_number":
                                page_number

                        })


                        current_chunk = (
                            current_chunk[
                                chunk_size - overlap:
                            ]
                        )


            # Save remaining chunk

            if current_chunk.strip():

                chunks.append({

                    "text":
                        current_chunk.strip(),

                    "page_number":
                        page_number

                })


        return chunks