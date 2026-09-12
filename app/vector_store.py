
import chromadb


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="data/chroma"
        )

        self.collection = (
            self.client.get_or_create_collection(
                name="study_documents"
            )
        )

        print(
            "ChromaDB initialized successfully."
        )

    def add_documents(
        self,
        chunks,
        embeddings,
        filename
    ):

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):

            chunk_id = (
                f"{filename}_page_"
                f"{chunk['page_number']}_"
                f"chunk_{i}"
            )

            ids.append(chunk_id)

            documents.append(
                chunk["text"]
            )

            metadatas.append({
                "filename": filename,
                "page_number": chunk["page_number"],
                "chunk_id": i
            })

        # Check existing IDs
        existing = self.collection.get(
            ids=ids
        )

        existing_ids = set(
            existing["ids"]
        )

        new_ids = []
        new_documents = []
        new_metadatas = []
        new_embeddings = []

        for i, document_id in enumerate(ids):

            if document_id not in existing_ids:

                new_ids.append(
                    document_id
                )

                new_documents.append(
                    documents[i]
                )

                new_metadatas.append(
                    metadatas[i]
                )

                new_embeddings.append(
                    embeddings[i]
                )

        # If everything already exists
        if not new_ids:

            print(
                f"{filename} is already "
                "in ChromaDB."
            )

            return 0

        # Add only new documents
        self.collection.add(
            ids=new_ids,
            documents=new_documents,
            embeddings=new_embeddings,
            metadatas=new_metadatas
        )

        print(
            f"{len(new_ids)} new chunks added "
            f"from {filename}."
        )

        return len(new_ids)

    def search(
        self,
        query_embedding,
        n_results=3,
        filename=None
    ):

        query_params = {
            "query_embeddings": [
                query_embedding
            ],
            "n_results": n_results
        }

        # Search only inside the selected PDF
        if filename:

            query_params["where"] = {
                "filename": filename
            }

        results = self.collection.query(
            **query_params
        )

        return results