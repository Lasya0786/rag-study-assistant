from app.retriever import Retriever
from app.gemini import GeminiModel
from app.reranker import Reranker
from app.chat_history import ChatHistory


class RAGPipeline:

    def __init__(self):

        # ====================================================
        # INITIALIZE COMPONENTS
        # ====================================================

        self.retriever = Retriever()

        self.reranker = Reranker()

        self.gemini = GeminiModel()

        self.chat_history = ChatHistory()


    def ask(
        self,
        question,
        filename=None,
        n_results=10,
        top_k=3
    ):

        # ====================================================
        # GET CHAT HISTORY
        # ====================================================

        history = (
            self.chat_history
            .get_history()
        )


        # ====================================================
        # RETRIEVE CANDIDATES
        # ====================================================

        results = self.retriever.retrieve(
            question,
            n_results=n_results,
            filename=filename
        )


        # ====================================================
        # EXTRACT DOCUMENTS
        # ====================================================

        documents = (
            results["documents"][0]
        )

        metadatas = (
            results["metadatas"][0]
        )


        # ====================================================
        # NO RESULTS
        # ====================================================

        if not documents:

            return {

                "answer": (
                    "I could not find relevant "
                    "information in the selected "
                    "study material."
                ),

                "sources": []

            }


        # ====================================================
        # RERANK RETRIEVED DOCUMENTS
        # ====================================================

        ranked = self.reranker.rerank(
            question,
            documents,
            top_k=top_k
        )


        # ====================================================
        # SELECT BEST DOCUMENTS
        # ====================================================

        selected_documents = []

        selected_metadatas = []

        selected_scores = []


        for index, score in ranked:

            selected_documents.append(
                documents[index]
            )

            selected_metadatas.append(
                metadatas[index]
            )

            selected_scores.append(
                float(score)
            )


        # ====================================================
        # BUILD CONTEXT
        # ====================================================

        context_parts = []


        for document, metadata in zip(
            selected_documents,
            selected_metadatas
        ):

            context_parts.append(

                f"[Source: "
                f"{metadata['filename']}, "
                f"Page "
                f"{metadata['page_number']}]\n"
                f"{document}"

            )


        context = "\n\n".join(
            context_parts
        )


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        answer = self.gemini.generate_answer(
            question,
            context,
            history
        )


        # ====================================================
        # SAVE CHAT HISTORY
        # ====================================================

        self.chat_history.add_message(
            "user",
            question
        )


        self.chat_history.add_message(
            "assistant",
            answer
        )


        # ====================================================
        # CREATE SOURCE INFORMATION
        # ====================================================

        sources = []


        for document, metadata, score in zip(
            selected_documents,
            selected_metadatas,
            selected_scores
        ):

            sources.append({

                "filename":
                    metadata["filename"],

                "page_number":
                    metadata["page_number"],

                "text":
                    document,

                "reranker_score":
                    score

            })


        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {

            "answer":
                answer,

            "sources":
                sources

        }