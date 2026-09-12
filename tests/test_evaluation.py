from app.retriever import Retriever


# ============================================================
# TEST QUESTIONS
# ============================================================

test_questions = [

    {
        "question": "What is an operating system?",
        "keywords": [
            "operating system",
            "system software",
            "hardware"
        ]
    },

    {
        "question": "What are the functions of an operating system?",
        "keywords": [
            "process",
            "memory",
            "file",
            "device"
        ]
    },

    {
        "question": "What is dynamic programming?",
        "keywords": [
            "dynamic programming",
            "subproblem",
            "optimal"
        ]
    },

    {
        "question": "What is a mathematical statement?",
        "keywords": [
            "statement",
            "true",
            "false"
        ]
    }

]


# ============================================================
# PRECISION@K
# ============================================================

def precision_at_k(
    retrieved_documents,
    keywords,
    k=5
):

    documents = retrieved_documents[:k]

    relevant_count = 0

    for document in documents:

        document_text = document.lower()

        if any(
            keyword.lower() in document_text
            for keyword in keywords
        ):

            relevant_count += 1

    if k == 0:
        return 0

    return relevant_count / k


# ============================================================
# RECALL@K
# ============================================================

def recall_at_k(
    retrieved_documents,
    keywords,
    k=5
):

    documents = retrieved_documents[:k]

    found_keywords = set()

    for document in documents:

        document_text = document.lower()

        for keyword in keywords:

            if keyword.lower() in document_text:

                found_keywords.add(keyword)

    if not keywords:
        return 0

    return (
        len(found_keywords)
        / len(keywords)
    )


# ============================================================
# RECIPROCAL RANK
# ============================================================

def reciprocal_rank(
    retrieved_documents,
    keywords
):

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        document_text = document.lower()

        if any(
            keyword.lower() in document_text
            for keyword in keywords
        ):

            return 1 / index

    return 0


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation():

    print("\n")
    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)


    # Initialize retriever

    retriever = Retriever()


    total_precision = 0

    total_recall = 0

    total_mrr = 0


    # ========================================================
    # TEST EACH QUESTION
    # ========================================================

    for test in test_questions:

        question = test["question"]

        keywords = test["keywords"]


        print("\n")
        print("-" * 70)

        print(
            f"Question: {question}"
        )


        # ----------------------------------------------------
        # RETRIEVE TOP 5 DOCUMENTS
        # ----------------------------------------------------

        results = retriever.retrieve(
            question,
            n_results=5
        )


        documents = (
            results["documents"][0]
        )


        # ----------------------------------------------------
        # CALCULATE PRECISION
        # ----------------------------------------------------

        precision = precision_at_k(
            documents,
            keywords,
            k=5
        )


        # ----------------------------------------------------
        # CALCULATE RECALL
        # ----------------------------------------------------

        recall = recall_at_k(
            documents,
            keywords,
            k=5
        )


        # ----------------------------------------------------
        # CALCULATE RECIPROCAL RANK
        # ----------------------------------------------------

        rr = reciprocal_rank(
            documents,
            keywords
        )


        # ----------------------------------------------------
        # ADD TO TOTALS
        # ----------------------------------------------------

        total_precision += precision

        total_recall += recall

        total_mrr += rr


        # ----------------------------------------------------
        # DISPLAY RESULTS
        # ----------------------------------------------------

        print(
            f"Precision@5 : {precision:.3f}"
        )

        print(
            f"Recall@5    : {recall:.3f}"
        )

        print(
            f"Reciprocal Rank : {rr:.3f}"
        )


    # ========================================================
    # AVERAGE RESULTS
    # ========================================================

    total_questions = len(
        test_questions
    )


    average_precision = (
        total_precision
        / total_questions
    )


    average_recall = (
        total_recall
        / total_questions
    )


    mrr = (
        total_mrr
        / total_questions
    )


    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)


    print(
        f"Precision@5 : "
        f"{average_precision:.3f}"
    )


    print(
        f"Recall@5    : "
        f"{average_recall:.3f}"
    )


    print(
        f"MRR         : "
        f"{mrr:.3f}"
    )


    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_evaluation()