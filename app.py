import os
import json
import traceback

import streamlit as st

from app.pdf_processor import PDFProcessor
from app.embedding_model import EmbeddingModel
from app.vector_store import VectorStore
from app.rag_pipeline import RAGPipeline
from app.summarizer import Summarizer
from app.mcq_generator import MCQGenerator
from app.flashcard_generator import FlashcardGenerator


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# LOAD CSS
# ============================================================

def load_css():

    css_path = os.path.join(
        "css",
        "style.css"
    )

    if os.path.exists(css_path):

        with open(
            css_path,
            "r",
            encoding="utf-8"
        ) as file:

            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# ============================================================
# HTML RENDER HELPER
# ============================================================

def render_html(html):

    html = " ".join(
        line.strip()
        for line in html.splitlines()
        if line.strip()
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# SESSION STATE
# ============================================================

if "rag" not in st.session_state:

    st.session_state.rag = RAGPipeline()


if "mcqs" not in st.session_state:

    st.session_state.mcqs = None


if "flashcards" not in st.session_state:

    st.session_state.flashcards = None


if "flashcard_index" not in st.session_state:

    st.session_state.flashcard_index = 0


if "show_flashcard_answer" not in st.session_state:

    st.session_state.show_flashcard_answer = False


if "selected_pdf" not in st.session_state:

    st.session_state.selected_pdf = None


# Track PDFs already processed during
# the current Streamlit session

if "processed_uploads" not in st.session_state:

    st.session_state.processed_uploads = set()


# ============================================================
# INITIALIZE RAG
# ============================================================

rag = st.session_state.rag


# ============================================================
# HEADER
# ============================================================

render_html(
    """
    <div class="main-title">
        📚 AI Study Assistant
    </div>

    <div class="subtitle">
        Learn smarter with your study material using
        Retrieval-Augmented Generation
    </div>
    """
)


# ============================================================
# FEATURE CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    render_html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                🤖
            </div>

            <div class="feature-title">
                Ask Questions
            </div>

            <div class="feature-text">
                Ask questions directly from your PDFs
            </div>

        </div>
        """
    )


with col2:

    render_html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                📝
            </div>

            <div class="feature-title">
                Summarize
            </div>

            <div class="feature-text">
                Generate concise study summaries
            </div>

        </div>
        """
    )


with col3:

    render_html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                ❓
            </div>

            <div class="feature-title">
                MCQ Quiz
            </div>

            <div class="feature-text">
                Practice with AI-generated questions
            </div>

        </div>
        """
    )


with col4:

    render_html(
        """
        <div class="feature-card">

            <div class="feature-icon">
                🧠
            </div>

            <div class="feature-title">
                Flashcards
            </div>

            <div class="feature-text">
                Revise important concepts quickly
            </div>

        </div>
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📂 Study Materials")


    # ========================================================
    # PDF UPLOAD
    # ========================================================

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )


    if uploaded_file is not None:

        # Check whether the PDF is empty

        if uploaded_file.size == 0:

            st.error(
                "❌ The uploaded PDF is empty."
            )

        elif (
            uploaded_file.name
            not in st.session_state.processed_uploads
        ):

            pdf_folder = "data/pdfs"

            os.makedirs(
                pdf_folder,
                exist_ok=True
            )


            pdf_path = os.path.join(
                pdf_folder,
                uploaded_file.name
            )


            try:

                # ==================================================
                # SAVE PDF
                # ==================================================

                with open(
                    pdf_path,
                    "wb"
                ) as file:

                    file.write(
                        uploaded_file.getbuffer()
                    )


                # ==================================================
                # PROCESS PDF
                # ==================================================

                with st.spinner(
                    "📚 Processing PDF..."
                ):

                    processor = PDFProcessor(
                        pdf_path
                    )


                    pages = (
                        processor.extract_pages()
                    )


                    if not pages:

                        raise ValueError(
                            "No readable text was found "
                            "in this PDF."
                        )


                    chunks = (
                        processor.create_chunks(
                            pages
                        )
                    )


                    if not chunks:

                        raise ValueError(
                            "No text chunks could be "
                            "created from this PDF."
                        )


                    texts = [
                        chunk["text"]
                        for chunk in chunks
                    ]


                    embedding_model = (
                        EmbeddingModel()
                    )


                    embeddings = (
                        embedding_model
                        .generate_embeddings(
                            texts
                        )
                    )


                    vector_store = (
                        VectorStore()
                    )


                    new_chunks = (
                        vector_store.add_documents(
                            chunks,
                            embeddings,
                            uploaded_file.name
                        )
                    )


                # Mark as processed

                st.session_state.processed_uploads.add(
                    uploaded_file.name
                )


                st.success(
                    "✅ PDF processed successfully!"
                )


                st.info(
                    f"Pages: {len(pages)}\n\n"
                    f"Chunks: {len(chunks)}\n\n"
                    f"New chunks added: {new_chunks}"
                )


            except Exception:

                st.error(
                    "❌ Could not process this PDF."
                )


                st.warning(
                    "Please check the terminal "
                    "for the detailed error."
                )


                print(
                    "\nERROR WHILE PROCESSING PDF:"
                )

                traceback.print_exc()


        else:

            st.info(
                "ℹ️ This PDF is already processed "
                "in the current session."
            )


    # ========================================================
    # AVAILABLE PDFs
    # ========================================================

    st.subheader(
        "Available PDFs"
    )


    pdf_folder = "data/pdfs"


    if os.path.exists(pdf_folder):

        pdf_files = [
            file
            for file in os.listdir(
                pdf_folder
            )
            if file.lower().endswith(".pdf")
        ]

    else:

        pdf_files = []


    if pdf_files:

        for pdf in pdf_files:

            render_html(
                f"""
                <div class="pdf-item">
                    📄 {pdf}
                </div>
                """
            )

    else:

        st.info(
            "No PDFs available."
        )


# ============================================================
# PDF SELECTION
# ============================================================

if pdf_files:

    selected_pdf = st.selectbox(
        "📚 Select Study Material",
        pdf_files
    )


    st.session_state.selected_pdf = (
        selected_pdf
    )

else:

    selected_pdf = None


# ============================================================
# PDF INFORMATION
# ============================================================

if selected_pdf:

    pdf_path = os.path.join(
        "data/pdfs",
        selected_pdf
    )


    if os.path.exists(pdf_path):

        try:

            processor = PDFProcessor(
                pdf_path
            )


            pages = (
                processor.extract_pages()
            )


            render_html(
                f"""
                <div class="feature-card">

                    <div class="feature-title">
                        📄 {selected_pdf}
                    </div>

                    <div class="feature-text">
                        {len(pages)} pages available
                    </div>

                </div>
                """
            )

        except Exception:

            st.warning(
                "Could not read PDF information."
            )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "💬 Ask Questions",
        "📝 Summarize",
        "❓ MCQs",
        "🧠 Flashcards"
    ]
)


# ============================================================
# TAB 1 — ASK QUESTIONS
# ============================================================

with tab1:

    st.header(
        "💬 Ask Questions"
    )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    history = (
        rag.chat_history
        .get_history()
    )


    if history:

        for message in history:

            if message["role"] == "user":

                with st.chat_message(
                    "user"
                ):

                    st.write(
                        message["content"]
                    )


            elif message["role"] == "assistant":

                with st.chat_message(
                    "assistant"
                ):

                    st.write(
                        message["content"]
                    )


    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear Chat",
        key="clear_chat"
    ):

        rag.chat_history.clear()

        st.success(
            "Chat history cleared!"
        )

        st.rerun()


    # ========================================================
    # QUESTION INPUT
    # ========================================================

    question = st.text_input(
        "Ask a question about your study material:",
        placeholder=(
            "Example: What is dynamic programming?"
        )
    )


    # ========================================================
    # ASK BUTTON
    # ========================================================

    if st.button(
        "🔍 Ask Question",
        key="ask_question"
    ):

        if not selected_pdf:

            st.warning(
                "📂 Please select a PDF first."
            )


        elif not question.strip():

            st.warning(
                "✏️ Please enter a question."
            )


        elif len(question.strip()) < 3:

            st.warning(
                "✏️ Please enter a more specific question."
            )


        else:

            result = None


            # ==================================================
            # CALL RAG PIPELINE
            # ==================================================

            try:

                with st.spinner(
                    "🔎 Searching your study material..."
                ):

                    result = rag.ask(
                        question,
                        filename=selected_pdf
                    )


            except Exception:

                st.error(
                    "❌ Something went wrong while "
                    "generating the answer."
                )


                st.warning(
                    "Please try again. If the problem "
                    "continues, check the terminal "
                    "for the detailed error."
                )


                print(
                    "\nERROR WHILE ASKING QUESTION:"
                )

                traceback.print_exc()


            # ==================================================
            # DISPLAY RESULT
            # ==================================================

            if result:

                # ==============================================
                # DISPLAY ANSWER
                # ==============================================

                with st.chat_message(
                    "assistant"
                ):

                    render_html(
                        f"""
                        <div class="answer-card">

                            <strong>
                                🤖 AI Answer
                            </strong>

                            <br><br>

                            {result["answer"]}

                        </div>
                        """
                    )


                # ==============================================
                # DISPLAY SOURCES
                # ==============================================

                sources = result.get(
                    "sources",
                    []
                )


                if sources:

                    st.markdown(
                        "### 📚 Sources"
                    )


                    for source in sources:

                        score = source.get(
                            "reranker_score",
                            0
                        )


                        render_html(
                            f"""
                            <div class="source-card">

                                <div>
                                    📄 <strong>
                                        {source["filename"]}
                                    </strong>
                                </div>

                                <div>
                                    📖 Page
                                    <strong>
                                        {source["page_number"]}
                                    </strong>
                                </div>

                                <div>
                                    🔎 Reranker Score
                                    <strong>
                                        {score:.3f}
                                    </strong>
                                </div>

                            </div>
                            """
                        )


                        with st.expander(
                            "📖 View relevant passage"
                        ):

                            st.write(
                                source["text"]
                            )

                else:

                    st.info(
                        "No source passages were returned."
                    )


# ============================================================
# TAB 2 — SUMMARIZE
# ============================================================

with tab2:

    st.header(
        "📝 Summarize Study Material"
    )


    if not selected_pdf:

        st.info(
            "📂 Please select a PDF first."
        )


    else:

        if st.button(
            "✨ Generate Summary",
            key="generate_summary"
        ):

            try:

                with st.spinner(
                    "📝 Reading your study material "
                    "and generating summary..."
                ):

                    pdf_path = os.path.join(
                        "data/pdfs",
                        selected_pdf
                    )


                    processor = PDFProcessor(
                        pdf_path
                    )


                    pages = (
                        processor.extract_pages()
                    )


                    full_text = "\n\n".join(
                        page["text"]
                        for page in pages
                    )


                    # Limit API input

                    summary_text = (
                        full_text[:30000]
                    )


                    summarizer = Summarizer()


                    summary = (
                        summarizer.summarize(
                            summary_text
                        )
                    )


                st.markdown(
                    "### 📖 Study Summary"
                )


                render_html(
                    f"""
                    <div class="answer-card">
                        {summary}
                    </div>
                    """
                )


            except Exception:

                st.error(
                    "❌ Could not generate the summary."
                )


                st.warning(
                    "Please try again. If the problem "
                    "continues, check the terminal."
                )


                print(
                    "\nERROR WHILE GENERATING SUMMARY:"
                )

                traceback.print_exc()


# ============================================================
# TAB 3 — MCQs
# ============================================================

with tab3:

    st.header(
        "❓ Interactive MCQ Quiz"
    )


    if not selected_pdf:

        st.info(
            "📂 Please select a PDF first."
        )


    else:

        number_of_questions = st.slider(
            "Number of Questions",
            min_value=3,
            max_value=10,
            value=5
        )


        if st.button(
            "🎯 Generate Quiz",
            key="generate_quiz"
        ):

            try:

                with st.spinner(
                    "❓ Creating your practice quiz..."
                ):

                    pdf_path = os.path.join(
                        "data/pdfs",
                        selected_pdf
                    )


                    processor = PDFProcessor(
                        pdf_path
                    )


                    pages = (
                        processor.extract_pages()
                    )


                    full_text = "\n\n".join(
                        page["text"]
                        for page in pages
                    )


                    # Limit API input

                    mcq_text = (
                        full_text[:30000]
                    )


                    generator = MCQGenerator()


                    generated = (
                        generator.generate_mcqs(
                            mcq_text,
                            number_of_questions
                        )
                    )


                # ==========================================
                # CLEAN JSON
                # ==========================================

                generated = generated.strip()


                if generated.startswith(
                    "```"
                ):

                    generated = (
                        generated
                        .replace(
                            "```json",
                            ""
                        )
                        .replace(
                            "```",
                            ""
                        )
                        .strip()
                    )


                mcqs = json.loads(
                    generated
                )


                st.session_state.mcqs = (
                    mcqs
                )


                st.success(
                    "✅ Quiz generated successfully!"
                )


            except Exception:

                st.error(
                    "❌ Could not generate or parse MCQs."
                )


                st.warning(
                    "Please try again. If the problem "
                    "continues, check the terminal."
                )


                print(
                    "\nERROR WHILE GENERATING MCQs:"
                )

                traceback.print_exc()


        # ====================================================
        # DISPLAY QUIZ
        # ====================================================

        if st.session_state.mcqs:

            mcqs = (
                st.session_state.mcqs
            )


            st.markdown(
                "### 📝 Answer the Questions"
            )


            user_answers = []


            for index, mcq in enumerate(
                mcqs
            ):

                st.markdown(
                    f"#### Q{index + 1}. "
                    f"{mcq['question']}"
                )


                answer = st.radio(
                    "Choose your answer:",
                    mcq["options"],
                    key=f"mcq_{index}"
                )


                user_answers.append(
                    answer
                )


            # =================================================
            # SUBMIT QUIZ
            # =================================================

            if st.button(
                "✅ Submit Quiz",
                key="submit_quiz"
            ):

                score = 0


                for index, mcq in enumerate(
                    mcqs
                ):

                    if (
                        user_answers[index]
                        == mcq["answer"]
                    ):

                        score += 1


                total = len(mcqs)


                percentage = (
                    score / total
                ) * 100


                # =============================================
                # PERFORMANCE
                # =============================================

                if percentage >= 80:

                    performance = (
                        "Excellent work! 🎉"
                    )

                elif percentage >= 60:

                    performance = (
                        "Good job! 👍"
                    )

                elif percentage >= 40:

                    performance = (
                        "Keep practicing! 📚"
                    )

                else:

                    performance = (
                        "You need more revision. 💪"
                    )


                # =============================================
                # SCORE CARD
                # =============================================

                render_html(
                    f"""
                    <div class="score-card">

                        <div class="score-number">
                            {score}/{total}
                        </div>

                        <div class="score-text">
                            {percentage:.1f}%
                        </div>

                        <div class="score-text">
                            {performance}
                        </div>

                    </div>
                    """
                )


                # =============================================
                # ANSWER REVIEW
                # =============================================

                st.markdown(
                    "### 📋 Answer Review"
                )


                for index, mcq in enumerate(
                    mcqs
                ):

                    user_answer = (
                        user_answers[index]
                    )


                    if (
                        user_answer
                        == mcq["answer"]
                    ):

                        st.success(
                            f"Q{index + 1}: Correct"
                        )

                    else:

                        st.error(
                            f"Q{index + 1}: Incorrect"
                        )


                    st.write(
                        f"**Correct answer:** "
                        f"{mcq['answer']}"
                    )


                    st.write(
                        f"**Explanation:** "
                        f"{mcq['explanation']}"
                    )


# ============================================================
# TAB 4 — FLASHCARDS
# ============================================================

with tab4:

    st.header(
        "🧠 Interactive Flashcards"
    )


    if not selected_pdf:

        st.info(
            "📂 Please select a PDF first."
        )


    else:

        number_of_cards = st.slider(
            "Number of Flashcards",
            min_value=3,
            max_value=10,
            value=5,
            key="flashcard_number"
        )


        if st.button(
            "✨ Generate Flashcards",
            key="generate_flashcards"
        ):

            try:

                with st.spinner(
                    "🧠 Creating your revision flashcards..."
                ):

                    pdf_path = os.path.join(
                        "data/pdfs",
                        selected_pdf
                    )


                    processor = PDFProcessor(
                        pdf_path
                    )


                    pages = (
                        processor.extract_pages()
                    )


                    full_text = "\n\n".join(
                        page["text"]
                        for page in pages
                    )


                    # Limit API input

                    flashcard_text = (
                        full_text[:30000]
                    )


                    generator = (
                        FlashcardGenerator()
                    )


                    generated = (
                        generator.generate_flashcards(
                            flashcard_text,
                            number_of_cards
                        )
                    )


                # ==========================================
                # CLEAN JSON
                # ==========================================

                generated = generated.strip()


                if generated.startswith(
                    "```"
                ):

                    generated = (
                        generated
                        .replace(
                            "```json",
                            ""
                        )
                        .replace(
                            "```",
                            ""
                        )
                        .strip()
                    )


                flashcards = json.loads(
                    generated
                )


                st.session_state.flashcards = (
                    flashcards
                )


                st.session_state.flashcard_index = 0


                st.session_state.show_flashcard_answer = (
                    False
                )


                st.success(
                    "✅ Flashcards generated successfully!"
                )


            except Exception:

                st.error(
                    "❌ Could not generate or parse flashcards."
                )


                st.warning(
                    "Please try again. If the problem "
                    "continues, check the terminal."
                )


                print(
                    "\nERROR WHILE GENERATING FLASHCARDS:"
                )

                traceback.print_exc()


        # ====================================================
        # DISPLAY FLASHCARDS
        # ====================================================

        if st.session_state.flashcards:

            flashcards = (
                st.session_state.flashcards
            )


            index = (
                st.session_state.flashcard_index
            )


            card = flashcards[index]


            # =================================================
            # ANSWER HTML
            # =================================================

            answer_html = ""


            if (
                st.session_state
                .show_flashcard_answer
            ):

                answer_html = f"""
                <div class="flashcard-answer">
                    💡 {card["answer"]}
                </div>
                """


            # =================================================
            # FLASHCARD
            # =================================================

            render_html(
                f"""
                <div class="flashcard">

                    <div class="flashcard-number">
                        Card {index + 1}
                        of {len(flashcards)}
                    </div>

                    <div class="flashcard-question">
                        {card["question"]}
                    </div>

                    {answer_html}

                </div>
                """
            )


            # =================================================
            # SHOW / HIDE ANSWER
            # =================================================

            if st.button(
                (
                    "🙈 Hide Answer"
                    if st.session_state
                    .show_flashcard_answer
                    else
                    "👀 Show Answer"
                ),
                key="show_answer"
            ):

                st.session_state.show_flashcard_answer = (
                    not st.session_state
                    .show_flashcard_answer
                )

                st.rerun()


            # =================================================
            # NAVIGATION
            # =================================================

            col1, col2, col3 = st.columns(
                [1, 2, 1]
            )


            with col1:

                if st.button(
                    "⬅️ Previous",
                    key="previous_card"
                ):

                    if index > 0:

                        st.session_state.flashcard_index -= 1

                        st.session_state.show_flashcard_answer = (
                            False
                        )

                        st.rerun()


            with col2:

                st.markdown(
                    f"""
                    <div style="
                        text-align:center;
                        font-weight:700;
                        padding:10px;
                    ">
                        Card {index + 1}
                        / {len(flashcards)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col3:

                if st.button(
                    "Next ➡️",
                    key="next_card"
                ):

                    if (
                        index
                        < len(flashcards) - 1
                    ):

                        st.session_state.flashcard_index += 1

                        st.session_state.show_flashcard_answer = (
                            False
                        )

                        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)


render_html(
    """
    <div class="footer">

        📚 AI Study Assistant

        <br>

        Powered by RAG + ChromaDB + Gemini

    </div>
    """
)