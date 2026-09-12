import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


class GeminiModel:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in .env file."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = "gemini-3.6-flash"

        print("Gemini initialized successfully.")

    def generate_answer(
        self,
        question,
        context,
        history=None
    ):

        # Convert previous conversation into text
        history_text = ""

        if history:

            history_text = "\n".join(
                f"{message['role'].upper()}: "
                f"{message['content']}"
                for message in history
            )

        prompt = f"""
You are an AI study assistant.

Answer the user's question using ONLY the
provided study material.

You may use the previous conversation to
understand follow-up questions.

If the answer cannot be found in the study
material, say:

"I could not find the answer in the provided
study material."

Do not invent information.

Previous conversation:
----------------
{history_text}
----------------

Study material:
----------------
{context}
----------------

Current question:
{question}

Give a clear and concise answer suitable
for a student.
"""

        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt
        )

        return interaction.output_text