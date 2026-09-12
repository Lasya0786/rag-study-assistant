from app.gemini import GeminiModel


class FlashcardGenerator:

    def __init__(self):
        self.gemini = GeminiModel()

    def generate_flashcards(
        self,
        text,
        number=5
    ):

        prompt = f"""
You are an AI study assistant.

Create exactly {number} study flashcards
from the provided study material.

Return ONLY valid JSON.

Use this exact structure:

[
  {{
    "question": "Question text",
    "answer": "Short answer"
  }}
]

Rules:
- Generate exactly {number} flashcards.
- Use ONLY the provided study material.
- Do not invent information.
- Focus on important concepts, definitions,
  formulas, and key facts.
- Keep answers short and easy to revise.
- Make every question clear.

Study material:
----------------
{text}
----------------
"""

        interaction = (
            self.gemini.client
            .interactions.create(
                model=self.gemini.model,
                input=prompt
            )
        )

        return interaction.output_text