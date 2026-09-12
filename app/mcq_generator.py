from app.gemini import GeminiModel


class MCQGenerator:

    def __init__(self):
        self.gemini = GeminiModel()

    def generate_mcqs(self, text, number=5):

        prompt = f"""
You are an AI study assistant.

Generate exactly {number} multiple-choice questions
from the study material below.

Return ONLY valid JSON.

Use this exact structure:

[
  {{
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Option A",
    "explanation": "Short explanation"
  }}
]

Rules:
- Exactly 4 options per question.
- Only one option is correct.
- The answer must exactly match one of the options.
- Include a short explanation.
- Use ONLY the provided study material.
- Do not invent information.
- Questions should test understanding.
- Generate exactly {number} questions.

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