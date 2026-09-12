from app.gemini import GeminiModel


class Summarizer:

    def __init__(self):
        self.gemini = GeminiModel()

    def summarize(self, text):

        prompt = f"""
You are an AI study assistant.

Create a clear and concise study summary from
the provided study material.

Include:
- Main concepts
- Important definitions
- Key points
- Important formulas
- Important facts

Use ONLY the provided material.
Do not invent information.

Study material:
----------------
{text}
----------------

Create a study-friendly summary.
"""

        interaction = (
            self.gemini.client
            .interactions.create(
                model=self.gemini.model,
                input=prompt
            )
        )

        return interaction.output_text