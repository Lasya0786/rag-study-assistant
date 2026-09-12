from app.gemini import GeminiModel


gemini = GeminiModel()


question = "What is a derivative?"


context = """
A derivative represents the rate at which
a function changes with respect to its variable.
"""


answer = gemini.generate_answer(
    question,
    context
)


print("\nGemini Answer:")
print("----------------")
print(answer)