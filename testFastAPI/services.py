from openai import OpenAI

client = OpenAI()


def generate_answer(question: str):
    response = client.responses.create(
        model="gpt-5.4",
        input=question,
    )
    return {
        "answer": response.output_text
    }