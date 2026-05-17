# LLM judge

from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def evaluate(agent_output):
    prompt = [
        {"role": "system", "content": "You are a strict evaluator of AI reasoning."},
        {"role": "user", "content": f"Evaluate this:\n{agent_output}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt
    )

    return response.choices[0].message.content