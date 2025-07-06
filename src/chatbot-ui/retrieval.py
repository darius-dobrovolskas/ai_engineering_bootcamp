import openai
from core.config import config
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPEN_AI_KEY")


def get_embedding(text, 
                  model=f"{config.EMBEDDING_MODEL}"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding


def retrieve_context(query, qdrant_client, top_k):
    query_embedding = get_embedding(query)
    print("Embedding length:", len(query_embedding)) 
    results = qdrant_client.query_points(
        collection_name=f"{config.QDRANT_COLLECTION_NAME}",
        query=query_embedding,
        limit=top_k
    )
    return results


def process_context(context):
    formatted_context = ""

    for chunk in context:
        formatted_context += f"- {chunk}\n" 

    return formatted_context


def build_prompt(context, question):

    processed_context = process_context(context)

    prompt = f"""
    You are an AI shopping assistant that can answer questions about the products in stock.

    You will be given a question and a list of context.

    Instructions:
    - You need to answer the question based on the provided context only.

    Context:
    {processed_context}

    Question:
    {question}
    """

    return prompt


def generate_answer(prompt, temperature):
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content


def rag_pipeline(question, qdrant_client, top_k, temperature):
    retrieved_context = retrieve_context(question, qdrant_client, top_k)
    prompt = build_prompt(retrieved_context, question)
    answer = generate_answer(prompt, temperature)

    return answer

