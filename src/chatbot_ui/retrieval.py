import openai
import instructor

from pydantic import BaseModel
from openai import OpenAI

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery

from langsmith import traceable, get_current_run_tree
from chatbot_ui.core.config import config

from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


@traceable(
        name="embed_query",
        run_type="embedding",
        metadata={"ls_provider": config.EMBEDDING_MODEL_PROVIDER, "ls_model_name": config.EMBEDDING_MODEL}
)
def get_embedding(text, 
                  model=f"{config.EMBEDDING_MODEL}"):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens
        }

    return response.data[0].embedding

@traceable(
        name="retrieve_top_n",
        run_type="retriever"
)
def retrieve_context(query, qdrant_client, top_k):

    query_embedding = get_embedding(query)

    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01-hybrid", # config.QDRANT_COLLECTION_NAME   
        prefetch=[
            Prefetch(
                query=query_embedding,
                limit=20
            ),
            Prefetch(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="text",
                            match=MatchText(text=query)
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k
    )

    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.id)
        retrieved_context.append(result.payload["text"])
        similarity_scores.append(result.score)

    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores
    }

@traceable(
        name="format_retrieved_context",
        run_type="prompt"
)
def process_context(context):
    formatted_context = ""

    for chunk in context["retrieved_context"]:
        formatted_context += f"- {chunk}\n" 

    return formatted_context

@traceable(
        name="render_prompt",
        run_type="prompt"
)
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


class RAGGenerationResponse(BaseModel): 
    answer: str


@traceable(
        name="generate_llm",
        run_type="llm",
        metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL}
)
def generate_answer(prompt, temperature):

    client = instructor.from_openai(OpenAI(api_key=openai.api_key))

    response, raw_response = client.chat.completions.create_with_completion(
        model="gpt-4.1",
        response_model=RAGGenerationResponse,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
    )

    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    return response

@traceable(
        name="rag_pipeline"
)
def rag_pipeline(question, qdrant_client, top_k, temperature):
    retrieved_context = retrieve_context(question, qdrant_client, top_k)
    prompt = build_prompt(retrieved_context, question)
    answer = generate_answer(prompt, temperature)

    final_result = {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
        "similarity_scores": retrieved_context["similarity_scores"]

    }
    return final_result

