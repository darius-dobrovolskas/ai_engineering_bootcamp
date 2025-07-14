import openai
import instructor
import json

from pydantic import BaseModel
from openai import OpenAI
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, Filter, FieldCondition, MatchText, FusionQuery

from langsmith import traceable, get_current_run_tree
from api.core.config import config

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

    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- {id}: {chunk}\n" 

    return formatted_context


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the question based on the provided context.",
        },
        "retrieved_context_ids": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer",
                        "description": "The index of the chunk that was used to answer the question.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of the item based on the context together with the id.",
                    },
                },
            },
        },
    },
}

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
    - Never use word context and refer to it as the available products.
    - As an output you need to provide: 

    * The answer to the question based on the provided context.
    * The list of the indexes of the chunks that were used to answer the question. Only return the ones that are used in the answer.
    * Short description of the item vased on the context.

    - The answer to the question should contain detailed information about the product and returned with detailed specification in bulletpoints.
    - The short description should have the name of the item.

    <OUTPUT JSON SCHEMA>
    {json.dumps(OUTPUT_SCHEMA, indent=2)}
    </OUTPUT JSON SCHEMA>
    
    Context:
    {processed_context}

    Question:
    {question}
    """

    return prompt


class RAGUsedContext(BaseModel):
    id: int
    description: str
    

class RAGGenerationResponse(BaseModel): 
    answer: str
    retrieved_context_ids: List[RAGUsedContext]


@traceable(
        name="generate_llm",
        run_type="llm",
        metadata={"ls_provider": config.GENERATION_MODEL_PROVIDER, "ls_model_name": config.GENERATION_MODEL}
)
def generate_answer(prompt):

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
def rag_pipeline(question, qdrant_client, top_k):
    retrieved_context = retrieve_context(question, qdrant_client, top_k)
    prompt = build_prompt(retrieved_context, question)
    answer = generate_answer(prompt)

    final_result = {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
        "similarity_scores": retrieved_context["similarity_scores"]

    }
    return final_result


def rag_pipeline_wrapper(question, top_k=5):
    
    qdrant_client = QdrantClient(url=config.QDRANT_URL)

    result = rag_pipeline(question, qdrant_client, top_k)

    image_url_list = []

    for id in result["answer"].retrieved_context_ids:
        payload = qdrant_client.retrieve(
            collection_name=config.QDRANT_COLLECTION_NAME,
            ids=[id.id],
        )[0].payload
        image_url = payload.get("first_large_image")
        price = payload.get("price")
        if image_url:
            image_url_list.append({"image_url": image_url, "price": price, "description": id.description})

    return {
        "answer": result["answer"].answer,
        "retrieved_images": image_url_list
    }

