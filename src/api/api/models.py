from pydantic import BaseModel, Field
from typing import List, Any, Optional, Union


class RAGRequest(BaseModel):
    query: str = Field(..., description="The query to be used in the RAG pipeline")
    thread_id: str = Field(..., description="The thread ID")


class RAGUsedImage(BaseModel):
    image_url: str = Field(..., description="The URL of the image")
    price: Optional[float] = Field(..., description="The price of the item")
    description: str = Field(..., description="The description of the item")


class RAGResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    answer: str = Field(..., description="The content of the RAG response")
    used_image_urls: List[RAGUsedImage]
    trace_id: str = Field(..., description="The trace ID")


class FeedbackRequest(BaseModel):
    feedback_score: Union[int, None] = Field(..., description="1 if feedback is positive, o if feedback is negative")
    feedback_text: str = Field(..., description="The feedback text")
    trace_id: str = Field(..., description="The trace ID")
    thread_id: str = Field(..., description="The thread ID")
    feedback_source_type: str = Field(..., description="The feedback source type. Human or API")


class FeedbackResponse(BaseModel):
    request_id: str = Field(..., description="The request ID")
    status: str = Field(..., description="The status of the feedback submission")