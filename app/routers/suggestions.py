from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
from app.database import get_db
from app.utils.auth import auth_dependency
from app.ingestion.utils import Ingestion
from app.suggestion.suggestion import EnhancedTopicSuggestionSystem

router = APIRouter(tags=["suggestions"])

class TopicSuggestion(BaseModel):
    title: str
    userPrompt: str

class ChatSuggestionsResponse(BaseModel):
    suggested_topics: List[TopicSuggestion]

@router.get('/suggestions/chat', response_model=ChatSuggestionsResponse)
async def get_chat_suggestions(
    db: Session = Depends(get_db),
    user: dict = Depends(auth_dependency)
):
    try:
        topic_system = EnhancedTopicSuggestionSystem()
        user_id = user.id

        # suggested_topics = await topic_system.generate_topic_suggestions(user_id)
        suggested_topics = topic_system.generate_static_suggestions()

        if len(suggested_topics) == 0:
            return []

        response = ChatSuggestionsResponse(
            suggested_topics=[TopicSuggestion(**topic) for topic in suggested_topics]
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
