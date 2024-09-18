import asyncio
from typing import List, Dict
from app.ingestion.utils import Ingestion
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import AsyncGroq
from app.database import get_settings
from pymongo import MongoClient

class EnhancedTopicSuggestionSystem:
    def __init__(self, collection_name="embeddings"):
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client['document_db']
        self.collection = self.db[collection_name]
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.tfidf_vectorizer = TfidfVectorizer()
        self.file_name_vectorizer = TfidfVectorizer()
        self.llm = AsyncGroq(api_key=get_settings().GROQ_API)
        self.ingestion = Ingestion(collection_name)

    async def generate_topic_suggestions(self, user_id: str, num_suggestions: int = 5) -> List[Dict]:
            # Get all tags for the user's documents
            pipeline = [
                {"$match": {"metadata.user_id": user_id}},
                {"$project": {"tags": "$metadata.tags", "file_name": "$metadata.file_name"}},
                {"$unwind": "$tags"},
                {"$group": {"_id": "$tags", "count": {"$sum": 1}, "files": {"$addToSet": "$file_name"}}}
            ]
            tag_results = list(self.collection.aggregate(pipeline))

            # Sort tags by frequency and get the top ones
            top_tags = sorted(tag_results, key=lambda x: x['count'], reverse=True)[:10]

            if len(top_tags) == 0:
                return []

            # Prepare the prompt for the LLM
            tags_summary = ", ".join([f"{tag['_id']} (in files: {', '.join(tag['files'][:3])})" for tag in top_tags])
            prompt = f"""Based on the following top tags and associated files from a user's document collection, generate {num_suggestions} topic suggestions. Each suggestion should include a title and a user prompt for an AI assistant named Aurora.

    Top Tags and Files: {tags_summary}

    Generate suggestions in the following JSON format:
    [
      {{
        "title": "Brief title for the topic",
        "userPrompt": "Hi Aurora, [request related to the topic]"
      }},
      ...
    ]
    """

            # Generate suggestions using the LLM
            response = await self.llm.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="mixtral-8x7b-32768"
            )

            # Parse the response
            suggestions = eval(response.choices[0].message.content.strip())

            return suggestions[:num_suggestions]
