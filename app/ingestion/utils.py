import pymongo
from pymongo import MongoClient, ASCENDING
from bson.son import SON
from typing import List, Dict
import tempfile
import os
import re
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import mimetypes
from sentence_transformers import SentenceTransformer

# Download NLTK data (run this once)
nltk.download('punkt')

# Create a temporary directory
temp_dir = tempfile.mkdtemp()

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['document_db']

class Ingestion:
    def __init__(self):
        self.collection = None
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def create_collection(self, collection_name):
        self.collection = db[collection_name]
        # Create an ascending index on the embedding field
        self.collection.create_index([("embedding", ASCENDING)], background=True)
        return self.collection

    def get_or_create_collection(self, collection_name):
        self.collection = db[collection_name]
        if collection_name not in db.list_collection_names():
            # Create an ascending index on the embedding field
            self.collection.create_index([("embedding", ASCENDING)], background=True)
        return self.collection

    def clean_text(self, text: str) -> str:
        # Preserve URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        for i, url in enumerate(urls):
            text = text.replace(url, f'[URL{i}]')

        # Remove special characters except periods, commas, and hyphens
        text = re.sub(r'[^\w\s.,;:?!-]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Restore URLs
        for i, url in enumerate(urls):
            text = text.replace(f'[URL{i}]', url)

        return text

    def chunk_text(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def add_documents(self, documents: List[str], ids: List[str], metadatas: List[Dict] = None):
        if metadatas is None:
            metadatas = [{}] * len(documents)

        embeddings = self.model.encode(documents)

        for doc, id, metadata, embedding in zip(documents, ids, metadatas, embeddings):
            self.collection.update_one(
                {'_id': id},
                {'$set': {
                    'text': doc,
                    'metadata': metadata,
                    'embedding': embedding.tolist()
                }},
                upsert=True
            )

    def process_document(self, file_path: str) -> str:
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type == 'application/pdf':
            return self.process_pdf(file_path)
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return self.process_docx(file_path)
        elif mime_type == 'text/plain':
            return self.process_txt(file_path)
        elif mime_type == 'text/html':
            return self.process_html(file_path)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")

    def process_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def process_docx(self, file_path: str) -> str:
        doc = Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def process_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def process_html(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            return soup.get_text()

    def add_file(self, file_path: str, user_id: str):
        try:
            raw_text = self.process_document(file_path)
            cleaned_text = self.clean_text(raw_text)
            chunks = self.chunk_text(cleaned_text)

            # Incorporate user_id into IDs and metadata
            ids = [f"{user_id}_{os.path.basename(file_path)}_{i}" for i in range(len(chunks))]
            metadatas = [{"user_id": user_id, "source": file_path, "chunk": i} for i in range(len(chunks))]

            self.add_documents(chunks, ids, metadatas)
        except Exception as e:
            print(f"Error processing file {file_path} for user {user_id}: {str(e)}")

    def query(self, query_text: str, user_id: str, n_results: int = 10):
            query_embedding = self.model.encode([query_text])[0].tolist()

            pipeline = [
                {
                    '$match': {
                        'metadata.user_id': user_id
                    }
                },
                {
                    '$addFields': {
                        'similarity': {
                            '$reduce': {
                                'input': {'$zip': {'inputs': ['$embedding', query_embedding]}},
                                'initialValue': 0,
                                'in': {
                                    '$add': ['$$value', {'$multiply': [{'$arrayElemAt': ['$$this', 0]}, {'$arrayElemAt': ['$$this', 1]}]}]
                                }
                            }
                        }
                    }
                },
                {
                    '$sort': SON([('similarity', -1)])
                },
                {
                    '$limit': n_results
                },
                {
                    '$project': {
                        'text': 1,
                        'metadata': 1,
                        'similarity': 1
                    }
                }
            ]

            results = self.collection.aggregate(pipeline)
            return list(results)

    def delete_collection(self, collection_name):
        db.drop_collection(collection_name)
