import chromadb
import chromadb.utils.embedding_functions as embedding_functions
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

# Download NLTK data (run this once)
nltk.download('punkt')

# Create a temporary directory
temp_dir = tempfile.mkdtemp()

chroma_client = chromadb.PersistentClient(path=os.path.join(temp_dir, "chroma"))

class Ingestion:
    def __init__(self):
        self.collection = None
        self.embed_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )

    def create_collection(self, collection_name):
        self.collection = chroma_client.create_collection(name=collection_name, embedding_function=self.embed_ef)
        return self.collection

    def get_or_create_collection(self, collection_name):
        try:
            self.collection = chroma_client.get_collection(name=collection_name, embedding_function=self.embed_ef)
            return self.collection
        except Exception:
            self.collection = chroma_client.create_collection(name=collection_name, embedding_function=self.embed_ef)
            return self.collection


    def process_document(self, file_path: str) -> str:
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()

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
        self.collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas
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

    def query(self, query_text: str, user_id: str, n_results: int = 2):
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"user_id": user_id}  # Filter by user_id
        )
        return results

    def delete_collection(self, collection_name):
        chroma_client.delete_collection(name=collection_name)
