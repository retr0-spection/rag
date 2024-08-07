import chromadb
import chromadb.utils.embedding_functions as embedding_functions

chroma_client = chromadb.PersistentClient(path="/tmp/db/chroma")

class Ingestion():
    def __init__(self):
        self.collection = None
        self.huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
            api_key="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",
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )



    def create_collection(self, collection_name):
        self.collection = chroma_client.create_collection(name=collection_name, embedding_function=self.huggingface_ef )
        return self.collection

    def get_collection(self, collection_name):

        self.collection = chroma_client.get_collection(name=collection_name,  embedding_function=self.huggingface_ef)
        return self.collection


    def add_documents(self):
        self.collection.upsert(
            documents=[
                "This is a document about pineapple",
                "This is a document about oranges"
            ],
            ids=["id1", "id2"],
        )

    def query(self, query):
        results = self.collection.query(
            query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
            n_results=2 # how many results to return
        )

        return results

    def _delete_collection(self, collection_name):
        chroma_client.delete_collection(name=collection_name)



_ = Ingestion()
_.create_collection('embeddings')
_.get_collection('embeddings')
_.add_documents()
results = _.query('')
print(results)
_._delete_collection('embeddings')
