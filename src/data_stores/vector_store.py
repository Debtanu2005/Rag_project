from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

_vector_store= None
class ChromaStore:
    def __init__(self, persist_directory="chroma_db"):
        global _vector_store
        self.persist_directory = persist_directory
        if _vector_store is None:
            _vector_store = Chroma(persist_directory=self.persist_directory, embedding_function=embeddings)
        self.client = _vector_store
        # self.collection = self.client.get_or_create_collection("my_collection")

    def add_documents(self, documents, metadata=None):
        self.client.add_documents(documents, metadatas=metadata)

    def query(self, query_text, n_results=5):
        results = self.client.query(query_text, n_results=n_results)
        return results