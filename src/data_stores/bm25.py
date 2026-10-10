from langchain_community.retrievers import BM25Retriever

_bm25_store = None

class BM25Store:
    def __init__(self, documents=None, k=3):
        global _bm25_store

        self.k = k

        if _bm25_store is None and documents:
            _bm25_store = BM25Retriever.from_documents(
                documents,
                k=k
            )

        self.client = _bm25_store

    def add_documents(self, documents):
        if self.client is None:
            self.client = BM25Retriever.from_documents(
                documents,
                k=self.k
            )
            global _bm25_store
            _bm25_store = self.client
        else:
            self.client.add_documents(documents)

    def query(self, query_text, n_results=None):
        if self.client is None:
            raise ValueError("BM25 index has not been initialized")

        self.client.k = n_results or self.k
        return self.client.invoke(query_text)