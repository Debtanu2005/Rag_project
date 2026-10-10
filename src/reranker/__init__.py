from sentence_transformers import CrossEncoder

class HybridReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", top_k=5, threshold=-2.0, device="cuda"):
        self.model = CrossEncoder(model_name, device=device)
        self.top_k = top_k
        self.threshold = threshold

    def rerank(self, query, documents):
        if not documents:
            return []
        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)
        ranked_docs = sorted(zip(documents, scores), key=lambda x: float(x[1]), reverse=True)
        results = []
        for doc, score in ranked_docs:
            if float(score) >= self.threshold:
                doc.metadata["rerank_score"] = float(score)
                results.append(doc)
            if len(results) >= self.top_k:
                break
        return results