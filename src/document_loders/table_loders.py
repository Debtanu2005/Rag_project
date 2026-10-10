import pandas as pd
from data_stores.bm25 import BM25Store
from langchain_core.documents import Document

from data_stores.vector_store import ChromaStore
from data_stores.bm25 import BM25Store

vector_store = ChromaStore(persist_directory="chroma_db")
bm25_store = BM25Store(k=3)


df = pd.read_csv('data.csv')

def create_summary(row):
    summary = []

    for col in df.columns:
        summary.append(f"{col.replace('_', ' ')} is {row[col]}")

    return ", ".join(summary)

df["summary"] = df.apply(create_summary, axis=1)



CHUNK_SIZE = 30
documents = []

for i in range(0, len(df), CHUNK_SIZE):
    batch = df.iloc[i:i+CHUNK_SIZE]

    text = " ".join(
        f"Row {idx}: {row['summary']}"
        for idx, row in batch.iterrows()
    )

    documents.append(
        Document(
            page_content=text,
            metadata={
                "source": "housing.csv",
                "start_row": int(batch.index[0]),
                "end_row": int(batch.index[-1])
            }
        )
    )

def initialize_stores():
    print("Initializing vector store and BM25 store with documents...")
    vector_store.add_documents(documents)
    bm25_store.add_documents(documents)


