from langchain_core.documents import Document
from utils import pdf_to_page_images, extract_ocr_text, recursive_text_splitter

from data_stores.vector_store import ChromaStore
from data_stores.bm25 import BM25Store

vector_store = ChromaStore(persist_directory="chroma_db")
bm25_store = BM25Store(k=3)


class PDFLoader:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def load_and_process_pdf(self):
        # Step 1: Convert PDF to images
        pages = pdf_to_page_images(self.pdf_path)

        documents = []
        for page in pages:
            # Step 2: Extract text from each image using OCR
            text = extract_ocr_text(page["image_path"])

            # Step 3: Create a Document object for each page
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": self.pdf_path,
                        "page_number": page["page"]
                    }
                )
            )

        # Step 4: Split the text into smaller chunks
        split_documents = recursive_text_splitter(documents)

        return split_documents

    def add_to_stores(self):
        split_documents = self.load_and_process_pdf()
        vector_store.add_documents(split_documents)
        bm25_store.add_documents(split_documents)