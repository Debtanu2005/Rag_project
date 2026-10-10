import fitz
import os
from PIL import Image
import io
import pytesseract
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf_to_page_images(pdf_path, output_dir="pdf_pages", dpi=250):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72),
            alpha=False
        )

        image_path = os.path.join(
            output_dir, f"page_{i + 1}.png"
        )

        pix.save(image_path)

        pages.append({
            "page": i + 1,
            "image_path": image_path
        })

    doc.close()
    return pages



def extract_ocr_text(image_path):
    image = Image.open(image_path).convert("RGB")

    text = pytesseract.image_to_string(
        image,
        config="--psm 3"
    )

    return text.strip()




def recursive_text_splitter(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    return text_splitter.split_documents(documents)