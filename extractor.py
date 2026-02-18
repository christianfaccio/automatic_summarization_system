import fitz  #PyMuPDF
import os

CORPUS_DIR = "docs"

def extract_text_from_pdfs(docs_folder):
    corpus_text = {}

    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            path = os.path.join(docs_folder, filename)
            doc = fitz.open(path)
            full_text = []

            for page in doc:
                # Obtain page dimension
                page_rect = page.rect
                page_height = page_rect.height
                page_width = page_rect.width
                
                # Compute margins
                header_margin_percent, footer_margin_percent = 10, 10
                left_margin_percent, right_margin_percent = 10, 10
    
                header_margin = page_height * (header_margin_percent / 100)
                footer_margin = page_height * (footer_margin_percent / 100)
                left_margin = page_width * (left_margin_percent / 100)
                right_margin = page_width * (right_margin_percent / 100)
                
                # Create rectangle in which the important content is (no header, footer, left and right margins)
                content_rect = fitz.Rect(left_margin, header_margin, page_width - right_margin, page_height - footer_margin)
            
                text = page.get_text("text", clip=content_rect)
                
                # Cleaning: Remove the common BOE footer pattern (CSV validation codes)
                lines = [line for line in text.split('\n') if "CSV :" not in line and "DIRECCIÓN DE VALIDACIÓN" not in line]
                full_text.append("\n".join(lines))
            
            corpus_text[filename] = "\n".join(full_text)
            print(f"Finished: {filename}")
            
    return corpus_text


corpus = extract_text_from_pdfs(CORPUS_DIR)

print(corpus["ayudas_21-22.pdf"])