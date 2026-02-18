from extractor import extract_text_from_pdfs

CORPUS_DIR = "docs"

corpus = extract_text_from_pdfs(CORPUS_DIR)

print(corpus["ayudas_21-22.pdf"])