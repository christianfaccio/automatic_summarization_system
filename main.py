import extractor
import json

CORPUS_DIR = "docs"

corpus = extractor.extract_text_from_pdfs(CORPUS_DIR)

#print(corpus["ayudas_21-22.pdf"])

first_doc_text = corpus["ayudas_21-22.pdf"]
segmented_doc = extractor.segment_by_articles(first_doc_text)
#print(segmented_doc.keys())
print(segmented_doc["Artículo 1."])