import extractor
import json

CORPUS_DIR = "docs"

corpus = extractor.extract_text_from_pdfs(CORPUS_DIR)

all_scholarships = []

for filename, text in corpus.items():
    segmented_doc = extractor.segment_by_articles(text) 
    
    print(f"Processing file {filename}...")
    scholarship_data = extractor.extract_full_corpus_data(segmented_doc, text)
    print(f"Finished processing file {filename}")
    
    all_scholarships.append(scholarship_data)

# Save the master "Ground Truth" file
with open("output/scholarship_corpus.json", "w", encoding="utf-8") as f:
    json.dump(all_scholarships, f, indent=4, ensure_ascii=False)
