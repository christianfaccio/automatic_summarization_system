import pdfplumber
import json
import re
import os

class Extraction:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = self.get_text()

    def clean_text(self, text):
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w+)- (\w+)', r'\1\2', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'[A-Z0-9\-]{20,}', '', text)
        text = re.sub(r'\S*//:sptth\S*', '', text)

        return text.strip()

    def get_text(self):
        text = ""
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return self.clean_text(text)

    def get_academic_year(self):
        match = re.findall(r'\d{4}\s?-\s?\d{4}', self.text)
        return match[0] if match else None
    
    def get_ministry(self):
        match = re.search(r'(MINISTERIO DE [A-ZÁÉÍÓÚÜÑ ]+)', self.text)
        return match.group(1).strip() if match else None
    
    def get_deadlines(self):
        dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', self.text)
        return list(dict.fromkeys(dates))
    
    def get_amounts(self):
        pattern = r'(?:cuant[ií]a|beca|importe|ayuda|dotaci[oó]n).*?(\d{1,3}(?:\.\d{3})*,\d+\s?euros?)'
        matches = re.findall(pattern, self.text, re.I)

        return list(dict.fromkeys(matches))
    
    def get_title(self):
        match = re.search(
            r'(RESOLUCIÓN.*?CURSO ACADÉMICO \d{4}-\d{4}.*?)\.',
            self.text,
            re.I
        )
        return match.group(1).strip() if match else None
    
    def get_all(self):
        return {
            "academic_year": self.get_academic_year(),
            "institution": self.get_ministry(),
            "title": self.get_title(),
            "deadlines": self.get_deadlines(),
            "amounts": self.get_amounts()
        }
    
def process_pdfs(pdf_folder, output_json="../output/scholarships.json"):
    data = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            extractor = Extraction(os.path.join(pdf_folder, filename))
            data.append(extractor.get_all())

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data extracted to {output_json}")

if __name__ == "__main__":
    process_pdfs("../docs")