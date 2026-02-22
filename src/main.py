from datetime import datetime
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
        academic_year = self.get_academic_year()  
        if not academic_year:
            return []

        start_year, end_year = map(int, academic_year.split('-'))
        academic_start = datetime(start_year, 9, 1) 
        academic_end   = datetime(end_year, 5, 31) 

        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}-\d{1,2}-\d{4}',
            r'\d{1,2} de [a-záéíóú]+ de \d{4}',
        ]

        all_dates = []
        for pattern in date_patterns:
            all_dates.extend(re.findall(pattern, self.text, re.I))

        def parse_date(s):
            try:
                return datetime.strptime(s, "%d/%m/%Y")
            except:
                try:
                    return datetime.strptime(s, "%d-%m-%Y")
                except:
                    months = {
                        'enero':1,'febrero':2,'marzo':3,'abril':4,'mayo':5,'junio':6,
                        'julio':7,'agosto':8,'septiembre':9,'octubre':10,'noviembre':11,'diciembre':12
                    }
                    m = re.match(r'(\d{1,2}) de ([a-záéíóú]+) de (\d{4})', s.lower())
                    if m:
                        day, month, year = int(m.group(1)), months[m.group(2)], int(m.group(3))
                        return datetime(year, month, day)
                return None

        parsed_dates = [d for d in map(parse_date, all_dates) if d]

        valid_dates = [d for d in parsed_dates if academic_start <= d <= academic_end]

        if not valid_dates:
            return []

        valid_dates.sort()
        start_date = valid_dates[0]
        end_date   = valid_dates[-1]

        return [(start_date.strftime("%d/%m/%Y"), end_date.strftime("%d/%m/%Y"))]
    
    def get_amounts(self):
        pattern = r'(?:cuant[ií]a|beca|importe|ayuda|dotaci[oó]n).*?(\d{1,3}(?:\.\d{3})*,\d+\s?euros?)'
        matches = re.findall(pattern, self.text, re.I)

        cleaned = []
        for m in matches:
            num = float(m.replace('.', '').replace(',', '.').split()[0])
            if 100 <= num <= 6000:  # typical range of scholarship
                cleaned.append(m)

        return list(dict.fromkeys(cleaned))
    
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