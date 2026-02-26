from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from collections import defaultdict
import json, os, re, nltk

nltk.download('punkt_tab', quiet=True)

docs_path = '../docs/'
output_path = '../output/'

def clean_text(text):
    text = re.sub(r'CSV\s*:.*', '', text)
    text = re.sub(r'DIRECCIÓN DE VALIDACIÓN\s*:.*', '', text)
    text = re.sub(r'FIRMANTE\(\d+\).*', '', text)
    text = re.sub(r'ABELARDO DE LA ROSA.*CET', '', text)
    text = re.sub(r'Puede comprobar la autenticidad.*', '', text)
    text = re.sub(r'Código seguro de Verificación\s*:.*', '', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def split_by_article(docs):
    article_pattern = re.compile(r'(?=Art[ií]culo\.?\s+\d+[\.\s\-])')

    source_pages = defaultdict(list)
    for doc in docs:
        source_pages[doc.metadata['source']].append(doc)
    for source in source_pages:
        source_pages[source].sort(key=lambda d: d.metadata['page'])

    chunks = []
    for source, pages in source_pages.items():
        current_text = ''
        current_page = None

        for page_doc in pages:
            page_num = page_doc.metadata['page']
            segments = article_pattern.split(page_doc.page_content)

            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue
                if re.match(r'Art[ií]culo\.?\s+\d+', segment):
                    if current_text.strip():
                        chunks.append(Document(
                            page_content=current_text.strip(),
                            metadata={'source': source, 'page': current_page},
                        ))
                    current_text = segment
                    current_page = page_num
                else:
                    current_text = (current_text + '\n' + segment).strip()
                    if current_page is None:
                        current_page = page_num

        if current_text.strip():
            chunks.append(Document(
                page_content=current_text.strip(),
                metadata={'source': source, 'page': current_page},
            ))

    return chunks

def parse_amount(s):
    """
    Convert a Spanish-format number string to an integer.
    """
    if not s:
        return None
    # Remove thousands-separator dots (dot followed by exactly 3 digits)
    s = re.sub(r'\.(?=\d{3}(?:[.,]|$))', '', s)
    s = s.replace(',', '.')
    try:
        return int(float(s))
    except ValueError:
        return None


def extract_academic_year(full_text):
    m = re.search(r'CURSO ACAD[EÉ]MICO\s+(\d{4}[-–]\d{2,4})', full_text, re.IGNORECASE)
    return m.group(1) if m else None


def extract_total_budget(full_text):
    m = re.search(
        r'([\d.,]+)\s+millones\s+se invertir[aá]n en\s+(?:las\s+)?becas de car[aá]cter general',
        full_text, re.IGNORECASE,
    )
    if m:
        return parse_amount(m.group(1))
    # Fallback for years that phrase it differently
    m = re.search(
        r'([\d.,]+)\s+millones[^,\n]*?\ben\s+(?:las\s+)?becas de car[aá]cter general',
        full_text, re.IGNORECASE,
    )
    return parse_amount(m.group(1)) if m else None


def extract_decree_reference(full_text):
    m = re.search(
        r'(Real Decreto\s+[\d/]+,\s+de\s+\d+\s+de\s+\w+[^,]*?),?\s+'
        r'(?:ha establecido|por el que se establecen?)\s+los umbrales',
        full_text, re.IGNORECASE,
    )
    # Collapse any OCR-introduced line breaks inside the matched string
    return re.sub(r'\s+', ' ', m.group(1)).strip() if m else None


def extract_eligible_programs(text):
    university = []
    non_university = []
    current_category = None

    for m in re.finditer(r'\n\s*(\d+\.|[a-z]\))\s+([^\n]{10,})', text):
        marker = m.group(1)
        item = m.group(2).strip().rstrip('.')

        if re.match(r'\d+\.', marker):
            # Section header — decide category from its text
            is_university = bool(re.search(r'universitari[ao]', item, re.IGNORECASE))
            is_non        = bool(re.search(r'no universitari',  item, re.IGNORECASE))
            if is_university and not is_non:
                current_category = 'university'
            else:
                current_category = 'non_university'
        elif current_category is not None:
            if current_category == 'university':
                university.append(item)
            else:
                non_university.append(item)

    return {'university': university, 'non_university': non_university}


def extract_scholarship_amounts(text):
    """
    Extract all scholarship component amounts from Art. 11 text.
    """
    amounts = {}

    # Tuition (beca de matrícula) — always covers public fees, no fixed amount
    if re.search(r'beca de matr[ií]cula', text, re.IGNORECASE):
        amounts['tuition'] = 'covered_by_public_fees'

    # Fixed income-linked
    m = re.search(
        r'cuant[ií]\s*a\s+fija\s+ligada\s+a\s+la\s+renta[^:]*:\s*([\d.,]+)\s*euros',
        text, re.IGNORECASE,
    )
    if m:
        amounts['fixed_income_eur'] = parse_amount(m.group(1))

    # Fixed residency
    m = re.search(
        r'cuant[ií]\s*a\s+fija\s+ligada\s+a\s+la\s+residencia[^:]*:\s*([\d.,]+)\s*euros',
        text, re.IGNORECASE,
    )
    if m:
        amounts['residency_eur'] = parse_amount(m.group(1))

    m_excel = re.search(r'excelencia.{0,300}?distribución[:\s]*(.*?)(?=\n[a-z]\)|\Z)',
                        text, re.IGNORECASE | re.DOTALL)
    if m_excel:
        excel_text = m_excel.group(1)
        tier_amounts = re.findall(r'\b(\d+)\s*euros', excel_text)
        if len(tier_amounts) >= 4:
            amounts['fixed_excellence_by_gpa'] = {
                '8.00-8.49': int(tier_amounts[0]),
                '8.50-8.99': int(tier_amounts[1]),
                '9.00-9.49': int(tier_amounts[2]),
                '9.50+':     int(tier_amounts[3]),
            }
    # Fallback: if table not found, at least store the stated min/max range
    if 'fixed_excellence_by_gpa' not in amounts:
        m = re.search(r'excelencia[^:]*:\s*entre\s+([\d.,]+)\s+y\s+([\d.,]+)\s*euros',
                      text, re.IGNORECASE)
        if m:
            amounts['fixed_excellence_min_eur'] = parse_amount(m.group(1))
            amounts['fixed_excellence_max_eur'] = parse_amount(m.group(2))

    # Basic grant — general amount
    m = re.search(r'[Bb]eca b[aá]sica:\s*([\d.,]+)\s*euros', text, re.IGNORECASE)
    if m:
        amounts['basic_grant_eur'] = parse_amount(m.group(1))
    m = re.search(r'[Gg]rado [Bb][aá]sico\D{0,60}?(\d+)\s*euros', text, re.IGNORECASE)
    if m:
        amounts['basic_grant_basic_cycle_eur'] = parse_amount(m.group(1))

    # Variable minimum
    m = re.search(
        r'importe m[ií]nimo ser[aá]\s+de\s+([\d.,]+)\s*euros',
        text, re.IGNORECASE,
    )
    if m:
        amounts['variable_min_eur'] = parse_amount(m.group(1))

    return amounts


def extract_application_period(text):
    """
    Extract the application window from Art. 48.
    """
    period = {}
    MONTHS = (r'enero|febrero|marzo|abril|mayo|junio|julio|agosto|'
               r'septiembre|octubre|noviembre')

    m = re.search(r'desde el d[ií]a\s+(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})',
                  text, re.IGNORECASE)
    if m:
        period['start'] = m.group(1).strip()

    m = re.search(
        r'hasta el\s+(?:d[ií]a\s+)?'
        r'(\d{1,2}\s+de\s+(?:' + MONTHS + r')\s+de\s+\d{4})',
        text, re.IGNORECASE,
    )
    if m:
        period['end'] = m.group(1).strip()

    # Exception deadline (always "31 de diciembre de YYYY")
    m = re.search(r'hasta el\s+31\s+de\s+diciembre\s+de\s+(\d{4})', text, re.IGNORECASE)
    if m:
        period['exception_until'] = f'31 de diciembre de {m.group(1)}'

    return period if period else None


def extract_academic_requirements(text_credits, text_gpa):
    """
    Extract minimum credit load (university) and minimum GPA.
    """
    reqs = {}

    # Min credits for full-time university enrollment.
    # Art. 23 first states the full-time requirement (60 créditos) and then
    # mentions the partial-enrollment exception (entre 30 y 59 créditos).
    # Searching specifically for "60" avoids capturing the lower partial value.
    m = re.search(r'\b60\s+cr[eé]?\s*ditos', text_credits, re.IGNORECASE)
    if m:
        reqs['university_min_credits'] = 60

    m = re.search(r'nota media de\s+(\d+[,.]\d+)\s*puntos', text_gpa, re.IGNORECASE)
    if m:
        reqs['min_gpa'] = float(m.group(1).replace(',', '.'))

    return reqs if reqs else None

# helper
def retrieve(bm25, query, k=2):
    """Return concatenated text of the top-k BM25 results for a query."""
    bm25.k = k
    return '\n\n'.join(d.page_content for d in bm25.invoke(query))

if __name__ == "__main__":
    loader = PyPDFDirectoryLoader(docs_path)
    raw_docs = loader.load()

    for doc in raw_docs:
        doc.page_content = clean_text(doc.page_content)

    chunks = split_by_article(raw_docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 100]

    # Full concatenated text per source
    full_texts = defaultdict(str)
    for chunk in chunks:
        full_texts[chunk.metadata['source']] += '\n' + chunk.page_content

    output = {}

    for source in sorted(os.listdir(docs_path)):
        source_path = f'../docs/{source}'
        source_chunks = [c for c in chunks if c.metadata['source'] == source_path]

        bm25 = BM25Retriever.from_documents(source_chunks, k=2)
        full_text = full_texts[source_path]

        data = {
            'academic_year': extract_academic_year(full_text),
            'total_budget_general_millions_eur': extract_total_budget(full_text),
            'income_thresholds_defined_by': extract_decree_reference(full_text),
            'eligible_programs': extract_eligible_programs(
                retrieve(bm25, 'enseñanzas universitarias bachillerato formación profesional ámbito', k=1)
            ),
            'scholarship_components': extract_scholarship_amounts(
                retrieve(bm25, 'cuantías becas matrícula renta residencia excelencia básica variable euros', k=3)
            ),
            'application_period': extract_application_period(
                retrieve(bm25, 'plazo presentación solicitud fecha día mes año desde hasta', k=3)
            ),
            'academic_requirements': extract_academic_requirements(
                retrieve(bm25, 'créditos matrícula número dedicación tiempo completo artículo 23', k=2),
                retrieve(bm25, 'nota media puntos mínimo requisito obtener beca año anterior rendimiento', k=2),
            ),
        }

        output[source] = data

    with open(output_path + 'info_regex.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
