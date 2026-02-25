import fitz  #PyMuPDF
import re
import os
import json
import argparse
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

LOCAL_MODEL_ID       = "Qwen/Qwen2.5-3B-Instruct"
CLOUD_MODEL_ID       = "mistralai/mistral-small-3.2-24b-instruct"
HALLUCINATION_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

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
                header_margin_percent, footer_margin_percent = 5, 10
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
            print(f"Finished extracting text from: {filename}")

    return corpus_text


def segment_by_articles(text):
    # Find "Artículo 1.", "Artículo 2.", etc. We divide the text by articles
    pattern = r'(Artículo\s+\d+\.)'

    # Split the text by the pattern
    parts = re.split(pattern, text)

    articles_dict = {}
    # Iterate through the split parts (skip first part because is the header)
    for i in range(1, len(parts), 2):
        article_title = parts[i].strip()
        article_content = parts[i+1].strip()
        articles_dict[article_title] = article_content

    return articles_dict


def load_local_model():
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer


def load_hallucination_model():
    """Load the multilingual NLI model used to score factual consistency."""
    tokenizer = AutoTokenizer.from_pretrained(HALLUCINATION_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HALLUCINATION_MODEL_ID)
    model.eval()
    return model, tokenizer


def score_factual_consistency(
    premise: str,
    hypothesis: str,
    halu_model,
    halu_tokenizer,
) -> float:
    """Return entailment probability: 1.0 = fully supported, 0.0 = hallucinated."""
    inputs = halu_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = halu_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    label2id = {v: k for k, v in halu_model.config.id2label.items()}
    return probs[label2id["entailment"]].item()


# Configuration of prompts and targets.
# key=None means the extracted fields are merged at the top level of the document dict.
# All keys and sub-keys mirror ir.py's output schema so summary.py can use either file.
EXTRACTION_TASKS = {
    "General": {
        "key": None,  # top-level merge
        "prompt": (
            "From this Spanish scholarship regulation preamble extract three things:\n"
            "1. The academic year (e.g. '2024-2025').\n"
            "2. The total budget in millions of euros as a plain number (no units).\n"
            "3. The full 'Real Decreto' reference string that defines income thresholds.\n"
            "Return JSON: {\"academic_year\": str, "
            "\"total_budget_general_millions_eur\": number, "
            "\"income_thresholds_defined_by\": str}."
        ),
    },
    "Artículo 3.": {
        "key": "eligible_programs",
        "prompt": (
            "Extract the educational programs eligible for scholarships.\n"
            "Return JSON: {\"university\": [str, ...], \"non_university\": [str, ...]}."
        ),
    },
    "Artículo 11.": {
        "key": "scholarship_components",
        "prompt": (
            "Extract scholarship component amounts present in the text.\n"
            "Return JSON with only the keys that appear: "
            "tuition (str, e.g. 'covered_by_public_fees'), "
            "fixed_income_eur (int), residency_eur (int), "
            "fixed_excellence_by_gpa (object mapping GPA-range strings to euro int amounts), "
            "basic_grant_eur (int), basic_grant_basic_cycle_eur (int), variable_min_eur (int)."
        ),
    },
    "Artículo 19.": {
        "key": "income_thresholds",
        "prompt": (
            "Extract income thresholds by threshold number and number of family members.\n"
            "Return JSON: {\"umbral_1\": {\"1\": int, ..., \"additional_member\": int}, "
            "\"umbral_2\": {...}, \"umbral_3\": {...}}."
        ),
    },
    "Artículo 23.": {
        "key": "academic_requirements",
        "prompt": (
            "Extract minimum academic requirements for scholarship eligibility.\n"
            "Return JSON: {\"university_min_credits\": int, \"min_gpa\": float}."
        ),
    },
    "Artículo 48.": {
        "key": "application_period",
        "prompt": (
            "Extract the application period dates.\n"
            "Return JSON: {\"start\": str, \"end\": str, \"exception_until\": str} "
            "using natural-language dates (e.g. '14 de junio de 2023'). Omit missing keys."
        ),
    },
}

def call_llm_for_json(prompt, context, model=None, tokenizer=None):
    """Sends the context and prompt to the LLM and parses JSON response."""
    system_msg = "You are a legal data extractor, specialized in Spanish documents about scholarships. Output ONLY raw JSON."
    full_prompt = f"TEXTO DEL DOCUMENTO: \n{context}\n\nINSTRUCCIÓN:\n{prompt}"

    if model is not None:
        # Local inference via transformers
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": full_prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        output_ids = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=False,
        )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        content = tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        # Cloud inference via OpenRouter
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="api_key", # Replace with the actual API KEY
        )

        completion = client.chat.completions.create(
            model=CLOUD_MODEL_ID,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": full_prompt}
            ],
            response_format={ "type": "json_object" }
        )
        content = completion.choices[0].message.content

    content = content.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"Error decodificando JSON. El modelo respondió: {content}")
        return {}

def extract_full_corpus_data(segmented_doc, raw_text, model=None, tokenizer=None, halu_model=None, halu_tokenizer=None):
    final_json = {}
    hallucination_scores = {}

    for section, task in EXTRACTION_TASKS.items():
        # General uses the document preamble; everything else uses its specific article
        context = raw_text[:5000] if section == "General" else segmented_doc.get(section, "")

        if not context:
            continue

        print(f"Processing {section}...")
        extracted_part = call_llm_for_json(task["prompt"], context, model=model, tokenizer=tokenizer)

        if halu_model is not None and extracted_part:
            hypothesis = json.dumps(extracted_part, ensure_ascii=False)
            score = score_factual_consistency(context, hypothesis, halu_model, halu_tokenizer)
            hallucination_scores[section] = round(score, 4)
            print(f"    Hallucination score: {score:.4f}")

        key = task.get("key")
        if key is None:
            # Merge all fields at the top level (used for General section)
            final_json.update(extracted_part)
        elif key in final_json and isinstance(final_json[key], dict):
            final_json[key].update(extracted_part)
        else:
            final_json[key] = extracted_part

    if hallucination_scores:
        final_json["hallucination_scores"] = hallucination_scores

    return final_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cloud', action='store_true',
                        help='Use mistral-small-3.2-24b via OpenRouter instead of local Qwen2.5-3B')
    args = parser.parse_args()

    if args.use_cloud:
        model, tokenizer = None, None
    else:
        model, tokenizer = load_local_model()

    print("Loading hallucination evaluation model...")
    halu_model, halu_tokenizer = load_hallucination_model()

    CORPUS_DIR = "../docs"

    corpus = extract_text_from_pdfs(CORPUS_DIR)

    all_scholarships = []

    for filename, text in corpus.items():
        segmented_doc = segment_by_articles(text)

        print(f"Processing file {filename}...")
        scholarship_data = extract_full_corpus_data(
            segmented_doc, text,
            model=model, tokenizer=tokenizer,
            halu_model=halu_model, halu_tokenizer=halu_tokenizer,
        )
        print(f"Finished processing file {filename}")

        all_scholarships.append(scholarship_data)

    # Save the master "Ground Truth" file
    with open("output/info_mistral.json", "w", encoding="utf-8") as f:
        json.dump(all_scholarships, f, indent=4, ensure_ascii=False)
