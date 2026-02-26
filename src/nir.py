import re
import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

import torch
from langchain_community.document_loaders import PyPDFDirectoryLoader
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from ir import clean_text, split_by_article

LOCAL_MODEL_ID       = "Qwen/Qwen2.5-3B-Instruct"
CLOUD_MODEL_ID       = "mistralai/mistral-small-3.2-24b-instruct"
HALLUCINATION_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

def load_corpus(docs_dir: str) -> dict:
    loader = PyPDFDirectoryLoader(docs_dir)
    raw_docs = loader.load()

    for doc in raw_docs:
        doc.page_content = clean_text(doc.page_content)

    chunks = split_by_article(raw_docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 100]

    source_chunks = defaultdict(list)
    for chunk in chunks:
        source_chunks[chunk.metadata["source"]].append(chunk)

    result = {}
    for source_path, doc_chunks in source_chunks.items():
        filename = os.path.basename(source_path)
        doc_chunks.sort(key=lambda d: d.metadata.get("page", 0))
        full_text = "\n".join(c.page_content for c in doc_chunks)

        articles = {}
        for chunk in doc_chunks:
            m = re.match(r'Art[ií]culo\.?\s+(\d+)', chunk.page_content)
            if m:
                key = f"Artículo {m.group(1)}."
                articles[key] = chunk.page_content

        result[filename] = {"full_text": full_text, "articles": articles}
        print(f"Loaded: {filename}  ({len(articles)} articles found)")

    return result


def load_local_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_ID,
        dtype=torch.float16,
    ).to(device)
    model.eval()
    return model, tokenizer


def load_hallucination_model():
    tokenizer = AutoTokenizer.from_pretrained(HALLUCINATION_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HALLUCINATION_MODEL_ID)
    model.eval()
    return model, tokenizer


def score_factual_consistency(premise: str, hypothesis: str, halu_model, halu_tokenizer,) -> float:
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
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": full_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        input_ids = inputs["input_ids"]

        output_ids = model.generate(
            input_ids,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
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
    # Collapse double-double-quoted keys/values that some models emit when the source text contains quoted spans
    content = re.sub(r'""([^"]*?)""', r'"\1"', content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"Error decodifying JSON.")
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
    parser.add_argument('--use-cloud', action='store_true', help='Use mistral-small-3.2-24b via OpenRouter instead of local Qwen2.5-3B')
    args = parser.parse_args()

    if args.use_cloud:
        model, tokenizer = None, None
    else:
        model, tokenizer = load_local_model()

    halu_model, halu_tokenizer = load_hallucination_model()

    CORPUS_DIR = "../docs"

    corpus = load_corpus(CORPUS_DIR)

    all_scholarships = []

    for filename, doc_data in corpus.items():
        print(f"\nProcessing file {filename}...")
        scholarship_data = extract_full_corpus_data(
            doc_data["articles"], doc_data["full_text"],
            model=model, tokenizer=tokenizer,
            halu_model=halu_model, halu_tokenizer=halu_tokenizer,
        )

        all_scholarships.append(scholarship_data)

    # Save the LLM extraction output
    out_path = Path("../output/info_llm.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_scholarships, f, indent=4, ensure_ascii=False)
