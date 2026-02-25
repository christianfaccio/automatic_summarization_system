import fitz  #PyMuPDF
import re
import os
import json
import argparse
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
CLOUD_MODEL_ID = "mistralai/mistral-small-3.2-24b-instruct"

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


# Configuration of prompts and targets
EXTRACTION_TASKS = {
    "General": {
        "source": "general_info",
        "prompt": "Identify the academic year (e.g., 2024-2025). Return JSON: {academic_year: str}."
    },
    "Artículo 48.": {
        "source": "general_info",
        "prompt": "Identify the deadlines with DD/MM/YYYY format. Only if the deadline includes a starting date, the format should be DD/MM/YYYY - DD/MM/YYYY. Return JSON: {deadlines: {university: str, non_university: str, exception: str}}."
    },
    "Artículo 3.": {
        "key": "eligible_programs",
        "prompt": "Extract educational levels. Group into JSON: {university_programs: [str, str...], non_university_programs: [str, str..]}."
    },
    "Artículo 11.": {
        "key": "scholarship_amounts",
        "prompt": "Extract fixed/variable amounts. JSON keys: fixed_income, fixed_residence, fixed_excellence_range (dict with gpas as keys), basic_scholarship (dict with general and basic_cycle as keys), min_variable."
    },
    "Artículo 19.":{
        "key": "income_thresholds",
        "prompt": "Extract income thresholds by threshold number and by number of family members. JSON keys: umbral_1, umbral_2, umbral_3. Values for each key will be Python dictionaries with the number of family members as keys (final key will be additional_member) and the total income as values."
    }
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

def extract_full_corpus_data(segmented_doc, raw_text, model=None, tokenizer=None):
    final_json = {}

    for section, task in EXTRACTION_TASKS.items():
        # Determine context: specific article or the whole header
        context = raw_text[:2000] if section == "General" else segmented_doc.get(section, "")

        if context:
            print(f"Processing {section}...")
            extracted_part = call_llm_for_json(task["prompt"], context, model=model, tokenizer=tokenizer)

            # Merge into the final document structure
            key = task.get("key", "general_info")

            # If the key already exists, update it (academic years and deadlines are in the same key)
            if key in final_json and isinstance(final_json[key], dict):
                final_json[key].update(extracted_part)
            else:
                final_json[key] = extracted_part

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

    CORPUS_DIR = "../docs"

    corpus = extract_text_from_pdfs(CORPUS_DIR)

    all_scholarships = []

    for filename, text in corpus.items():
        segmented_doc = segment_by_articles(text)

        print(f"Processing file {filename}...")
        scholarship_data = extract_full_corpus_data(segmented_doc, text, model=model, tokenizer=tokenizer)
        print(f"Finished processing file {filename}")

        all_scholarships.append(scholarship_data)

    # Save the master "Ground Truth" file
    with open("output/info_mistral.json", "w", encoding="utf-8") as f:
        json.dump(all_scholarships, f, indent=4, ensure_ascii=False)
