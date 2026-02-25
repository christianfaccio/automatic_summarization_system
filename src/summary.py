import json
from pathlib import Path
import argparse 

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default='../output/info_mistral.json')
args = parser.parse_args()

INPUT_FILE = args.input_file
OUTPUT_DIR = Path("../output/generated_summaries")

MODELS = {
    "phi35":   "microsoft/Phi-3.5-mini-instruct",
    "llama32": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen25":  "Qwen/Qwen2.5-3B-Instruct",
}

HALLUCINATION_MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

GENERATION_PARAMS = {
    "max_new_tokens": 300,
    "temperature": 0.4,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}

def build_prompt(year_data: dict) -> str:
    return (
        "You are an academic writer.\n"
        "Write a precise and structured summary (200-250 words).\n"
        "Do not invent information.\n\n"
        "DATA:\n"
        f"Academic year: {year_data.get('academic_year')}\n"
        f"Budget: {year_data.get('total_budget_general_millions_eur')} million euros\n"
        f"Regulation: {year_data.get('income_thresholds_defined_by')}\n"
        f"Programs: {year_data.get('eligible_programs')}\n"
        f"Scholarship components: {year_data.get('scholarship_components')}\n"
        f"Academic requirements: {year_data.get('academic_requirements')}\n"
        f"Application period: {year_data.get('application_period')}\n\n"
        "SUMMARY:\n"
    )

def generate_summary(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, **GENERATION_PARAMS)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("SUMMARY:")[-1].strip()

def load_model(model_id: str):
    """Load a causal-LM model and its tokenizer in float16 with auto device mapping."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer


def unload_model(*objects) -> None:
    """Delete model/tokenizer references and free GPU memory."""
    for obj in objects:
        del obj
    torch.cuda.empty_cache()


def load_hallucination_model():
    """Load the multilingual NLI model used to score factual consistency."""
    tokenizer = AutoTokenizer.from_pretrained(HALLUCINATION_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HALLUCINATION_MODEL_ID)
    model.eval()
    return model, tokenizer


def score_factual_consistency(
    premise: str,
    hypothesis: str,
    halu_model: AutoModelForSequenceClassification,
    halu_tokenizer: AutoTokenizer,
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

def run_pipeline_for_model(
    model_tag: str,
    model_id: str,
    data: dict,
    halu_model=None,
    halu_tokenizer=None,
) -> dict:
    """Load a model, generate summaries for every year, save them, and release GPU memory."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_id}")
    print(f"{'='*60}")

    model, tokenizer = load_model(model_id)
    summaries = {}
    halu_scores = {}

    for year_name, year_data in data.items():
        print(f"  Generating summary for {year_name} ...")
        prompt = build_prompt(year_data)
        summary = generate_summary(model, tokenizer, prompt)
        summaries[year_name] = summary

        # Save each summary immediately to avoid data loss
        output_path = OUTPUT_DIR / f"{year_name}_{model_tag}.txt"
        output_path.write_text(summary, encoding="utf-8")

        if halu_model is not None:
            premise = json.dumps(year_data, ensure_ascii=False)
            score = score_factual_consistency(premise, summary, halu_model, halu_tokenizer)
            halu_scores[year_name] = round(score, 4)
            print(f"    Hallucination score ({year_name}): {score:.4f}")

    print(f"    All summaries for {model_tag} saved to '{OUTPUT_DIR}/'")

    if halu_scores:
        scores_path = OUTPUT_DIR / f"{model_tag}_hallucination_scores.json"
        scores_path.write_text(json.dumps(halu_scores, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    Hallucination scores saved to '{scores_path}'")

    unload_model(model, tokenizer)
    return summaries


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # nir.py saves a list of per-document dicts; convert to {year: entry}
    if isinstance(raw, list):
        data = {}
        for i, entry in enumerate(raw):
            year = (
                entry.get("academic_year")
                or entry.get("general_info", {}).get("academic_year")
                or f"document_{i}"
            )
            data[year] = entry
    else:
        data = raw

    print("Loading hallucination evaluation model...")
    halu_model, halu_tokenizer = load_hallucination_model()

    # Run each model sequentially (one at a time to fit in GPU memory)
    all_summaries = {}
    for model_tag, model_id in MODELS.items():
        all_summaries[model_tag] = run_pipeline_for_model(
            model_tag, model_id, data, halu_model, halu_tokenizer
        )

    print("\n   All models processed successfully.")