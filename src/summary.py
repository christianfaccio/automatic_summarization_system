import json
from transformers import pipeline

def generate_summary(json_file, output_file="output/summary.txt", model_name="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model_name)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert JSON to plain text
    text_for_summary = ""
    for item in data:
        text_for_summary += f"Academic Year: {item.get('academic_year', 'N/A')}. "
        text_for_summary += f"Institution: {item.get('institution', 'N/A')}. "
        text_for_summary += f"Title: {item.get('title', 'N/A')}. "
        deadlines = ", ".join(item.get("deadlines", []))
        text_for_summary += f"Deadlines: {deadlines}. "
        amounts = ", ".join(item.get("amounts", []))
        text_for_summary += f"Scholarship Amounts: {amounts}. "
        text_for_summary += "\n"

    # Generate summary (divide in chunks if the text is long)
    max_chunk = 1024  # approximate tokens
    summaries = []
    for i in range(0, len(text_for_summary), max_chunk):
        chunk = text_for_summary[i:i+max_chunk]
        summary = summarizer(chunk, max_length=200, min_length=80, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    # Combine chunks
    final_summary = " ".join(summaries)

    # Save summary
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_summary)

if __name__ == "__main__":
    generate_summary("output/scholarships.json")