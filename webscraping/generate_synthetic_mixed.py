import os
import io
import time
import re
import logging
from typing import Set
from difflib import SequenceMatcher
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Fast_Mixed_Generator")

def is_too_similar(new_text: str, existing_set: set, threshold: float = 0.70) -> bool:
    new_clean = new_text.strip().lower()
    for existing in existing_set:
        if SequenceMatcher(None, new_clean, existing).ratio() > threshold:
            return True
    return False

def generate_synthetic_mixed_data(output_file="synthetic_mixed_output.csv", total_target=350, batch_size=15):
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ValueError("❌ OPENROUTER_API_KEY environment variable is missing.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    unique_texts = set()
    records_saved = []

    # Recovery handling
    if os.path.exists(output_file):
        try:
            df_existing = pd.read_csv(output_file)
            records_saved = df_existing.to_dict(orient="records")
            unique_texts = set(df_existing['text'].str.strip().str.lower().tolist())
            logger.info(f"🔄 Checkpoint found. Recovered {len(records_saved)} unique synthetic rows.")
        except Exception:
            pass

    system_prompt = (
        "You are an expert economic data generator. Output raw data rows using pipe delimitation (|).\n"
        "Do not write any markdown, intro text, explanation, or wrapping code blocks. Start outputting data rows immediately.\n\n"
        "MANDATORY ROW FORMAT:\n"
        "Sentence_Text | Positive_Aspect_Name | Negative_Aspect_Name\n\n"
        "RULES:\n"
        "1. Every sentence must contain exactly ONE clear macroeconomic aspect that is Positive, and exactly ONE distinct macro aspect that is Negative.\n"
        "2. Choose aspect names strictly from this list: Monetary_Financial, Inflation_Prices, Real_Economic_Activity, Labor_Consumption, Fiscal_Government, External_Sector\n"
        "3. Ensure the sentence language is highly realistic, variable, and professional."
    )

    pbar = tqdm(total=total_target, initial=len(records_saved), desc="Progress")

    while len(records_saved) < total_target:
        user_prompt = f"Generate exactly {batch_size} unique rows using the pipe (|) format. Cycle through varied aspect pairs randomly."
        
        try:
            response = client.chat.completions.create(
                model="deepseek/deepseek-v4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.85,
                presence_penalty=0.6,
                frequency_penalty=0.6,
                max_tokens=2000
            )
            raw_content = response.choices[0].message.content
        except Exception as e:
            logger.warning(f"⚠️ API Connection issue: {str(e)}. Retrying...")
            time.sleep(3)
            continue

        if raw_content:
            # Clean up potential thinking blocks or code fence leftovers safely
            clean_text = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
            clean_text = clean_text.replace("```", "").strip()
            
            new_additions = 0
            for line in clean_text.splitlines():
                if "|" not in line or line.count("|") < 2:
                    continue
                
                try:
                    parts = [p.strip() for p in line.split("|")]
                    sentence_text = parts[0]
                    pos_aspect = parts[1]
                    neg_aspect = parts[2]
                    
                    # Basic sanitization validation guards
                    if len(sentence_text.split()) < 5 or pos_aspect == neg_aspect:
                        continue
                        
                    if sentence_text.lower() not in unique_texts and not is_too_similar(sentence_text, unique_texts):
                        unique_texts.add(sentence_text.lower())
                        
                        # Build the default full structure seamlessly out of the flat variables
                        record = {
                            "title": "Synthetic_Engine_Fast",
                            "text": sentence_text,
                            "llm_justification": "Programmatic flat layout split optimization.",
                            "Monetary_Financial": "POSITIVE" if pos_aspect == "Monetary_Financial" else ("NEGATIVE" if neg_aspect == "Monetary_Financial" else "ABSTAIN"),
                            "Inflation_Prices": "POSITIVE" if pos_aspect == "Inflation_Prices" else ("NEGATIVE" if neg_aspect == "Inflation_Prices" else "ABSTAIN"),
                            "Real_Economic_Activity": "POSITIVE" if pos_aspect == "Real_Economic_Activity" else ("NEGATIVE" if neg_aspect == "Real_Economic_Activity" else "ABSTAIN"),
                            "Labor_Consumption": "POSITIVE" if pos_aspect == "Labor_Consumption" else ("NEGATIVE" if neg_aspect == "Labor_Consumption" else "ABSTAIN"),
                            "Fiscal_Government": "POSITIVE" if pos_aspect == "Fiscal_Government" else ("NEGATIVE" if neg_aspect == "Fiscal_Government" else "ABSTAIN"),
                            "External_Sector": "POSITIVE" if pos_aspect == "External_Sector" else ("NEGATIVE" if neg_aspect == "External_Sector" else "ABSTAIN")
                        }
                        records_saved.append(record)
                        new_additions += 1
                        pbar.update(1)
                        
                        if len(records_saved) >= total_target:
                            break
                except Exception:
                    continue
            
            if new_additions > 0:
                pd.DataFrame(records_saved).to_csv(output_file, index=False, encoding='utf-8')
            else:
                # If it's hitting duplicate blockades, relax the penalty variables on the next API call loop
                time.sleep(1)

    pbar.close()

if __name__ == "__main__":
    output_csv = "synthetic_mixed_sentences_fast.csv"
    final_output_ready = "finbert_absa_synthetic_training_ready.csv"
    
    generate_synthetic_mixed_data(output_file=output_csv, total_target=350, batch_size=15)
    
    # Fast map strings cleanly back into the training domain integer layout
    if os.path.exists(output_csv):
        df_sync = pd.read_csv(output_csv)
        absa_mapping = {"ABSTAIN": 0.0, "POSITIVE": 1.0, "NEGATIVE": 2.0, "NEUTRAL": 3.0}
        target_cols = ["Monetary_Financial", "Inflation_Prices", "Real_Economic_Activity",
                       "Labor_Consumption", "Fiscal_Government", "External_Sector"]
        
        for col in target_cols:
            df_sync[col] = df_sync[col].map(absa_mapping).fillna(0.0)
            
        df_sync.to_csv(final_output_ready, index=False, encoding='utf-8')
        print(f"✅ Fast Processing Success! File ready for fine-tuning: {final_output_ready}")