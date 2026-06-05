import os
import logging
import json
import time
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# =====================================================================
# 1. Pipeline Telemetry Configuration
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline_absa.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FinBERT_ABSA_Annotator")

# =====================================================================
# 2. Rigid Pydantic Schema Matrix
# =====================================================================
SentimentState = Literal["POSITIVE", "NEGATIVE", "NEUTRAL", "ABSTAIN"]

class FinancialAspectSentimentAnnotation(BaseModel):
    llm_justification: str = Field(description="A concise 1-sentence macro justification for the ratings under 30 words.")
    Monetary_Financial: SentimentState
    Inflation_Prices: SentimentState
    Real_Economic_Activity: SentimentState
    Labor_Consumption: SentimentState
    Fiscal_Government: SentimentState
    External_Sector: SentimentState

# =====================================================================
# 3. Processing Core Engine
# =====================================================================
def annotate_absa_dataset(df, checkpoint_file="absa_new_checkpoint.csv", model_name="meta-llama/llama-3-8b-instruct"):
    
    if not os.environ.get("OPENROUTER_API_KEY"):
        logger.critical("❌ Missing Environment Variable: Please set your 'OPENROUTER_API_KEY' before running.")
        raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY")
    )

    system_prompt = (
        "You are an expert macroeconomic annotator performing Aspect-Based Sentiment Analysis (ABSA).\n"
        "Analyze the provided Target Sentence within the context of its Article Title to evaluate across 6 domains.\n\n"
        "For EACH domain, assign exactly one of these tags:\n"
        "- 'ABSTAIN': The domain is completely unmentioned or irrelevant.\n"
        "- 'POSITIVE': National/systemic growth, structural stability, or improving macro conditions.\n"
        "- 'NEGATIVE': Systemic stress, economic decline, adverse volatility, or structural headwinds.\n"
        "- 'NEUTRAL': The aspect is present but presents mixed data, flat performance, or objective tracking without momentum.\n\n"
        
        "Strict Domain Boundaries:\n"
        "- Monetary_Financial: Central bank actions, interest rates, systemic banking stability/risks, liquidity cycles.\n"
        "- Inflation_Prices: CPI/PPI, inflation expectations, widespread asset/commodity price shocks.\n"
        "- Real_Economic_Activity: GDP, national manufacturing (PMI), industrial output, structural macro shifts.\n"
        "- Labor_Consumption: National employment rates, widespread layoffs, aggregate consumer spending metrics.\n"
        "- Fiscal_Government: National budgets, sovereign debt, federal tax changes, state spending policies.\n"
        "- External_Sector: Balance of trade, net imports/exports, foreign direct investment (FDI), exchange rate shocks.\n\n"
        
        "CRITICAL ANCHORING RULES:\n"
        "1. Do NOT classify individual commercial bank product features, corporate marketing promotions, or standard local retail terms "
        "as systemic 'POSITIVE' or 'NEGATIVE' macro events. If a sentence just describes a specific bank's local product offer, use 'ABSTAIN'.\n"
        "2. For 'llm_justification', write a concise 1-sentence macro justification under 30 words.\n\n"
        "You MUST strictly output following this exact JSON structure sample:\n"
        "{\n"
        '  "llm_justification": "Put your reasoning here.",\n'
        '  "Monetary_Financial": "ABSTAIN",\n'
        '  "Inflation_Prices": "ABSTAIN",\n'
        '  "Real_Economic_Activity": "POSITIVE",\n'
        '  "Labor_Consumption": "ABSTAIN",\n'
        '  "Fiscal_Government": "ABSTAIN",\n'
        '  "External_Sector": "ABSTAIN"\n'
        "}"
    )

    start_idx = 0

    # Resume checkpoint handling
    if os.path.exists(checkpoint_file):
        try:
            df_checkpoint = pd.read_csv(checkpoint_file)
            start_idx = len(df_checkpoint)
            logger.info(f"🔄 Checkpoint found. Resuming from index row: {start_idx}")
        except Exception:
            start_idx = 0

    if start_idx == 0:
        logger.info("🆕 Initializing clean output schema baseline.")
        # Setup headers using only clean layout targets
        empty_template = pd.DataFrame(columns=["title", "text", "llm_justification", 
                                               "Monetary_Financial", "Inflation_Prices", "Real_Economic_Activity", 
                                               "Labor_Consumption", "Fiscal_Government", "External_Sector"])
        empty_template.to_csv(checkpoint_file, index=False, encoding='utf-8')

    logger.info(f"🚀 Streaming to OpenRouter -> '{model_name}'...")

    for idx in tqdm(range(start_idx, len(df)), desc="ABSA Pipeline", initial=start_idx, total=len(df)):
        row = df.iloc[idx]
        
        # Explicitly build record to save only title and text fields
        record = {
            "title": row["title"],
            "text": row["text"]
        }
        
        # Contextual prompt engineering payload
        user_content = f"Article Title: {row['title']}\nTarget Sentence to Analyze: {row['text']}"
        
        raw_content = None
        retries = 4
        delay = 2.0
        
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    # Enforce rigid structural validation schemas at transmission level
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "MacroFinancialABSA",
                            "strict": True,
                            "schema": FinancialAspectSentimentAnnotation.model_json_schema()
                        }
                    },
                    extra_headers={
                        "HTTP-Referer": "https://github.com/finbert-pipeline",
                        "X-Title": "FinBERT ABSA Engine"
                    },
                    temperature=0.0,
                    seed=42,
                    max_tokens=512
                )
                raw_content = response.choices[0].message.content
                break
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                break

        if raw_content:
            try:
                parsed_json = json.loads(raw_content)
                if isinstance(parsed_json, list) and len(parsed_json) > 0:
                    parsed_json = parsed_json[0]
                
                record.update(parsed_json)
            except Exception as parse_error:
                record.update({
                    "llm_justification": f"Parsing Error fallback: {str(parse_error)}",
                    "Monetary_Financial": "ABSTAIN", "Inflation_Prices": "ABSTAIN",
                    "Real_Economic_Activity": "ABSTAIN", "Labor_Consumption": "ABSTAIN",
                    "Fiscal_Government": "ABSTAIN", "External_Sector": "ABSTAIN"
                })
        else:
            record.update({
                "llm_justification": "API Network connection timeout fallback exception.",
                "Monetary_Financial": "ABSTAIN", "Inflation_Prices": "ABSTAIN",
                "Real_Economic_Activity": "ABSTAIN", "Labor_Consumption": "ABSTAIN",
                "Fiscal_Government": "ABSTAIN", "External_Sector": "ABSTAIN"
            })
            
        # 💡 FIXED LINE: Forces the dictionary to export values in the exact column sequence
        export_columns = [
            "title", "text", "llm_justification", "Monetary_Financial", 
            "Inflation_Prices", "Real_Economic_Activity", "Labor_Consumption", 
            "Fiscal_Government", "External_Sector"
        ]
        
        # Convert to DataFrame and enforce strict column slice ordering
        df_row = pd.DataFrame([record])[export_columns]
        df_row.to_csv(checkpoint_file, mode='a', header=False, index=False, encoding='utf-8')
        time.sleep(0.1)

# =====================================================================
# 4. Ingestion and Execution Execution Block
# =====================================================================
if __name__ == "__main__":
    input_file = "staged_new_economic_sentences.csv" 
    checkpoint_file = "absa_new_checkpoint.csv"
    output_file = "finbert_absa_training_ready_1.csv"

    try:
        # Load raw dataset
        df_raw = pd.read_csv(input_file)
        
        # Isolate baseline columns to ensure no downstream indexing pollution
        df_cleaned = df_raw[["title", "text"]].copy()
        
        logger.info(f"📊 Dataset sanitized successfully. Rows to process: {len(df_cleaned)}")
        
        # Fire pipeline loop
        annotate_absa_dataset(df_cleaned, checkpoint_file=checkpoint_file)
        
        # Read final generated checkpoints out of workspace storage
        df_annotated = pd.read_csv(checkpoint_file)
        
        # Remap string labels into clean machine learning target category integers
        absa_mapping = {"ABSTAIN": 0.0, "POSITIVE": 1.0, "NEGATIVE": 2.0, "NEUTRAL": 3.0}
        target_cols = ["Monetary_Financial", "Inflation_Prices", "Real_Economic_Activity",
                       "Labor_Consumption", "Fiscal_Government", "External_Sector"]
        
        for col in target_cols:
            df_annotated[col] = df_annotated[col].map(absa_mapping).fillna(0.0)
            
        df_annotated.to_csv(output_file, index=False)
        logger.info(f"🎉 Complete! Processed training ready file generated: {output_file}")
        
        if os.path.exists(checkpoint_file):
            logger.info(f"🎉 Complete! CHECKKKd: {checkpoint_file}")
            #os.remove(checkpoint_file)
            
    except Exception as e:
        logger.critical(f"💥 Runtime Error: {str(e)}", exc_info=True)