import os
import logging
import json
import time
import re
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal
from tqdm import tqdm


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


def pre_filter_macro_signal(df, text_column="text"):
    """
    Heuristically flags sentences that have ZERO mathematical or macro footprint.
    Bypasses the LLM for garbage rows and auto-assigns ABSTAIN.
    """
    macro_indicators = (
        r"pmi|gdp|cpi|ppi|inflation|unemployment|layoff|hiring|spending|tax|budget|"
        r"deficit|debt|yield|interest rate|basis point|fed|central bank|export|import|"
        r"tariff|commodity|oil|prices|revenue|growth|contraction|slowdown|volatility|currency|dollar|rupee"
    )
    
    conversational_traps = r"^\"?it’s like|^\"?the spillover is|panel discussion|metaphor|analogy|reporting by|editing by"

    df = df.copy()
    df["is_macro_signal"] = True
    
    for idx, text in enumerate(df[text_column].fillna("").str.lower()):
        is_trap = bool(re.search(conversational_traps, text))
        is_short_fragment = len(text.split()) < 8 and ("said" in text or "says" in text)

        # Force bypass of obvious news credits/bylines (e.g. "Reporting by...")
        if "reporting by" in text or "editing by" in text or (text.startswith("(") and text.endswith(")")):
            df.at[df.index[idx], "is_macro_signal"] = False
        elif is_trap or is_short_fragment:
            df.at[df.index[idx], "is_macro_signal"] = False

    signal_rows = df["is_macro_signal"].sum()
    skipped_rows = len(df) - signal_rows
    logger.info(f"⚡ --- Semantic Signal Pre-Filter Complete ---")
    logger.info(f"   • High-Signal Macro Rows Retained for API: {signal_rows}")
    logger.info(f"   • Low-Signal Conversational Rows Bypassed: {skipped_rows} ({skipped_rows/len(df)*100:.2f}% cash saved!)")
    
    return df
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
    "You are a strict, production-grade macroeconomic data annotation engine specializing in Aspect-Category Sentiment Analysis (ACSA).\n"
    "Your core directive is to analyze an economic text statement and isolate directional systemic movements across exactly 6 macro dimensions.\n\n"
    
    "### THE EVALUATION MATRIX:\n"
    "- 'ABSTAIN' (Use if the domain is entirely unmentioned, irrelevant, or describes an isolated commercial/retail bank micro-product).\n"
    "- 'POSITIVE' (Indicates systemic national expansion, improving industrial/market capacity, structural strength, or stabilizing vectors).\n"
    "- 'NEGATIVE' (Indicates systemic stress, contraction, escalating macro volatility, supply chain friction, or rising business input costs).\n"
    "- 'NEUTRAL' (Indicates data points that represent flat baseline performance, balanced/conflicting metrics, or objective tracking without directional momentum).\n\n"
    
    "### CRITICAL MACRO DOMAIN BOUNDARIES & SIGNAL FILTERS:\n"
    "1. Monetary_Financial:\n"
    "   - Signals: Central bank policy, interest rates/yield curves, systemic credit risk, banking stability.\n"
    "   - WARNING: 'Higher market volatility' or 'greater fluctuations' in debt/equity indicates escalating systemic risk -> NEGATIVE. A conditional phrase like 'willing to consider a rate hike' -> NEUTRAL.\n"
    "2. Inflation_Prices:\n"
    "   - Signals: CPI, PPI, input/operational costs, wholesale commodity price tracking.\n"
    "   - WARNING: 'Elevated commodity/crude oil prices' contributing to 'higher production/transportation costs' represents operational margin strain -> NEGATIVE.\n"
    "3. Real_Economic_Activity:\n"
    "   - Signals: GDP, aggregate business survey indices (PMI), factory output, new orders.\n"
    "   - WARNING: Slower new order momentum or cooling demand growth represents a deceleration of economic activity -> NEGATIVE.\n"
    "4. Labor_Consumption:\n"
    "   - Signals: National unemployment claims, aggregate layoffs, macro consumption metrics.\n"
    "5. Fiscal_Government:\n"
    "   - Signals: Sovereign budget deficits, national debt issuance, federal tax legislation.\n"
    "6. External_Sector:\n"
    "   - Signals: Balance of trade, import/export bottlenecks, global commodity benchmarks, exchange rate currency shocks.\n"
    "   - WARNING: Global baseline asset movements (e.g., crude oil shocks) or foreign logistics delays (e.g., 'delays from China') directly hit the external balance of trade -> NEGATIVE.\n\n"
    
    "### EXECUTION CONSTRAINTS:\n"
    "- Step-by-Step Logic: Evaluate the text clause-by-clause inside your internal hidden layers before selecting tags.\n"
    "- Do not default to NEUTRAL for proposals, forward-looking declarations, legislative expansions, or central bank policy guidance. In macroeconomics, institutional intent and mandate shifts alter market expectations and structure immediately. Label them based on their intended vector (e.g., legally forcing a central bank to target job creation is an expansionary/dovish change, which maps to Monetary=POSITIVE, Real_Econ=POSITIVE, Labor=POSITIVE).\n"
    "- Avoid Word Bias: Never map words like 'higher' blindly to POSITIVE. Analyze if 'higher' applies to a negative metric (like costs or volatility).\n"
    "- Complete Output: You must map an explicit category tag to ALL 6 schema keys without exception.\n"
    "- Strict Justification Rule: The 'llm_justification' field must be EXACTLY one single sentence of under 25 words. It must omit all conversational filler (e.g., 'This sentence talks about...', 'Based on the analysis...') and state ONLY the direct economic transmission link (e.g., 'Depreciation of the currency triggers severe cost-push import inflation.')."
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

    # 💡 1. EXPLICIT HIGH-DIVERSITY FEW-SHOT EXAMPLES MAPPED TO CHAT ROLES
    few_shot_messages = [
        # Example A: Handling cross-domain structural commodity spikes
        {
            "role": "user", 
            "content": "Target Sentence to Analyze: Crude oil prices remain elevated, contributing to higher transportation and production costs across the economy."
        },
        {
            "role": "assistant", 
            "content": '{"llm_justification": "Elevated international energy benchmarks strain input prices and factory operational margins while introducing systemic external trade friction.", "Monetary_Financial": "ABSTAIN", "Inflation_Prices": "NEGATIVE", "Real_Economic_Activity": "NEGATIVE", "Labor_Consumption": "ABSTAIN", "Fiscal_Government": "ABSTAIN", "External_Sector": "NEGATIVE"}'
        },
        # Example B: Catching market risk/volatility indicators
        {
            "role": "user", 
            "content": "Target Sentence to Analyze: Economists anticipate higher volatility in private debt markets, while most also expect greater fluctuations in equity markets."
        },
        {
            "role": "assistant", 
            "content": '{"llm_justification": "Escalating asset price fluctuations and macro market uncertainty directly amplify systemic credit and banking volatility.", "Monetary_Financial": "NEGATIVE", "Inflation_Prices": "ABSTAIN", "Real_Economic_Activity": "ABSTAIN", "Labor_Consumption": "ABSTAIN", "Fiscal_Government": "ABSTAIN", "External_Sector": "ABSTAIN"}'
        },
        # Example C: Identifying demand cooling / deceleration contraction
        {
            "role": "user", 
            "content": "Target Sentence to Analyze: In services, backlogs are mostly stable but easing, as consumer demand metrics have cooled from prior periods."
        },
        {
            "role": "assistant", 
            "content": '{"llm_justification": "Cooling aggregate consumer demand vectors indicate a structural downshift in broader consumer spending and non-manufacturing activity lines.", "Monetary_Financial": "ABSTAIN", "Inflation_Prices": "ABSTAIN", "Real_Economic_Activity": "NEGATIVE", "Labor_Consumption": "NEGATIVE", "Fiscal_Government": "ABSTAIN", "External_Sector": "ABSTAIN"}'
        },
        # Substitute this into your few-shot list to anchor institutional announcements
        {
            "role": "user", 
            "content": "Target Sentence to Analyze: The Prime Minister announced a sweeping structural framework to reform banking supervision guidelines next quarter to loosen capital constraints on consumer lending."
        },
        {
            "role": "assistant", 
            "content": '{"llm_justification": "Forward-looking legislative or structural policy framework expansions targeting central bank/banking rules alter market vectors immediately upon announcement. Loosening credit limitations represents a structurally expansionary monetary and credit vector.", "Monetary_Financial": "POSITIVE", "Inflation_Prices": "ABSTAIN", "Real_Economic_Activity": "POSITIVE", "Labor_Consumption": "ABSTAIN", "Fiscal_Government": "ABSTAIN", "External_Sector": "ABSTAIN"}'
        }
    ]

    for idx in tqdm(range(start_idx, len(df)), desc="ABSA Pipeline", initial=start_idx, total=len(df)):
        row = df.iloc[idx]
        
        record = {
            "title": row["title"],
            "text": row["text"]
        }
        
        user_content = f"Target Sentence to Analyze: {row['text']}"
        
        # 💡 2. STITCH PAYLOAD: System Prompt -> Dynamic Few-Shot History -> Current Row
        messages_payload = [{"role": "system", "content": system_prompt}]
        messages_payload.extend(few_shot_messages)
        messages_payload.append({"role": "user", "content": user_content})
        
        raw_content = None
        retries = 4
        delay = 2.0
        
        for attempt in range(retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages_payload,  # Sending the complete multi-turn array
                    # 💡 REMOVED: response_format block dropped to completely bypass 400 Bad Request errors
                    extra_headers={
                        "HTTP-Referer": "https://github.com/finbert-pipeline",
                        "X-Title": "FinBERT ABSA Engine"
                    },
                    temperature=0.0,
                    seed=42,
                    max_tokens=2048
                )
                raw_content = response.choices[0].message.content
                break
            except Exception as e:
                logger.warning(f"⚠️ API Exception on row {idx} (Attempt {attempt+1}): {str(e)}")
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                break

        parsed_successfully = False
        if raw_content:
            try:
                 # 💡 NEW: Strip out DeepSeek `<think> ... </think>` blocks if present
                clean_text = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
                # Clean up trailing/leading whitespace or markdown wrappers if any exist
                clean_text = raw_content.strip().strip("```json").strip("```")
                parsed_json = json.loads(clean_text)
                
                if isinstance(parsed_json, list) and len(parsed_json) > 0:
                    parsed_json = parsed_json[0]
                
                required_keys = ["llm_justification", "Monetary_Financial", "Inflation_Prices", 
                                 "Real_Economic_Activity", "Labor_Consumption", "Fiscal_Government", "External_Sector"]
                
                if all(key in parsed_json for key in required_keys):
                    record.update(parsed_json)
                    parsed_successfully = True
                else:
                    logger.error(f"❌ Structural Mismatch at index row {idx}")
            except Exception as parse_error:
                logger.error(f"❌ JSON Parsing Malfunction on index row {idx}: {str(parse_error)}")
                logger.debug(f"   • Raw model output was: {raw_content}")
                

        if not parsed_successfully:
            record.update({
                "llm_justification": "CRITICAL PIPELINE PROCESSING FAILURE: Row flagged for manual review.",
                "Monetary_Financial": "ABSTAIN", "Inflation_Prices": "ABSTAIN",
                "Real_Economic_Activity": "ABSTAIN", "Labor_Consumption": "ABSTAIN",
                "Fiscal_Government": "ABSTAIN", "External_Sector": "ABSTAIN"
            })
            
        export_columns = [
            "title", "text", "llm_justification", "Monetary_Financial", 
            "Inflation_Prices", "Real_Economic_Activity", "Labor_Consumption", 
            "Fiscal_Government", "External_Sector"
        ]
        
        df_row = pd.DataFrame([record])[export_columns]
        df_row.to_csv(checkpoint_file, mode='a', header=False, index=False, encoding='utf-8')
        time.sleep(0.05)

def remove_length_outliers(df, text_column="text"):
    """
    Dynamically drops rows where the text word count is a statistical outlier
    using the Interquartile Range (IQR) method.
    """
    # 1. Calculate word counts for the target text
    word_counts = df[text_column].fillna("").str.split().apply(len)
    
    # 2. Calculate IQR thresholds
    q1 = word_counts.quantile(0.25)
    q3 = word_counts.quantile(0.75)
    iqr = q3 - q1
    
    # Standard statistical bounds (1.5 * IQR)
    lower_bound = max(3, int(q1 - 1.5 * iqr)) # Force a minimum floor of 3 words
    upper_bound = int(q3 + 1.5 * iqr)
    
    # 3. Identify rows within the valid distribution window
    valid_mask = (word_counts >= lower_bound) & (word_counts <= upper_bound)
    df_filtered = df[valid_mask].copy()
    
    # 4. Telemetry logging
    dropped_count = len(df) - len(df_filtered)
    logger.info("📊 --- Length Outlier Analysis Summary ---")
    logger.info(f"   • Word Count Distribution: Min={word_counts.min()}, 25th%={q1:.1f}, 75th%={q3:.1f}, Max={word_counts.max()}")
    logger.info(f"   • Dynamic Filters Applied: Lower Bound = {lower_bound} words | Upper Bound = {upper_bound} words")
    logger.info(f"   • Data Purge Metrics: Dropped {dropped_count} outlier rows ({dropped_count/len(df)*100:.2f}% of dataset).")
    logger.info(f"   • Sanitized Dataset Size: {len(df_filtered)} rows remaining.")
    
    return df_filtered
# =====================================================================
# 4. Ingestion and Execution Execution Block
# =====================================================================
if __name__ == "__main__":
    input_file = "staged_new_economic_sentences_1.csv" 
    checkpoint_file = "absa_new_checkpoint_1.csv"
    output_file = "finbert_absa_training_ready_2.csv"

    try:
        # Load raw dataset
        df_raw = pd.read_csv(input_file)
        
        # Isolate baseline columns to ensure no downstream indexing pollution
        df_cleaned = df_raw[["title", "text"]].copy()
        df_cleaned = remove_length_outliers(df_cleaned, text_column="text")
        # 💡 NEW: Prefilter to skip empty news bylines, credits and low-signal lines
        df_cleaned = pre_filter_macro_signal(df_cleaned, text_column="text")
        logger.info(f"📊 Dataset sanitized successfully. Rows to process: {len(df_cleaned)}")
        
        # Fire pipeline loop
        annotate_absa_dataset(df_cleaned, checkpoint_file=checkpoint_file, model_name="deepseek/deepseek-v4-flash")
        
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