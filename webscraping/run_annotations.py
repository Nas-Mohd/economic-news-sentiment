import os
import logging
import pandas as pd
import ollama
from pydantic import BaseModel, Field
from typing import Literal
from tqdm import tqdm

# =====================================================================
# 1. Pipeline Telemetry Configuration (Logging)
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FilBERT_Annotator")

# =====================================================================
# 2. Rigid 6-Aspect Macroeconomic Output Schema
# =====================================================================
class FinancialTextAnnotation(BaseModel):
    # Completely avoids column collisions with your CSV metadata
    llm_justification: str = Field(description="A 1-sentence economic reasoning.")
    Monetary_Financial: Literal["PRESENT", "ABSTAIN"]
    Inflation_Prices: Literal["PRESENT", "ABSTAIN"]
    Real_Economic_Activity: Literal["PRESENT", "ABSTAIN"]
    Labor_Consumption: Literal["PRESENT", "ABSTAIN"]
    Fiscal_Government: Literal["PRESENT", "ABSTAIN"]
    External_Sector: Literal["PRESENT", "ABSTAIN"]

# =====================================================================
# 3. State Management & Batch Processing Loop
# =====================================================================
def annotate_dataset_locally(df, text_column, checkpoint_file="annotations_checkpoint.csv", model_name="wiro-finance"):
    
    # System Instruction targeting concurrent multi-label activation
    system_prompt = (
        "You are an expert macroeconomic annotator. Evaluate the provided text sentence against "
        "6 separate economic domains. A sentence can contain MULTIPLE aspects simultaneously. "
        "Assign 'PRESENT' to EVERY aspect that has clear direct or indirect textual evidence, "
        "and 'ABSTAIN' only if the domain is completely unmentioned.\n\n"
        "CRITICAL OUTPUT REQUIREMENT:\n"
        "For the 'llm_justification' field, you MUST write a complete, detailed 1-sentence macroeconomic "
        "justification explaining why the target labels were selected. Do NOT output single words, "
        "and do NOT leave it empty.\n\n"
        "Strict Domain Boundaries:\n"
        "- Monetary_Financial: Central bank actions, interest rates, monetary policy, banking sector, credit liquidity, currency movements, exchange rates, and bond yields.\n"
        "- Inflation_Prices: CPI, PPI, wholesale/retail price changes, inflation expectations, wage inflation, or direct asset/commodity price spikes (e.g., oil/gas price volatility).\n"
        "- Real_Economic_Activity: GDP growth, industrial production, manufacturing indices (PMI), corporate revenue/profits, infrastructure construction, investments, and overall economic health/resilience.\n"
        "- Labor_Consumption: Employment, unemployment rates, layoffs, labor force participation, retail sales, consumer confidence, household spending, and domestic market demand.\n"
        "- Fiscal_Government: National budgets, sovereign debt, government spending packages, taxation policies, tariffs, subsidies, and state-level regulatory interventions.\n"
        "- External_Sector: Cross-border trade, imports/exports, international shipping, supply chain logistics, trade balances, foreign direct investment (FDI), and global economic agreements (e.g., BRICS summits).\n\n"
        "Respond strictly using the required JSON schema structure."
    )

    start_idx = 0

    # Crash-recovery state check
    if os.path.exists(checkpoint_file):
        try:
            df_checkpoint = pd.read_csv(checkpoint_file)
            start_idx = len(df_checkpoint)
            logger.info(f"🔄 Found crash checkpoint. Resuming dataset execution from index row: {start_idx}")
        except Exception as e:
            logger.error(f"❌ Checkpoint file found but unreadable: {e}. Re-initializing clean tracking structure.")
            start_idx = 0
            
    # If starting fresh, initialize file with explicit structural headers
    if start_idx == 0:
        logger.info("🆕 Initializing clean data pipeline baseline on local drive.")
        empty_template = pd.DataFrame(columns=df.columns.tolist() + [
            "llm_justification", "Monetary_Financial", "Inflation_Prices", 
            "Real_Economic_Activity", "Labor_Consumption", "Fiscal_Government", "External_Sector"
        ])
        empty_template.to_csv(checkpoint_file, index=False, encoding='utf-8')

    logger.info(f"🚀 Launching local processing matrix via engine: '{model_name}'...")
    
    # Process slices sequentially via lightning-fast disk appends
    for idx in tqdm(range(start_idx, len(df)), desc="GTX 1650 Processing Pipeline", initial=start_idx, total=len(df)):
        row = df.iloc[idx]
        record = row.to_dict()
        
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Text to analyze: {row[text_column]}"}
                ],
                format=FinancialTextAnnotation.model_json_schema(),
                options={
                    "temperature": 0.0,   # Fully deterministic extraction
                    "seed": 42,
                    "num_ctx": 1024,      # Safe context floor constraint
                    "num_predict": 256    # Constrain generation length to preserve system RAM speed
                }
            )
            
            # De-serialize validated structural JSON response
            raw_content = response['message']['content']
            structured_data = FinancialTextAnnotation.model_validate_json(raw_content)
            record.update(structured_data.model_dump())
            
        except Exception as e:
            logger.warning(f"⚠️ Index Row {idx}: Encountered extraction exception. Falling back to ABSTAIN markers. Error: {e}")
            record.update({
                "llm_justification": f"Skipped due to pipeline exception: {str(e)}",
                "Monetary_Financial": "ABSTAIN", "Inflation_Prices": "ABSTAIN",
                "Real_Economic_Activity": "ABSTAIN", "Labor_Consumption": "ABSTAIN",
                "Fiscal_Government": "ABSTAIN", "External_Sector": "ABSTAIN"
            })
            
        # Stream processed row straight to storage disk in absolute O(1) time
        pd.DataFrame([record]).to_csv(checkpoint_file, mode='a', header=False, index=False, encoding='utf-8')

# =====================================================================
# 4. Core Execution Script
# =====================================================================
if __name__ == "__main__":
    input_file = "staged_economic_sentences.csv" 
    text_column_name = "text"   
    checkpoint_file = "annotations_checkpoint.csv"
    output_file = "finbert_training_ready.csv"

    logger.info("⚡ System Init: Validating workspace conditions...")

    try:
        # Load raw news data
        df_raw = pd.read_csv(input_file)
        logger.info(f"📊 Original dataset parsed successfully. Found total target records: {len(df_raw)}")
        
        # Run local data extraction engine
        annotate_dataset_locally(df_raw, text_column=text_column_name, checkpoint_file=checkpoint_file)
        
        # Read the raw string annotations collected inside the checkpoint file
        logger.info("Reading streaming checkpoint file into memory for vector mapping...")
        df_annotated = pd.read_csv(checkpoint_file)
        
        logger.info("Mapping extracted text tags into normalized binary training arrays...")
        target_cols = [
            "Monetary_Financial", "Inflation_Prices", "Real_Economic_Activity",
            "Labor_Consumption", "Fiscal_Government", "External_Sector"
        ]
        
        # Convert literal strings directly into FilBERT binary floats
        for col in target_cols:
            df_annotated[col] = df_annotated[col].map({"PRESENT": 1.0, "ABSTAIN": 0.0}).fillna(0.0)
            
        # Deliver optimized production dataset output
        df_annotated.to_csv(output_file, index=False)
        logger.info(f"🎉 Complete! Production training-ready dataset committed cleanly to: {output_file}")
        
        # Clean up temporary checkpoint files upon a successful full completion run
        if os.path.exists(checkpoint_file):
            #os.remove(checkpoint_file)
            logger.info("🧹 Verification check complete. Redundant runtime checkpoints have been removed.")
            
    except FileNotFoundError:
        logger.critical(f"❌ Execution Halted: Data source file '{input_file}' is missing from the directory environment.")
    except Exception as e:
        logger.critical(f"💥 Critical crash caught during data ingestion: {str(e)}", exc_info=True)