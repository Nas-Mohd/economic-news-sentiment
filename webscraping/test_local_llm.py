import pandas as pd
import ollama
from pydantic import BaseModel, Field
from typing import Literal

# 1. Define the structural output schema
class FinancialTextAnnotation(BaseModel):
    reasoning: str = Field(description="1-sentence economic justification for the assigned labels.")
    Monetary_Financial: Literal["PRESENT", "ABSTAIN"]
    Inflation_Prices: Literal["PRESENT", "ABSTAIN"]
    Real_Economic_Activity: Literal["PRESENT", "ABSTAIN"]
    Labor_Consumption: Literal["PRESENT", "ABSTAIN"]
    Fiscal_Government: Literal["PRESENT", "ABSTAIN"]
    External_Sector: Literal["PRESENT", "ABSTAIN"]

# 2. Create the mock single-row DataFrame using our benchmark sentence
mock_data = {
    "text": [
        "Kuala Lumpur: Retail sales inside the domestic economy surged month-on-month, "
        "though escalating regional logistics bottlenecks and shipping container shortages "
        "continue to drive up wholesale import pricing."
    ]
}
df = pd.DataFrame(mock_data)

print("🚀 Booting optimized wiro-finance configuration for a single-row test...")

# 3. Test a single deterministic pass with reinforced instructions
try:
    response = ollama.chat(
        model="wiro-finance",
        messages=[
            {
                "role": "system", 
                "content": (
                    "You are an expert macroeconomic annotator. Evaluate the provided text sentence against "
                    "6 separate economic domains. A sentence can contain MULTIPLE aspects simultaneously. "
                    "Assign 'PRESENT' to EVERY aspect that has clear direct or indirect textual evidence, "
                    "and 'ABSTAIN' only if the domain is completely unmentioned.\n\n"
                    "Domain Guidelines:\n"
                    "- Labor_Consumption: Mark PRESENT for retail sales, consumer spending, and demand.\n"
                    "- External_Sector: Mark PRESENT for imports, international logistics, shipping, and trade."
                )
            },
            {"role": "user", "content": f"Text to analyze: {df.loc[0, 'text']}"}
        ],
        format=FinancialTextAnnotation.model_json_schema(),
        options={
            "temperature": 0.0,
            "seed": 42,
            "num_ctx": 1024,      # Expanded memory boundary for multi-label attention
            "num_predict": 128
        }
    )

    # 4. Parse and structure the output
    raw_json = response['message']['content']
    structured_data = FinancialTextAnnotation.model_validate_json(raw_json)
    
    # Merge back into our test dataframe row
    result_dict = df.loc[0].to_dict()
    result_dict.update(structured_data.model_dump())
    df_result = pd.DataFrame([result_dict])
    
    print("\n--- [STEP 1: Raw Structured Extraction Output] ---")
    print(df_result.to_string(index=False))

    # 5. Map literals to machine learning binary floats
    target_cols = [
        "Monetary_Financial", "Inflation_Prices", "Real_Economic_Activity",
        "Labor_Consumption", "Fiscal_Government", "External_Sector"
    ]
    for col in target_cols:
        df_result[col] = df_result[col].map({"PRESENT": 1.0, "ABSTAIN": 0.0})

    print("\n--- [STEP 2: Final Binary Mapped Output for FilBERT] ---")
    print(df_result.to_string(index=False))
    print("\n✅ Test completed successfully!")

except Exception as e:
    print(f"\n❌ Test failed: {str(e)}")