from huggingface_hub import HfApi
api = HfApi()

# Replace with your HF username and chosen model name
api.upload_folder(
    folder_path="models/final1_finbert_aspect_sentiment",
    repo_id="dummfak/finbert-macroeconomic-absa",
    repo_type="model"
)