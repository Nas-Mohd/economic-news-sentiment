import pandas as pd
import nltk
import os
import sys

# Download the sentence tokenizer models quietly
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def split_csv_to_sentences(input_filename, output_filename, text_col="text"):
    print(f"📋 Loading raw dataset: {input_filename}")
    
    if not os.path.exists(input_filename):
        print(f"❌ Error: Could not find '{input_filename}' in this directory.")
        sys.exit(1)
        
    df = pd.read_csv(input_filename)
    
    # Drop rows that don't have text
    df = df.dropna(subset=[text_col])
    
    sentence_rows = []
    
    print("✂️ Splitting text columns into clean macroeconomic sentences...")
    for idx, row in df.iterrows():
        raw_text = str(row[text_col]).strip()
        
        # Clean up double line breaks often caused by web scrapers
        cleaned_text = " ".join(raw_text.split())
        
        # Use NLTK to intelligently tokenize text into individual sentences
        sentences = nltk.tokenize.sent_tokenize(cleaned_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out empty snippets or weird single characters
            if len(sentence) > 10:  
                # Duplicate the metadata row structure
                new_row = row.to_dict()
                # Replace the entire text wall with just this individual sentence
                new_row[text_col] = sentence
                sentence_rows.append(new_row)
                
    # Re-compile into a flat structural DataFrame
    df_sentences = pd.DataFrame(sentence_rows)
    
    df_sentences.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"🎉 Success! Expanded {len(df)} articles into {len(df_sentences)} independent sentences.")
    print(f"💾 Staging file saved to: {output_filename}\n")

if __name__ == "__main__":
    # Configure your workspace targets
    RAW_INPUT = "malaysian_economic_news_1000.csv"
    STAGED_OUTPUT = "staged_economic_sentences.csv"
    TEXT_COLUMN = "text"
    
    split_csv_to_sentences(RAW_INPUT, STAGED_OUTPUT, text_col=TEXT_COLUMN)