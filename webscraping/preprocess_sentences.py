import pandas as pd
import nltk
import os
import sys

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def split_csv_to_sentences(input_files, output_filename, text_col="text"):
    all_dfs = []

    for f in input_files:
        if not os.path.exists(f):
            print(f"⚠️  Skipping '{f}' — file not found.")
            continue
        print(f"📋 Loading: {f}")
        df = pd.read_csv(f)
        all_dfs.append(df)

    if not all_dfs:
        print("❌ No valid input files found. Exiting.")
        sys.exit(1)

    # Combine all files
    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n📦 Combined: {len(combined)} total rows across {len(all_dfs)} files")

    # Deduplicate on URL first (same article from different files), then on title
    before = len(combined)
    combined = combined.drop_duplicates(subset=['url'], keep='first')
    combined = combined.drop_duplicates(subset=['title'], keep='first')
    print(f"🧹 Removed {before - len(combined)} duplicate articles → {len(combined)} unique articles")

    # Drop rows with no text
    combined = combined.dropna(subset=[text_col])

    sentence_rows = []
    print("\n✂️  Splitting text into sentences...")

    for idx, row in combined.iterrows():
        raw_text = str(row[text_col]).strip()
        cleaned_text = " ".join(raw_text.split())
        sentences = nltk.tokenize.sent_tokenize(cleaned_text)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                new_row = row.to_dict()
                new_row[text_col] = sentence
                sentence_rows.append(new_row)

    df_sentences = pd.DataFrame(sentence_rows)

    # Deduplicate on sentence level too (same sentence appearing in multiple articles)
    before_sent = len(df_sentences)
    df_sentences = df_sentences.drop_duplicates(subset=[text_col], keep='first')
    print(f"🧹 Removed {before_sent - len(df_sentences)} duplicate sentences")

    df_sentences.to_csv(output_filename, index=False, encoding='utf-8')
    print(f"\n🎉 Expanded {len(combined)} articles into {len(df_sentences)} unique sentences.")
    print(f"💾 Saved to: {output_filename}")


if __name__ == "__main__":
    INPUT_FILES = [
        "malaysian_economic_new_1_year_1000.csv",
        "malaysian_economic_news_1000_1.csv",
        "economic_news_1000.csv",
    ]
    OUTPUT_FILE = "staged_new_economic_sentences.csv"
    TEXT_COLUMN = "text"

    split_csv_to_sentences(INPUT_FILES, OUTPUT_FILE, text_col=TEXT_COLUMN)