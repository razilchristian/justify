import pandas as pd
import re
from html import unescape
import os

def clean_html(text):
    if not isinstance(text, str):
        return ""
    clean = re.sub(r"<.*?>", "", text)
    return unescape(clean).strip()

def load_and_prepare_justice_data():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, "justice.csv")
    df = pd.read_csv(file_path)
    df["clean_facts"] = df["facts"].apply(clean_html)
    return df

if __name__ == "__main__":
    df = load_and_prepare_justice_data()
    print(f"Total cases loaded: {len(df)}")
    print(df[["name", "docket", "clean_facts"]].head())
