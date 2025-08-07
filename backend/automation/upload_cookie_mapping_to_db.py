import pandas as pd
import os
from sqlalchemy import create_engine
from gspread import service_account
import gspread_dataframe as gd
from typing import Any

MAX_SHARE_COOKIES = 5  # Set how many 'share-from' columns you want to support

def fetch_and_upload_cookie_mapping() -> None:
    # Authenticate with Google Sheets
    gc = service_account(filename="service_account.json")
    spreadsheet = gc.open("cookie_mapping")
    print("🔍 Opening 'cookie_mapping' Google Sheet...")

    # Read tabs
    print("📥 Reading tabs from Google Sheet...")
    transitions_ws = spreadsheet.worksheet("cookie_transitions")
    active_ws = spreadsheet.worksheet("active_cookies")

    transitions_df = gd.get_as_dataframe(transitions_ws).dropna(how='all')
    active_df = gd.get_as_dataframe(active_ws).dropna(how='all')

    # Clean column names
    transitions_df.columns = [col.strip() for col in transitions_df.columns]
    active_df.columns = [col.strip() for col in active_df.columns]

    # Flatten 'Takes Share from (Other Cookies)' and 'Share % Taken'
    def flatten_share_columns(row):
        takes_from = str(row.get("Takes Share from (Other Cookies)", "")).split(",")
        share_pct = str(row.get("Share % Taken", "")).replace("%", "").split(",")

        takes_from = [t.strip() for t in takes_from if t.strip()]
        share_pct = [p.strip() for p in share_pct if p.strip()]

        for i in range(MAX_SHARE_COOKIES):
            row[f"ShareFrom_{i+1}"] = takes_from[i] if i < len(takes_from) else None
            row[f"SharePct_{i+1}"] = float(share_pct[i]) if i < len(share_pct) else None

        return row

    transitions_df = transitions_df.apply(flatten_share_columns, axis=1)
    transitions_df.drop(columns=["Takes Share from (Other Cookies)", "Share % Taken"], inplace=True)

    # Upload to Render DB
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL not set in environment variables")

    engine = create_engine(db_url)
    transitions_df.to_sql("cookie_transitions", con=engine, if_exists="replace", index=False)
    active_df.to_sql("active_cookies", con=engine, if_exists="replace", index=False)

    print("✅ Uploaded cookie_transitions and active_cookies to Render DB")

if __name__ == "__main__":
    fetch_and_upload_cookie_mapping()
