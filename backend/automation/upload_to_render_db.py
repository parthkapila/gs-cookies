from sqlalchemy import create_engine
import pandas as pd
import os
from typing import Any

def format_troop_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all troop_ids are formatted as 5-digit strings with leading zeros"""
    df_copy = df.copy()
    
    # Format troop_id as 5-character string with leading zeros for numerical values
    df_copy['troop_id'] = df_copy['troop_id'].astype(str).str.strip()
    df_copy['troop_id'] = df_copy['troop_id'].apply(
        lambda x: f"{int(x):05d}" if x.strip().isdigit() else f"{x:>5}"
    )
    
    print(f"✅ Formatted troop_ids to 5-digit format")
    return df_copy

def format_su_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all SU_Num values are formatted as integers without decimal points"""
    df_copy = df.copy()
    
    if 'SU_Num' in df_copy.columns:
        # Convert SU_Num to integers to remove decimal points
        df_copy['SU_Num'] = df_copy['SU_Num'].astype(str).str.strip()
        # Remove any 'SU' prefix and convert to integer
        df_copy['SU_Num'] = df_copy['SU_Num'].str.replace(r'^SU\s*', '', regex=True).str.replace(r'^SU', '', regex=True)
        df_copy['SU_Num'] = pd.to_numeric(df_copy['SU_Num'], errors='coerce').astype('Int64').astype(str)
        
        print(f"✅ Formatted SU_Num values to integers without decimal points")
        print(f"📊 Sample SU_Num values after formatting: {df_copy['SU_Num'].unique()[:5].tolist()}")
    
    return df_copy

def upload_to_render_db(df: pd.DataFrame, table_name: str = "final_cookie_sales_all_years") -> None:
    db_url = os.getenv("RENDER_DATABASE_URL")
    if not db_url:
        raise ValueError("RENDER_DATABASE_URL not set in environment variables")

    # Import validation functions
    from validate_data_formats import validate_data_formats, auto_fix_data_formats
    
    # Validate data formats before uploading
    print("🔍 Validating data formats before upload...")
    is_valid, issues = validate_data_formats(df)
    
    if not is_valid:
        print("⚠️  Found formatting issues. Attempting auto-fix...")
        df = auto_fix_data_formats(df)
        
        # Validate again after auto-fix
        is_valid, issues = validate_data_formats(df)
        if not is_valid:
            print("❌ Auto-fix failed. Manual intervention required.")
            for issue in issues:
                print(f"  - {issue}")
            raise ValueError("Data format validation failed")
    
    # Ensure troop_ids and SU_Num values are properly formatted before uploading
    df_formatted = format_troop_ids(df)
    df_formatted = format_su_numbers(df_formatted)
    
    engine = create_engine(db_url)

    df_formatted.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",  # or "append" if you're inserting year-by-year
        index=False
    )
    print(f"✅ Uploaded to table `{table_name}` in Render DB with validated and formatted data")
