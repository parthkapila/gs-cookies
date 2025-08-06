#!/usr/bin/env python3
"""
Data validation script to ensure all data formats are correct before uploading.
This should be run before uploading data to prevent formatting issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def validate_data_formats(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that all data formats are correct before uploading.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_issues)
    """
    issues = []
    
    print("🔍 Validating data formats...")
    
    # 1. Validate troop_id formats
    if 'troop_id' in df.columns:
        print("📝 Checking troop_id formats...")
        numeric_troop_ids = df[df['troop_id'].astype(str).str.match(r'^\d+$', na=False)]
        
        if not numeric_troop_ids.empty:
            # Check if numeric troop_ids are properly formatted (5-digit)
            improperly_formatted = numeric_troop_ids[
                numeric_troop_ids['troop_id'].astype(str).str.len() != 5
            ]
            
            if not improperly_formatted.empty:
                issues.append(f"Found {len(improperly_formatted)} troop_ids that are not 5-digit format")
                print(f"⚠️  Found {len(improperly_formatted)} troop_ids that need formatting")
            else:
                print("✅ All troop_ids are properly formatted")
    
    # 2. Validate SU_Num formats
    if 'SU_Num' in df.columns:
        print("📝 Checking SU_Num formats...")
        
        # Check for decimal values
        decimal_su_nums = df[df['SU_Num'].astype(str).str.contains(r'\.0$', na=False)]
        
        if not decimal_su_nums.empty:
            issues.append(f"Found {len(decimal_su_nums)} SU_Num values with decimal points")
            print(f"⚠️  Found {len(decimal_su_nums)} SU_Num values with decimal points")
        else:
            print("✅ All SU_Num values are properly formatted")
    
    # 3. Validate numeric columns
    numeric_columns = ['number_of_girls', 'number_cases_sold']
    for col in numeric_columns:
        if col in df.columns:
            print(f"📝 Checking {col} formats...")
            
            # Check for non-numeric values
            non_numeric = df[pd.to_numeric(df[col], errors='coerce').isna()]
            
            if not non_numeric.empty:
                issues.append(f"Found {len(non_numeric)} non-numeric values in {col}")
                print(f"⚠️  Found {len(non_numeric)} non-numeric values in {col}")
            else:
                print(f"✅ All {col} values are numeric")
    
    # 4. Validate required columns exist
    required_columns = ['troop_id', 'SU_Num', 'number_of_girls', 'number_cases_sold', 'cookie_type']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
        print(f"⚠️  Missing required columns: {missing_columns}")
    else:
        print("✅ All required columns are present")
    
    # 5. Check for empty/null values in critical columns
    critical_columns = ['troop_id', 'SU_Num', 'number_of_girls', 'number_cases_sold']
    for col in critical_columns:
        if col in df.columns:
            empty_count = df[col].isna().sum() + (df[col] == '').sum()
            if empty_count > 0:
                issues.append(f"Found {empty_count} empty/null values in {col}")
                print(f"⚠️  Found {empty_count} empty/null values in {col}")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        print("✅ All data formats are valid!")
    else:
        print(f"❌ Found {len(issues)} formatting issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    return is_valid, issues

def auto_fix_data_formats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically fix common data format issues.
    
    Returns:
        pd.DataFrame: Fixed dataframe
    """
    print("🔧 Auto-fixing data formats...")
    
    df_fixed = df.copy()
    
    # 1. Fix troop_id formats
    if 'troop_id' in df_fixed.columns:
        print("📝 Fixing troop_id formats...")
        df_fixed['troop_id'] = df_fixed['troop_id'].astype(str).str.strip()
        df_fixed['troop_id'] = df_fixed['troop_id'].apply(
            lambda x: f"{int(x):05d}" if x.strip().isdigit() else f"{x:>5}"
        )
    
    # 2. Fix SU_Num formats
    if 'SU_Num' in df_fixed.columns:
        print("📝 Fixing SU_Num formats...")
        df_fixed['SU_Num'] = df_fixed['SU_Num'].astype(str).str.strip()
        # Remove any 'SU' prefix
        df_fixed['SU_Num'] = df_fixed['SU_Num'].str.replace(r'^SU\s*', '', regex=True)
        # Convert to integer to remove decimal points
        df_fixed['SU_Num'] = pd.to_numeric(df_fixed['SU_Num'], errors='coerce').astype('Int64').astype(str)
    
    # 3. Fix numeric columns
    numeric_columns = ['number_of_girls', 'number_cases_sold']
    for col in numeric_columns:
        if col in df_fixed.columns:
            print(f"📝 Fixing {col} formats...")
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')
    
    print("✅ Auto-fixing completed!")
    return df_fixed

if __name__ == "__main__":
    # This script can be run independently for testing
    print("🔍 Data validation script")
    print("This script validates and fixes data formats before uploading.") 