#!/usr/bin/env python3
"""
Comprehensive data consistency script to ensure all data formatting is correct.
This should be run as part of the automation pipeline to prevent formatting issues.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def ensure_data_consistency():
    """Ensure all data in the database is properly formatted"""
    
    # Load environment variables
    load_dotenv()
    database_url = os.getenv('RENDER_DATABASE_URL')
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    engine = create_engine(database_url)
    
    print("🔍 Checking data consistency in database...")
    
    # 1. Fix troop_id formatting
    print("\n📝 Step 1: Ensuring troop_id formatting...")
    with engine.connect() as conn:
        # Get all unique troop_ids that need formatting
        result = conn.execute(text("""
            SELECT troop_id, COUNT(*) as count 
            FROM final_cookie_sales_all_years 
            WHERE troop_id ~ '^[0-9]+$'  -- Only numeric troop_ids
            GROUP BY troop_id 
            ORDER BY troop_id
        """))
        
        current_troop_ids = result.fetchall()
        
        # Find troop_ids that need updating
        troop_ids_to_update = []
        for troop_id, count in current_troop_ids:
            if troop_id.strip().isdigit():
                expected_format = f"{int(troop_id):05d}"
                if troop_id != expected_format:
                    troop_ids_to_update.append((troop_id, expected_format, count))
    
    if not troop_ids_to_update:
        print("✅ All troop_ids are already in the correct format!")
    else:
        print(f"📝 Found {len(troop_ids_to_update)} troop_ids that need formatting:")
        for old_id, new_id, count in troop_ids_to_update[:10]:
            print(f"  {old_id} → {new_id} ({count} records)")
        
        if len(troop_ids_to_update) > 10:
            print(f"  ... and {len(troop_ids_to_update) - 10} more")
        
        # Update the database
        print("\n🔄 Updating troop_ids in database...")
        
        with engine.connect() as conn:
            for old_id, new_id, count in troop_ids_to_update:
                result = conn.execute(text("""
                    UPDATE final_cookie_sales_all_years 
                    SET troop_id = :new_id 
                    WHERE troop_id = :old_id
                """), {"old_id": old_id, "new_id": new_id})
                
                print(f"  Updated {result.rowcount} records: {old_id} → {new_id}")
            
            conn.commit()
        
        print("✅ troop_id formatting completed!")
    
    # 2. Fix SU_Num formatting
    print("\n📝 Step 2: Ensuring SU_Num formatting...")
    
    with engine.connect() as conn:
        # Get all unique SU_Num values that need formatting
        result = conn.execute(text("""
            SELECT "SU_Num", COUNT(*) as count 
            FROM final_cookie_sales_all_years 
            WHERE CAST("SU_Num" AS TEXT) ~ '^[0-9]+\\.0$'  -- Only SU_Num values with .0
            GROUP BY "SU_Num" 
            ORDER BY "SU_Num"
        """))
        
        current_su_nums = result.fetchall()
        
        # Find SU_Num values that need updating
        su_nums_to_update = []
        for su_num, count in current_su_nums:
            if su_num.endswith('.0'):
                expected_format = su_num.replace('.0', '')
                su_nums_to_update.append((su_num, expected_format, count))
        
        if not su_nums_to_update:
            print("✅ All SU_Num values are already in the correct format!")
        else:
            print(f"📝 Found {len(su_nums_to_update)} SU_Num values that need formatting:")
            for old_id, new_id, count in su_nums_to_update[:10]:
                print(f"  {old_id} → {new_id} ({count} records)")
            
            if len(su_nums_to_update) > 10:
                print(f"  ... and {len(su_nums_to_update) - 10} more")
            
            # Update the database
            print("\n🔄 Updating SU_Num values in database...")
            
            with engine.connect() as conn:
                for old_id, new_id, count in su_nums_to_update:
                    result = conn.execute(text("""
                        UPDATE final_cookie_sales_all_years 
                        SET "SU_Num" = :new_id 
                        WHERE "SU_Num" = :old_id
                    """), {"old_id": old_id, "new_id": new_id})
                    
                    print(f"  Updated {result.rowcount} records: {old_id} → {new_id}")
                
                conn.commit()
            
            print("✅ SU_Num formatting completed!")
        
        current_su_nums = result.fetchall()
        
        # Find SU_Num values that need updating
        su_nums_to_update = []
        for su_num, count in current_su_nums:
            if su_num.endswith('.0'):
                expected_format = su_num.replace('.0', '')
                su_nums_to_update.append((su_num, expected_format, count))
    
    if not su_nums_to_update:
        print("✅ All SU_Num values are already in the correct format!")
    else:
        print(f"📝 Found {len(su_nums_to_update)} SU_Num values that need formatting:")
        for old_id, new_id, count in su_nums_to_update[:10]:
            print(f"  {old_id} → {new_id} ({count} records)")
        
        if len(su_nums_to_update) > 10:
            print(f"  ... and {len(su_nums_to_update) - 10} more")
        
        # Update the database
        print("\n🔄 Updating SU_Num values in database...")
        
        with engine.connect() as conn:
            for old_id, new_id, count in su_nums_to_update:
                result = conn.execute(text("""
                    UPDATE final_cookie_sales_all_years 
                    SET "SU_Num" = :new_id 
                    WHERE "SU_Num" = :old_id
                """), {"old_id": old_id, "new_id": new_id})
                
                print(f"  Updated {result.rowcount} records: {old_id} → {new_id}")
            
            conn.commit()
        
        print("✅ SU_Num formatting completed!")
    
    # 3. Verify data consistency
    print("\n🔍 Step 3: Verifying data consistency...")
    
    with engine.connect() as conn:
        # Check troop_id consistency
        result = conn.execute(text("""
            SELECT COUNT(*) as total_records,
                   COUNT(CASE WHEN troop_id ~ '^[0-9]{5}$' THEN 1 END) as properly_formatted_troop_ids
            FROM final_cookie_sales_all_years 
            WHERE troop_id ~ '^[0-9]+$'
        """))
        
        troop_stats = result.fetchone()
        print(f"📊 Troop ID Consistency:")
        print(f"  Total numeric troop_ids: {troop_stats[0]}")
        print(f"  Properly formatted (5-digit): {troop_stats[1]}")
        print(f"  Consistency: {troop_stats[1]/troop_stats[0]*100:.1f}%" if troop_stats[0] > 0 else "  Consistency: N/A")
        
        # Check SU_Num consistency
        result = conn.execute(text("""
            SELECT COUNT(*) as total_records,
                   COUNT(CASE WHEN CAST("SU_Num" AS TEXT) ~ '^[0-9]+$' THEN 1 END) as properly_formatted_su_nums
            FROM final_cookie_sales_all_years 
            WHERE CAST("SU_Num" AS TEXT) ~ '^[0-9]+(\\.0)?$'
        """))
        
        su_stats = result.fetchone()
        print(f"📊 SU_Num Consistency:")
        print(f"  Total SU_Num records: {su_stats[0]}")
        print(f"  Properly formatted (integer): {su_stats[1]}")
        print(f"  Consistency: {su_stats[1]/su_stats[0]*100:.1f}%" if su_stats[0] > 0 else "  Consistency: N/A")
    
    print("\n✅ Data consistency check completed!")

if __name__ == "__main__":
    ensure_data_consistency() 