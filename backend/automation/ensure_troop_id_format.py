#!/usr/bin/env python3
"""
Script to ensure all troop_ids in the database are properly formatted as 5-digit strings with leading zeros.
This should be run as part of the automation pipeline to maintain consistency.
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

def ensure_troop_id_format():
    """Ensure all troop_ids and SU_Num values in the database are properly formatted"""
    
    # Load environment variables
    load_dotenv()
    database_url = os.getenv('RENDER_DATABASE_URL')
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    engine = create_engine(database_url)
    
    print("🔍 Checking troop_id and SU_Num formats in database...")
    
    # Check current state
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
        return
    
    print(f"📝 Found {len(troop_ids_to_update)} troop_ids that need formatting:")
    for old_id, new_id, count in troop_ids_to_update[:10]:
        print(f"  {old_id} → {new_id} ({count} records)")
    
    if len(troop_ids_to_update) > 10:
        print(f"  ... and {len(troop_ids_to_update) - 10} more")
    
    # Update the database
    print("\n🔄 Updating troop_ids in database...")
    
    with engine.connect() as conn:
        for old_id, new_id, count in troop_ids_to_update:
            # Update all records with this troop_id
            result = conn.execute(text("""
                UPDATE final_cookie_sales_all_years 
                SET troop_id = :new_id 
                WHERE troop_id = :old_id
            """), {"old_id": old_id, "new_id": new_id})
            
            print(f"  Updated {result.rowcount} records: {old_id} → {new_id}")
        
        conn.commit()
    
    print("\n✅ Database troop_id formatting completed!")
    
    # Now fix SU_Num values (remove decimal points)
    print("\n🔍 Checking SU_Num formats in database...")
    
    with engine.connect() as conn:
        # Get all unique SU_Num values that need formatting
        result = conn.execute(text("""
            SELECT SU_Num, COUNT(*) as count 
            FROM final_cookie_sales_all_years 
            WHERE SU_Num ~ '^[0-9]+\\.0$'  -- Only SU_Num values with .0
            GROUP BY SU_Num 
            ORDER BY SU_Num
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
                # Update all records with this SU_Num
                result = conn.execute(text("""
                    UPDATE final_cookie_sales_all_years 
                    SET SU_Num = :new_id 
                    WHERE SU_Num = :old_id
                """), {"old_id": old_id, "new_id": new_id})
                
                print(f"  Updated {result.rowcount} records: {old_id} → {new_id}")
            
            conn.commit()
        
        print("\n✅ Database SU_Num formatting completed!")
    
    # Verify the changes
    print("\n🔍 Verifying changes...")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT troop_id, COUNT(*) as count 
            FROM final_cookie_sales_all_years 
            WHERE troop_id ~ '^[0-9]+$'  -- Only numeric troop_ids
            GROUP BY troop_id 
            ORDER BY troop_id
            LIMIT 10
        """))
        
        updated_troop_ids = result.fetchall()
        print("Updated troop_ids (first 10):")
        for troop_id, count in updated_troop_ids:
            print(f"  {troop_id} (appears {count} times)")
        
        # Also verify SU_Num values
        result = conn.execute(text("""
            SELECT SU_Num, COUNT(*) as count 
            FROM final_cookie_sales_all_years 
            WHERE SU_Num ~ '^[0-9]+$'  -- Only numeric SU_Num values
            GROUP BY SU_Num 
            ORDER BY SU_Num
            LIMIT 10
        """))
        
        updated_su_nums = result.fetchall()
        print("\nUpdated SU_Num values (first 10):")
        for su_num, count in updated_su_nums:
            print(f"  {su_num} (appears {count} times)")

if __name__ == "__main__":
    ensure_troop_id_format() 