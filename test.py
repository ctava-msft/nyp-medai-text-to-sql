"""
Demo script to test the medical data CSV loading and basic SQL queries
without requiring Azure OpenAI setup.
"""

import pandas as pd
import sqlite3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_csv_and_database():
    """Test loading CSV data and basic SQL operations."""
    
    csv_file = "./data/medical_data.csv"
    
    try:
        # Load CSV data
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded CSV with {len(df)} records")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print(f"\nData types:")
        print(df.dtypes)
        
        # Create in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        
        # Load data into SQLite
        df.to_sql('medical_data', conn, if_exists='replace', index=False)
        
        # Test queries
        test_queries = [
            ("All records for MEDCode 1302", "SELECT * FROM medical_data WHERE MEDCode = 1302"),
            ("Records containing 'sodium'", "SELECT * FROM medical_data WHERE Value LIKE '%sodium%'"),
            ("All slot 150 records", "SELECT * FROM medical_data WHERE Slot = 150"),
            ("Count by MEDCode", "SELECT MEDCode, COUNT(*) as count FROM medical_data GROUP BY MEDCode"),
            ("Distinct slots", "SELECT DISTINCT Slot FROM medical_data ORDER BY Slot")
        ]
        
        print("\n" + "="*60)
        print("Testing SQL Queries")
        print("="*60)
        
        for description, query in test_queries:
            print(f"\n{description}:")
            print(f"SQL: {query}")
            
            result_df = pd.read_sql_query(query, conn)
            print(f"Results ({len(result_df)} rows):")
            
            if len(result_df) > 0:
                print(result_df.to_string(index=False))
            else:
                print("No results found")
            
            print("-" * 40)
        
        conn.close()
        print("\nDatabase test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_csv_and_database()
