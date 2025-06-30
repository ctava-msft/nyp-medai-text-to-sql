"""
Natural Language to SQL Query Converter using Azure OpenAI GPT-4
This script converts English text to SQL queries for the medical triplets database.
"""

import os
import pandas as pd
import sqlite3
import logging
from typing import Optional, Dict, Any
from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.keyvault.secrets import SecretClient
from openai import AzureOpenAI
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDataQueryProcessor:
    """
    Processes natural language queries and converts them to SQL for medical triplet data.
    Uses Azure OpenAI GPT-4 for natural language understanding and SQL generation.
    """
    
    def __init__(self, csv_file_path: str, azure_openai_endpoint: str, 
                 api_version: str = "2024-05-01-preview", 
                 deployment_name: str = "gpt-4", 
                 key_vault_url: Optional[str] = None):
        """
        Initialize the query processor.
        
        Args:
            csv_file_path: Path to the medical data CSV file
            azure_openai_endpoint: Azure OpenAI service endpoint
            api_version: Azure OpenAI API version
            deployment_name: Name of the GPT-4 deployment
            key_vault_url: Optional Key Vault URL for storing API keys
        """
        self.csv_file_path = csv_file_path
        self.azure_openai_endpoint = azure_openai_endpoint
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.key_vault_url = key_vault_url
        
        # Initialize credentials using managed identity
        self.credential = DefaultAzureCredential()
        
        # Initialize database
        self.db_path = ":memory:"  # In-memory SQLite database
        self.conn = None
        
        # Initialize Azure OpenAI client
        self.openai_client = None
        
        self._setup_database()
        self._setup_azure_openai()
    
    def _setup_database(self) -> None:
        """Set up SQLite database and load CSV data."""
        try:
            # Create in-memory SQLite database
            self.conn = sqlite3.connect(self.db_path)
            
            # Load CSV data into pandas DataFrame
            df = pd.read_csv(self.csv_file_path)
            
            # Create table and insert data
            df.to_sql('medical_data', self.conn, if_exists='replace', index=False)
            
            logger.info(f"Successfully loaded {len(df)} records into database")
            
            # Log table schema for debugging
            cursor = self.conn.cursor()
            cursor.execute("PRAGMA table_info(medical_data)")
            schema = cursor.fetchall()
            logger.info(f"Database schema: {schema}")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise
    
    def _setup_azure_openai(self) -> None:
        """Set up Azure OpenAI client with managed identity authentication."""
        try:
            if self.key_vault_url:
                # Get API key from Key Vault
                secret_client = SecretClient(vault_url=self.key_vault_url, credential=self.credential)
                api_key = secret_client.get_secret("azure-openai-api-key").value
                
                self.openai_client = AzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    api_key=api_key,
                    api_version=self.api_version
                )
            else:
                # Use managed identity for authentication
                # Get token for Azure OpenAI
                token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
                
                self.openai_client = AzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    azure_ad_token=token.token,
                    api_version=self.api_version
                )
            
            logger.info("Azure OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up Azure OpenAI: {e}")
            raise
    
    def _refresh_token_if_needed(self) -> None:
        """Refresh Azure AD token if using managed identity."""
        if not self.key_vault_url:
            try:
                token = self.credential.get_token("https://cognitiveservices.azure.com/.default")
                self.openai_client.azure_ad_token = token.token
            except Exception as e:
                logger.warning(f"Failed to refresh token: {e}")
    
    def get_database_schema(self) -> str:
        """Get the database schema as a string for the AI prompt."""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA table_info(medical_data)")
        schema_info = cursor.fetchall()
        
        # Get sample data
        cursor.execute("SELECT * FROM medical_data LIMIT 5")
        sample_data = cursor.fetchall()
        
        schema_description = """
        Database Schema:
        Table: medical_data
        Columns:
        - MEDCode (INTEGER): Medical code identifier
        - Slot (INTEGER): Slot number for the measurement
        - Value (TEXT): The measurement value or description
        
        Sample data:
        MEDCode | Slot | Value
        --------|------|-------
        """
        
        for row in sample_data:
            schema_description += f"        {row[0]} | {row[1]} | {row[2]}\n"
        
        return schema_description
    
    def generate_sql_query(self, natural_language_query: str) -> str:
        """
        Convert natural language query to SQL using Azure OpenAI GPT-4.
        
        Args:
            natural_language_query: The user's question in natural language
            
        Returns:
            Generated SQL query string
        """
        schema = self.get_database_schema()
        
        system_prompt = f"""You are an expert SQL query generator for medical data analysis.
        
        {schema}
        
        Rules:
        1. Only generate SELECT statements
        2. Use proper SQL syntax for SQLite
        3. Return only the SQL query, no explanations
        4. Handle text matching with LIKE for partial matches
        5. Use proper quoting for text values
        6. Consider case-insensitive matching where appropriate
        
        Common query patterns:
        - "Find all records for MEDCode X" -> SELECT * FROM medical_data WHERE MEDCode = X
        - "Show measurements containing 'sodium'" -> SELECT * FROM medical_data WHERE Value LIKE '%sodium%'
        - "Get all slot 150 records" -> SELECT * FROM medical_data WHERE Slot = 150
        """
        
        user_prompt = f"Convert this natural language query to SQL: {natural_language_query}"
        
        try:
            self._refresh_token_if_needed()
            
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            sql_query = response.choices[0].message.content.strip()
            
            # Clean up the SQL query (remove markdown formatting if present)
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = sql_query.strip()
            
            logger.info(f"Generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            raise
    
    def execute_query(self, sql_query: str) -> pd.DataFrame:
        """
        Execute the SQL query and return results as a DataFrame.
        
        Args:
            sql_query: The SQL query to execute
            
        Returns:
            Query results as a pandas DataFrame
        """
        try:
            # Validate that it's a SELECT query for security
            if not sql_query.strip().upper().startswith('SELECT'):
                raise ValueError("Only SELECT queries are allowed")
            
            result_df = pd.read_sql_query(sql_query, self.conn)
            logger.info(f"Query executed successfully, returned {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def process_natural_language_query(self, query: str) -> Dict[str, Any]:
        """
        Process a natural language query end-to-end.
        
        Args:
            query: Natural language query from user
            
        Returns:
            Dictionary containing SQL query, results, and metadata
        """
        try:
            # Generate SQL from natural language
            sql_query = self.generate_sql_query(query)
            
            # Execute the query
            results_df = self.execute_query(sql_query)
            
            return {
                "natural_language_query": query,
                "generated_sql": sql_query,
                "results": results_df,
                "row_count": len(results_df),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {
                "natural_language_query": query,
                "generated_sql": None,
                "results": None,
                "row_count": 0,
                "success": False,
                "error": str(e)
            }
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Main function to demonstrate usage."""
    # Configuration - replace with your Azure OpenAI details
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-openai-resource.openai.azure.com/")
    DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    KEY_VAULT_URL = os.getenv("AZURE_KEY_VAULT_URL")  # Optional
    CSV_FILE_PATH = "medical_data.csv"
    
    # Initialize the processor
    try:
        processor = MedicalDataQueryProcessor(
            csv_file_path=CSV_FILE_PATH,
            azure_openai_endpoint=AZURE_OPENAI_ENDPOINT,
            deployment_name=DEPLOYMENT_NAME,
            key_vault_url=KEY_VAULT_URL
        )
        
        # Example queries
        example_queries = [
            "Show me all records for MEDCode 1302",
            "Find measurements containing 'sodium'",
            "Get all records where the slot is 150",
            "Show me all measurement descriptions",
            "Find records with empty values"
        ]
        
        print("Medical Data Query Processor")
        print("=" * 50)
        
        # Process example queries
        for query in example_queries:
            print(f"\nQuery: {query}")
            result = processor.process_natural_language_query(query)
            
            if result["success"]:
                print(f"Generated SQL: {result['generated_sql']}")
                print(f"Results ({result['row_count']} rows):")
                if result['row_count'] > 0:
                    print(result['results'].to_string(index=False))
                else:
                    print("No results found")
            else:
                print(f"Error: {result['error']}")
            
            print("-" * 50)
        
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.")
        while True:
            user_query = input("\nEnter your query: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_query:
                result = processor.process_natural_language_query(user_query)
                
                if result["success"]:
                    print(f"Generated SQL: {result['generated_sql']}")
                    print(f"Results ({result['row_count']} rows):")
                    if result['row_count'] > 0:
                        print(result['results'].to_string(index=False))
                    else:
                        print("No results found")
                else:
                    print(f"Error: {result['error']}")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
    
    finally:
        if 'processor' in locals():
            processor.close()

if __name__ == "__main__":
    main()
