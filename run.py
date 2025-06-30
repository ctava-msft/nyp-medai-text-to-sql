"""
Natural Language to SQL Query Converter using Azure OpenAI
Combines working Azure OpenAI authentication with SQLite database functionality.
"""

import os
import pandas as pd
import sqlite3
import logging
import requests
import re
import random
from typing import Dict, Any
from azure.identity import InteractiveBrowserCredential
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class MedicalDataQueryProcessor:
    """
    Processes natural language queries and converts them to SQL for medical triplet data.
    Uses Azure OpenAI for natural language understanding and SQL generation.
    """
    
    def __init__(self, csv_file_path: str):
        """
        Initialize the query processor.
        
        Args:
            csv_file_path: Path to the medical data CSV file
        """
        self.csv_file_path = csv_file_path
        
        # Validate required environment variables
        self.required_vars = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION", 
            "AZURE_OPENAI_DEPLOYMENT_NAME",
            "AZURE_TENANT_ID"
        ]
        
        self._validate_environment()
        
        # Initialize database
        self.db_path = ":memory:"  # In-memory SQLite database
        self.conn = None
        
        # Initialize Azure authentication
        self.access_token = None
        
        self._setup_database()
        self._setup_azure_auth()
    
    def _validate_environment(self) -> None:
        """Validate that all required environment variables are set."""
        for var in self.required_vars:
            if not os.getenv(var):
                logger.error(f"Missing required environment variable: {var}")
                raise ValueError(f"Missing required environment variable: {var}")
    
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
    
    def _setup_azure_auth(self) -> None:
        """Set up Azure authentication using InteractiveBrowserCredential."""
        try:
            tenant_id = os.getenv("AZURE_TENANT_ID")
            credential = InteractiveBrowserCredential(tenant_id=tenant_id)
            scope = "https://cognitiveservices.azure.com/.default"
            self.access_token = credential.get_token(scope)
            logger.info("Authentication successful!")
            
        except Exception as e:
            logger.error(f"Error setting up Azure authentication: {e}")
            raise
    
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
    
    def query_azure_openai(self, prompt: str) -> str:
        """
        Query Azure OpenAI with the given prompt.
        
        Args:
            prompt: The prompt to send to Azure OpenAI
            
        Returns:
            The response from Azure OpenAI
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.access_token.token}"
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": prompt}
            ],
            "max_tokens": 500,
            "temperature": 0.1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "top_p": 0.90,
            "stop": None
        }
        
        try:
            response = requests.post(
                f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/deployments/{os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}/chat/completions?api-version={os.getenv('AZURE_OPENAI_API_VERSION')}",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Azure OpenAI: {e.response.text if e.response else e}")
            raise
    
    def generate_sql_query(self, natural_language_query: str) -> str:
        """
        Convert natural language query to SQL using Azure OpenAI.
        
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
        3. Return only the SQL query, no explanations or markdown formatting
        4. Handle text matching with LIKE for partial matches
        5. Use proper quoting for text values
        6. Consider case-insensitive matching where appropriate
        
        Common query patterns:
        - "Find all records for MEDCode X" -> SELECT * FROM medical_data WHERE MEDCode = X
        - "Show measurements containing 'sodium'" -> SELECT * FROM medical_data WHERE Value LIKE '%sodium%'
        - "Get all slot 150 records" -> SELECT * FROM medical_data WHERE Slot = 150
        
        Convert this natural language query to SQL: {natural_language_query}"""
        
        try:
            sql_response = self.query_azure_openai(system_prompt)
            
            # Clean up the SQL query (remove markdown formatting if present)
            sql_query = re.sub(r'```sql\n?', '', sql_response)
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

def format_results_to_markdown(result: Dict[str, Any]) -> str:
    """
    Format query results to markdown format.
    
    Args:
        result: Query result dictionary
        
    Returns:
        Formatted markdown string
    """
    markdown_content = f"## Query: {result['natural_language_query']}\n\n"
    
    if result["success"]:
        markdown_content += f"**Generated SQL:** `{result['generated_sql']}`\n\n"
        markdown_content += f"**Results ({result['row_count']} rows):**\n\n"
        
        if result['row_count'] > 0:
            # Convert DataFrame to markdown table
            markdown_content += result['results'].to_markdown(index=False)
            markdown_content += "\n\n"
        else:
            markdown_content += "No results found\n\n"
    else:
        markdown_content += f"**Error:** {result['error']}\n\n"
    
    markdown_content += "---\n\n"
    return markdown_content

def main():
    """Main function to demonstrate usage."""
    CSV_FILE_PATH = "./data/medical_data.csv"
    
    # Generate random suffix for output file
    random_suffix = random.randint(1000, 9999)
    output_file = f"output_{random_suffix}.md"
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE_PATH):
        logger.error(f"CSV file not found: {CSV_FILE_PATH}")
        error_msg = f"Error: CSV file not found at {CSV_FILE_PATH}\nPlease ensure the medical data CSV file exists"
        print(error_msg)
        
        # Create error file immediately
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Medical Data Query Results\n\n**Error:** {error_msg}\n")
        print(f"Error details saved to {output_file}")
        return
    
    # Initialize markdown content and create file immediately
    initial_content = f"# Medical Data Query Results\n\n"
    initial_content += f"Generated at: {pd.Timestamp.now()}\n\n"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(initial_content)
    
    print(f"Medical Data Query Processor - Output will be saved to {output_file}")
    print(f"Output file created: {output_file}")
    print("=" * 50)
    
    # Initialize the processor
    try:
        processor = MedicalDataQueryProcessor(csv_file_path=CSV_FILE_PATH)
        
        # Example queries
        example_queries = [
            "Show me all records for MEDCode 1302",
            "Find measurements containing 'sodium'", 
            "Get all records where the slot is 150",
            "Show me all measurement descriptions",
            "Find records with empty values"
        ]
        
        # Process example queries
        for query in example_queries:
            print(f"Processing: {query}")
            result = processor.process_natural_language_query(query)
            
            # Append to file immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(format_results_to_markdown(result))
            
            print(f"  → Added to {output_file}")
        
        # Interactive mode
        print("\nEntering interactive mode. Type 'quit' to exit.")
        
        # Add interactive section header
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("## Interactive Queries\n\n")
        
        while True:
            user_query = input("\nEnter your query: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_query:
                result = processor.process_natural_language_query(user_query)
                
                # Append to file immediately
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(format_results_to_markdown(result))
                
                print(f"Query processed and added to {output_file}")
        
        print(f"\nAll results have been saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        
        # Append error to existing file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n## Application Error\n\n**Error:** {e}\n")
        
        print(f"Error details appended to {output_file}")
    
    finally:
        if 'processor' in locals():
            processor.close()

if __name__ == "__main__":
    main()
