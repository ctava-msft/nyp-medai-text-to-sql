# Medical Data Natural Language Query System

This project provides a natural language interface to query medical triplet data using Azure OpenAI GPT-4.

## Features

- Convert natural language queries to SQL using Azure OpenAI GPT-4
- Query medical triplet data stored in CSV format
- Secure authentication using Azure Managed Identity
- Support for Azure Key Vault for credential management
- Interactive command-line interface
- Comprehensive error handling and logging

## Files

- `medical_data.csv` - The medical triplet data in CSV format
- `natural_language_to_sql.py` - Main Python script for natural language to SQL conversion
- `requirements.txt` - Python dependencies
- `.env.template` - Environment configuration template

# Setup venv environment

```
python -m venv .venv
./.venv/Scripts/activate
pip install -r requirements.txt
```

## Setup

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI

1. Copy `.env.template` to `.env`
2. Update the Azure OpenAI endpoint and deployment name in `.env`
3. Ensure you have access to Azure OpenAI service

### 3. Authentication

The script uses Azure Managed Identity for authentication. For local development, you can:

- Use Azure CLI: `az login`
- Or set up a service principal with appropriate permissions
- Or use Azure Key Vault to store API keys

### 4. Azure OpenAI Setup

Ensure your Azure OpenAI resource has:
- A GPT-4 deployment (or update the deployment name in configuration)
- Appropriate RBAC permissions for your identity

## Usage

### Run the Script

```powershell
python natural_language_to_sql.py
```

### Example Queries

The script supports natural language queries like:

- "Show me all records for MEDCode 1302"
- "Find measurements containing 'sodium'"
- "Get all records where the slot is 150"
- "Show me all measurement descriptions"
- "Find records with empty values"

### Interactive Mode

After running example queries, the script enters interactive mode where you can type your own natural language queries.

## Data Schema

The medical data contains three columns:
- `MEDCode`: Medical code identifier (INTEGER)
- `Slot`: Slot number for the measurement (INTEGER)  
- `Value`: The measurement value or description (TEXT)

## Security Features

- Uses Azure Managed Identity for authentication
- Supports Azure Key Vault for secure credential storage
- Only allows SELECT queries for data safety
- Proper input validation and error handling
- No hardcoded credentials in the code

## Error Handling

The script includes comprehensive error handling for:
- Database connection issues
- Azure OpenAI API errors
- Authentication failures
- Invalid SQL queries
- Network connectivity issues

## Logging

Detailed logging is provided for:
- Database operations
- Azure OpenAI interactions
- Query processing
- Error conditions

## Configuration Options

You can customize the following in your `.env` file:
- Azure OpenAI endpoint and deployment
- API version
- Key Vault URL (optional)
- CSV file path

## Troubleshooting

1. **Authentication Issues**: Ensure you're logged in with `az login` or have proper managed identity setup
2. **OpenAI Errors**: Verify your deployment name and endpoint are correct
3. **CSV Issues**: Ensure the CSV file exists and has the correct format
4. **Dependencies**: Run `pip install -r requirements.txt` to install all required packages

## Architecture

The system follows Azure best practices:
- Managed Identity for secure authentication
- Key Vault integration for credential management
- Proper error handling and retry logic
- Logging and monitoring capabilities
- Separation of concerns with clean code structure
