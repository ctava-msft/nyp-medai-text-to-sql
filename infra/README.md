# Azure AI Foundry

[![Deploy to Azure](https://raw.githubusercontent.com/Azure/azure-quickstart-templates/master/1-CONTRIBUTION-GUIDE/images/deploytoazure.svg?sanitize=true)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fctava-msft%2Faifoundry%2Fmain%2Fazuredeploy.json)

This template sets up Azure AI Foundry Hub and Project and connected resources.

Azure AI Foundry is built on Azure Machine Learning as the primary resource provider and takes a dependency on the Cognitive Services (Azure AI Services) resource provider to surface model-as-a-service endpoints for Azure Speech, Azure Content Safety, and Azure OpenAI service.

An 'Azure AI Hub' is a special kind of 'Azure Machine Learning workspace', that is kind = "hub".

## Design

This deployment creates a fully private Azure AIFoundry environment with:
- **Inbound Access**: Disabled public access, enabled from selected IP addresses only
- **Outbound Access**: Private with Allow Internet Outbound enabled
- **Identity**: System-managed identity with credential-based authentication
- **Networking**: Complete private endpoint isolation with managed virtual network

## Resources

The following table describes the resources created in the deployment:

| Provider and type | Description |
| - | - |
| `Microsoft.Storage/storageAccounts` | Azure Storage Account for workspace data and artifacts |
| `Microsoft.Storage/storageAccounts/blobServices/containers` | Blob container within the storage account |
| `Microsoft.KeyVault/vaults` | Azure Key Vault for storing secrets and keys |
| `Microsoft.MachineLearningServices/workspaces` | Azure AI Hub (ML workspace with kind=hub) |
| `Microsoft.MachineLearningServices/workspaces/connections` | Connection to Azure OpenAI service from the AI Hub |
| `Microsoft.MachineLearningServices/workspaces/outboundRules` | Managed outbound rules for AI Search and AI Services |
| `Microsoft.Search/searchServices` | Azure AI Search service with private endpoint connectivity |
| `Microsoft.CognitiveServices/accounts` | Azure AI Services (kind=AIServices) for OpenAI and other AI models |
| `Microsoft.Network/virtualNetworks` | Virtual network for managed network isolation |
| `Microsoft.Network/privateEndpoints` | Private endpoints for all services (AI Hub, Storage, Key Vault, AI Search, AI Services) |
| `Microsoft.Network/privateDnsZones` | Private DNS zones for all private endpoint resolution |
| `Microsoft.Network/privateDnsZones/virtualNetworkLinks` | Links between private DNS zones and virtual network |
| `Microsoft.Network/privateEndpoints/privateDnsZoneGroups` | DNS zone group configuration for private endpoints |
| `Microsoft.Authorization/roleAssignments` | Comprehensive RBAC role assignments for secure access between services |

## Security Configuration

### Network Security
- **Public Access**: Enabled for all services
- **Trusted Services**: Enabled where available to allow Microsoft services
- **IP Restrictions**: Customer IP addresses added to allowlists
- **Private Endpoints**: All inter-service communication via private endpoints

### Identity and Access Management
- **Managed Identity**: System-assigned managed identity enabled for all services
- **Authentication**: Entra ID authentication for all connected resources (API keys disabled)
- **Credential Management**: Account key authentication for blob storage connections

## Role Assignments

### Developer/Data Engineer Permissions
| Resource | Role | Description |
| - | - | - |
| Azure AI Search | Search Services Contributor | List API-Keys to list indexes from Azure AI Foundry portal |
| Azure AI Search | Search Index Data Contributor | Required for the indexing scenario |
| Azure AI Search | Search Index Data Reader | Read access to search indexes |
| Azure AI Services/OpenAI | Cognitive Services OpenAI Contributor | Call public ingestion API from Azure AI Foundry portal |
| Azure AI Services/OpenAI | Cognitive Services Contributor | List API-Keys from Azure AI Foundry portal |
| Azure AI Services/OpenAI | Contributor | Allows for calls to the control plane |
| Azure Storage Account | Contributor | List Account SAS to upload files from Azure AI Foundry portal |
| Azure Storage Account | Storage Blob Data Contributor | Read and write to blob storage |
| Azure Storage Account | Storage File Data Privileged Contributor | Access File Share in Storage for Promptflow data |

### Service-to-Service Permissions
| Resource | Role | Assignee | Description |
| - | - | - | - |
| Azure AI Search | Search Index Data Contributor | Azure AI Services/OpenAI | Read-write access to content in indexes |
| Azure AI Search | Search Index Data Reader | Azure AI Services/OpenAI | Inference service queries the data from the index |
| Azure AI Search | Search Service Contributor | Azure AI Services/OpenAI | Read-write access to object definitions |
| Azure AI Services/OpenAI | Cognitive Services Contributor | Azure AI Search | Allow Search to create, read, and update AI Services resource |
| Azure AI Services/OpenAI | Cognitive Services OpenAI Contributor | Azure AI Search | Allow Search the ability to fine-tune, deploy, and generate text |
| Azure Storage Account | Storage Blob Data Contributor | Azure AI Search | Reads blob and writes knowledge store |
| Azure Storage Account | Storage Blob Data Contributor | Azure AI Services/OpenAI | Reads from input container, writes preprocess results |

## Deployment Features

### AI Foundry Hub Configuration
- **Inbound Access**: Disabled with selected IP addresses allowed
- **Outbound Access**: Private with Allow Internet Outbound
- **Identity**: System-managed identity enabled
- **Connections**: Entra ID authentication for all connected resources

### AI Search Configuration
- **Endpoint Connectivity**: Private only
- **Network Access**: Customer IP addresses in allowlist
- **Trusted Services**: Microsoft trusted services allowed

### AI Foundry Project
- **Model Deployment**: GPT-4.1 model deployment ready
- **Data Sources**: PDF upload capability to AI Foundry Data
- **Chat Playground**: Ready with data source integration

### Connected Resources
- AI Search service connected to AI Foundry Hub
- Outbound rules configured for AI Search access
- Storage account configured with account key authentication
- All connections use Entra ID authentication

## Learn more

If you are new to Azure AIFoundry, see:

- [Azure AIFoundry](https://aka.ms/aistudio/docs)
- [Securely use playground chat - Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/secure-playground-chat)

If you are new to Azure Machine Learning, see:

- [Azure Machine Learning service](https://azure.microsoft.com/services/machine-learning-service/)
- [Azure Machine Learning documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Azure Machine Learning compute instance documentation](https://docs.microsoft.com/azure/machine-learning/concept-compute-instance)
- [Azure Machine Learning template reference](https://docs.microsoft.com/azure/templates/microsoft.machinelearningservices/allversions)
- [Quickstart templates](https://azure.microsoft.com/resources/templates/)

Reference:

- [AIFoundry Managed Network](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/configure-managed-network?tabs=portal)
- [Azure AIFoundry private networking](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/configure-private-link)