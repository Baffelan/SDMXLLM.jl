# SDMXLLM.jl

This package provides extensions to `SDMX.jl` for integrating with Large Language Models (LLMs). It is designed to be used in conjunction with the base `SDMX.jl` package.

## Features

The `SDMXLLM.jl` package provides the following features:

*   **LLM Integration:** Connect to various LLM providers (Ollama, OpenAI, Anthropic, Azure OpenAI) to perform advanced data analysis and script generation.
*   **Advanced Mapping Inference:** Intelligently map source data to SDMX schemas using fuzzy matching, statistical analysis, and pattern recognition.
*   **Script Generation:** Automatically generate `Tidier.jl` transformation scripts from source data and SDMX schemas.
*   **Workflow Orchestration:** A complete workflow system to orchestrate the entire data transformation process from source analysis to final output.

## Configuration

To use the LLM features, you need to configure an LLM provider using the `setup_sdmx_llm` function.

### Supported Providers

SDMXLLM supports multiple LLM providers through PromptingTools.jl:
- `:ollama` - Local Ollama models
- `:openai` - OpenAI GPT models  
- `:anthropic` - Anthropic Claude models
- `:google` - Google Gemini models
- `:mistral` - Mistral AI models
- `:groq` - Groq fast inference
- `:azure_openai` - Azure OpenAI service

### API Key Configuration

For cloud providers, you need to set API keys. There are two methods:

#### Method 1: Environment Variables (Recommended for Google)
```julia
# IMPORTANT: For Google AI, set the API key BEFORE importing packages
ENV["GOOGLE_API_KEY"] = "your-api-key-here"

using SDMXLLM
setup_sdmx_llm(:google, model="gemini-1.5-flash")
```

#### Method 2: Using .env File
Create a `.env` file in YAML format:
```yaml
GOOGLE_API_KEY: "your-google-api-key"
OPENAI_API_KEY: "your-openai-api-key"
ANTHROPIC_API_KEY: "your-anthropic-api-key"
```

Then load it:
```julia
using SDMXLLM
setup_sdmx_llm(:openai, env_file=".env")
```

**⚠️ Important Note for Google AI Users:**
Due to how GoogleGenAI.jl initializes, the `GOOGLE_API_KEY` environment variable must be set BEFORE importing SDMXLLM. The `.env` file loading will work for future imports but not for the current session. For Google AI, we recommend setting the environment variable directly in your script or shell before starting Julia.

### Example Usage

```julia
# For local Ollama
using SDMXLLM
setup_sdmx_llm(:ollama, model="llama3")

# For OpenAI (with env var already set)
using SDMXLLM
setup_sdmx_llm(:openai, model="gpt-4o")

# For Google (requires env var before import)
ENV["GOOGLE_API_KEY"] = "your-key"
using SDMXLLM
setup_sdmx_llm(:google, model="gemini-1.5-flash")

# Use the configured provider
response = sdmx_aigenerate("Your prompt here", provider=:google)
```

## Available Functions

The main functions exported by this package are:

*   `setup_llm_config`, `query_llm`, `analyze_excel_structure`
*   `create_inference_engine`, `infer_advanced_mappings`
*   `create_script_generator`, `generate_transformation_script`
*   `create_workflow`, `execute_workflow`
