# SDMXLLM.jl

This package provides extensions to `SDMX.jl` for integrating with Large Language Models (LLMs). It is designed to be used in conjunction with the base `SDMX.jl` package.

## Features

The `SDMXLLM.jl` package provides the following features:

*   **LLM Integration:** Connect to various LLM providers (Ollama, OpenAI, Anthropic, Azure OpenAI) to perform advanced data analysis and script generation.
*   **Advanced Mapping Inference:** Intelligently map source data to SDMX schemas using fuzzy matching, statistical analysis, and pattern recognition.
*   **Script Generation:** Automatically generate `Tidier.jl` transformation scripts from source data and SDMX schemas.
*   **Workflow Orchestration:** A complete workflow system to orchestrate the entire data transformation process from source analysis to final output.

## Configuration

To use the LLM features, you need to configure an LLM provider. This is done using the `setup_llm_config` function.

### Example: Using a local Ollama model

```julia
using SDMXLLM

# Set up a configuration for a local Ollama model
ollama_config = setup_llm_config(SDMXLLM.OLLAMA, model="llama2")

# You can now use this configuration with other functions in the package
# For example, to create a script generator:
script_generator = create_script_generator(ollama_config)
```

### Supported Providers

The following LLM providers are supported:
- `SDMXLLM.OLLAMA`
- `SDMXLLM.OPENAI`
- `SDMXLLM.ANTHROPIC`
- `SDMXLLM.AZURE_OPENAI`

For cloud providers like OpenAI, you will need to set the appropriate API key as an environment variable (e.g., `OPENAI_API_KEY`).

## Available Functions

The main functions exported by this package are:

*   `setup_llm_config`, `query_llm`, `analyze_excel_structure`
*   `create_inference_engine`, `infer_advanced_mappings`
*   `create_script_generator`, `generate_transformation_script`
*   `create_workflow`, `execute_workflow`
