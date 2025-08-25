# SDMXLLM.jl

[![Build Status](https://github.com/yourusername/julia_sdmx/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/yourusername/julia_sdmx/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/yourusername/julia_sdmx/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/julia_sdmx)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

LLM-powered extension for SDMX.jl that adds intelligent data transformation capabilities. Use Large Language Models to analyze complex data structures, generate transformation scripts, and orchestrate end-to-end SDMX conversion workflows.

## Requirements

- Julia 1.11 or higher
- SDMX.jl package (see [../SDMX.jl](../SDMX.jl))
- See [Project.toml](Project.toml) for full dependencies

## Features

- 🤖 **Multi-Provider LLM Support**: Ollama, OpenAI, Anthropic, Google, Mistral, Groq
- 🧠 **Intelligent Mapping**: Advanced fuzzy matching and semantic analysis
- 📝 **Script Generation**: Automatic Tidier.jl transformation code generation
- 🔄 **Workflow Orchestration**: Complete transformation pipelines
- 📊 **Excel Analysis**: Multi-sheet workbook structure understanding
- ✨ **Pattern Recognition**: Hierarchical relationship detection

## Installation

```julia
using Pkg
# Install SDMX.jl first
Pkg.add(url="https://github.com/yourusername/SDMX.jl")
# Then install SDMXLLM.jl
Pkg.add(url="https://github.com/yourusername/SDMXLLM.jl")

# For development
Pkg.develop(path="path/to/SDMX.jl")
Pkg.develop(path="path/to/SDMXLLM.jl")
```

## Quick Start

```julia
using SDMX
using SDMXLLM
using DataFrames

# Configure LLM provider
setup_sdmx_llm(:ollama; model="llama3")

# Load SDMX schema
url = "https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/latest?references=all"
schema = extract_dataflow_schema(url)

# Profile source data
source_data = CSV.read("my_data.csv", DataFrame)
profile = profile_source_data(source_data, "my_data.csv")

# Create inference engine for advanced mapping
engine = create_inference_engine(fuzzy_threshold=0.7)
mapping_result = infer_advanced_mappings(engine, profile, schema, source_data)

println("Mapping quality: " * string(round(mapping_result.quality_score, digits=2)))
println("Coverage: " * string(round(mapping_result.coverage_analysis["required_coverage"] * 100, digits=1)) * "%")

# Generate transformation script
generator = create_script_generator(:ollama, "llama3")
script = generate_transformation_script(
    generator, profile, schema, mapping_result,
    output_path="transformed_data.csv"
)

# Save and review script
write("transform_to_sdmx.jl", script.generated_code)
println("Script saved to transform_to_sdmx.jl")
```

## LLM Provider Configuration

### Local Models (Ollama)
```julia
# Default Ollama setup
setup_sdmx_llm(:ollama; model="llama3")

# Custom Ollama endpoint
setup_sdmx_llm(:ollama; 
    model="mixtral",
    base_url="http://localhost:11434"
)
```

### Cloud Providers

#### OpenAI
```julia
# Set API key via environment variable
ENV["OPENAI_API_KEY"] = "sk-..."
setup_sdmx_llm(:openai; model="gpt-4")

# Or load from .env file
setup_sdmx_llm(:openai; model="gpt-4", env_file=".env")
```

#### Google Gemini
```julia
# IMPORTANT: Set BEFORE importing SDMXLLM
ENV["GOOGLE_API_KEY"] = "AIza..."
using SDMXLLM
setup_sdmx_llm(:google; model="gemini-1.5-flash")
```

#### Anthropic Claude
```julia
ENV["ANTHROPIC_API_KEY"] = "sk-ant-..."
setup_sdmx_llm(:anthropic; model="claude-3-sonnet")
```

### .env File Format
```yaml
OPENAI_API_KEY: "sk-..."
GOOGLE_API_KEY: "AIza..."
ANTHROPIC_API_KEY: "sk-ant-..."
MISTRAL_API_KEY: "..."
GROQ_API_KEY: "..."
```

## Core Modules

### LLM Integration
Direct LLM queries with SDMX context:
```julia
# Query LLM about data
response = sdmx_aigenerate(
    "Analyze this data structure and suggest SDMX mappings",
    provider=:ollama
)

# Analyze Excel structure
excel_analysis = analyze_excel_structure("complex_workbook.xlsx")
println("Sheets: " * join(excel_analysis.sheets, ", "))
println("Recommended sheet: " * excel_analysis.recommended_sheet)
if excel_analysis.pivoting_detected
    println("Pivoting required for transformation")
end
```

### Advanced Mapping Inference
Intelligent column mapping with confidence scoring:
```julia
# Configure inference engine
engine = create_inference_engine(
    fuzzy_threshold=0.6,      # Minimum fuzzy match score
    min_confidence=0.2,        # Minimum confidence threshold
    enable_llm=true           # Use LLM for semantic matching
)

# Run advanced mapping
result = infer_advanced_mappings(engine, profile, schema, source_data)

# Examine mappings
for mapping in result.mappings
    println(mapping.source_column * " -> " * mapping.target_column)
    println("  Confidence: " * string(mapping.confidence_level))
    println("  Match type: " * mapping.match_type)
    if mapping.suggested_transformation !== nothing
        println("  Transform: " * mapping.suggested_transformation)
    end
end

# Check unmapped columns
if !isempty(result.unmapped_target_columns)
    println("Missing required columns: " * join(result.unmapped_target_columns, ", "))
end
```

### Script Generation
Generate complete transformation scripts:
```julia
# Create script generator with options
generator = create_script_generator(:ollama, "llama3";
    include_validation=true,    # Add validation checks
    include_comments=true,      # Add explanatory comments
    tidier_style="pipes"        # Use pipe syntax
)

# Generate transformation script
script = generate_transformation_script(
    generator, profile, schema, mapping_result;
    output_path="sdmx_output.csv",
    custom_transformations=Dict(
        "country" => "uppercase(country)",
        "value" => "round(value, digits=2)"
    )
)

# Validate generated script
validation = validate_generated_script(script)
println("Script quality: " * validation["overall_quality"])

# Preview script output
preview = preview_script_output(script; max_lines=10)
println(preview)
```

### Workflow Orchestration
End-to-end transformation pipelines:
```julia
# Create complete workflow
workflow = create_workflow(
    source_path="input_data.xlsx",
    target_schema_url=url,
    output_path="output_sdmx.csv";
    llm_provider=:ollama,
    llm_model="llama3"
)

# Execute workflow
result = execute_workflow(workflow)

if result.success
    println("Transformation complete!")
    println("Output saved to: " * result.output_path)
    println("Script saved to: " * result.script_path)
else
    println("Workflow failed: " * result.error_message)
end

# Access workflow artifacts
println("Mapping quality: " * string(result.mapping_quality))
println("Data coverage: " * string(result.data_coverage))
```

## Advanced Features

### Hierarchical Relationship Detection
```julia
# Detect parent-child relationships in data
hierarchy = detect_hierarchical_relationships(profile, schema)

for (parent, children) in hierarchy.parent_child_relationships
    println(parent * " -> " * join(children, ", "))
end
```

### Pattern Analysis
```julia
# Analyze value patterns against codelists
patterns = analyze_value_patterns(
    source_data.country,
    codelists[codelists.codelist_id .== "CL_GEO_PICT", :]
)

println("Exact matches: " * string(patterns["exact_matches"]))
println("Confidence: " * string(patterns["confidence_score"]))
```

### Transformation Steps Builder
```julia
# Build transformation pipeline
steps = build_transformation_steps(mapping_result, profile, schema)

for step in steps
    println(step.step_name * ": " * step.description)
    if step.operation_type == "pivot"
        println("  Pivot columns: " * join(step.parameters["id_cols"], ", "))
    end
end
```

## API Reference

### Setup Functions
- `setup_sdmx_llm(provider; kwargs...)` - Configure LLM provider
- `sdmx_aigenerate(prompt; provider)` - Query LLM with context

### Inference Functions
- `create_inference_engine(kwargs...)` - Create mapping engine
- `infer_advanced_mappings(engine, profile, schema, data)` - Run inference
- `fuzzy_match_score(str1, str2)` - Calculate string similarity
- `detect_hierarchical_relationships(profile, schema)` - Find hierarchies

### Generation Functions
- `create_script_generator(provider, model; kwargs...)` - Create generator
- `generate_transformation_script(generator, profile, schema, mapping)` - Generate code
- `validate_generated_script(script)` - Validate script quality
- `preview_script_output(script; max_lines)` - Preview results

### Workflow Functions
- `create_workflow(source, schema, output; kwargs...)` - Define workflow
- `execute_workflow(workflow)` - Run complete pipeline
- `build_transformation_steps(mapping, profile, schema)` - Build steps

### Excel Analysis
- `analyze_excel_structure(filepath)` - Analyze workbook structure

## Templates

SDMXLLM includes several transformation templates:

1. **Standard Transformation** - Basic column mapping and renaming
2. **Pivot Transformation** - Wide to long format conversion
3. **Excel Multi-Sheet** - Complex workbook handling
4. **Simple CSV** - Optimized for simple CSV files

Templates are automatically selected based on data complexity.

## Testing

Run the test suite:
```julia
using Pkg
Pkg.test("SDMXLLM")
```

All 61 tests should pass, covering:
- LLM provider configuration
- Advanced mapping inference
- Script generation
- Workflow orchestration
- Excel analysis
- Pattern recognition
- Validation logic

## Performance Tips

1. **Use local models (Ollama) for development** - Faster iteration, no API costs
2. **Cache LLM responses** - Reuse analysis results when possible
3. **Filter codelists by availability** - Reduce mapping search space
4. **Set appropriate thresholds** - Balance precision vs recall in fuzzy matching
5. **Use batch operations** - Process multiple files in one workflow

## Troubleshooting

### Google API Key Issues
The Google API key must be set before importing SDMXLLM:
```julia
# ✅ Correct
ENV["GOOGLE_API_KEY"] = "your-key"
using SDMXLLM

# ❌ Wrong - too late!
using SDMXLLM
ENV["GOOGLE_API_KEY"] = "your-key"
```

### Ollama Connection
Ensure Ollama is running:
```bash
ollama serve
ollama list  # Check available models
```

### API Rate Limits
For cloud providers, implement retry logic:
```julia
for attempt in 1:3
    try
        result = generate_transformation_script(...)
        break
    catch e
        if occursin("rate limit", string(e))
            sleep(2^attempt)
        else
            rethrow(e)
        end
    end
end
```

## Contributing

Contributions welcome! Please ensure:
1. All tests pass
2. New features include tests
3. LLM calls are mockable for testing
4. Documentation is updated

## License

MIT License - see [LICENSE](LICENSE) file for details.

## See Also

- [SDMX.jl](../SDMX.jl) - Core SDMX processing functionality
- [Tidier.jl](https://github.com/TidierOrg/Tidier.jl) - Data transformation framework
- [PromptingTools.jl](https://github.com/svilupp/PromptingTools.jl) - LLM integration backend