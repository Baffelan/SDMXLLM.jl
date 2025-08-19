"""
Modern SDMX-LLM Integration using PromptingTools.jl

This module provides AI-powered analysis and transformation script generation using:
- All major LLM providers via PromptingTools.jl schemas
- Advanced Excel analysis (multi-sheet, header extraction, cell range detection)  
- Complex transformation scenarios (pivoting, wide-to-long, metadata extraction)
- Unified interface supporting Ollama, OpenAI, Anthropic, Mistral, Groq, etc.
"""

# Dependencies loaded at package level

export ExcelAnalysis, SheetInfo, CellRange, analyze_excel_structure,
       generate_transformation_script, extract_excel_metadata, detect_data_ranges, 
       analyze_pivoting_needs, sdmx_aigenerate, sdmx_aiextract,
       infer_sdmx_column_mappings, SDMX_PROVIDERS

# =================== PROVIDER CONFIGURATION ===================

"""
Supported LLM providers with their PromptingTools.jl schemas and capabilities
"""
const SDMX_PROVIDERS = Dict{Symbol, NamedTuple}(
    :openai => (
        schema = PromptingTools.OpenAISchema(),
        supports = [:generation, :extraction, :embedding, :classification],
        default_model = "gpt-4o",
        description = "OpenAI GPT models"
    ),
    :anthropic => (
        schema = PromptingTools.AnthropicSchema(),
        supports = [:generation, :extraction],
        default_model = "claude-3-5-sonnet-20241022",
        description = "Anthropic Claude models"
    ),
    :ollama => (
        schema = PromptingTools.OllamaSchema(),
        supports = [:generation, :embedding],
        default_model = "qwen3:0.6b",
        description = "Local Ollama models"
    ),
    :mistral => (
        schema = PromptingTools.MistralOpenAISchema(),
        supports = [:generation, :extraction, :embedding],
        default_model = "mistral-large",
        description = "Mistral AI models"
    ),
    :groq => (
        schema = PromptingTools.GroqOpenAISchema(),
        supports = [:generation, :extraction],
        default_model = "llama-3.1-70b-versatile",
        description = "Groq fast inference"
    ),
    :together => (
        schema = PromptingTools.TogetherOpenAISchema(),
        supports = [:generation, :extraction, :embedding],
        default_model = "meta-llama/Llama-3-70b-chat-hf",
        description = "Together AI models"
    ),
    :fireworks => (
        schema = PromptingTools.FireworksOpenAISchema(),
        supports = [:generation, :extraction, :embedding],
        default_model = "accounts/fireworks/models/llama-v3-70b-instruct",
        description = "Fireworks AI models"
    ),
    :databricks => (
        schema = PromptingTools.DatabricksOpenAISchema(),
        supports = [:generation, :extraction, :embedding],
        default_model = "databricks-meta-llama-3-70b-instruct",
        description = "Databricks models"
    ),
    :google => (
        schema = PromptingTools.GoogleSchema(),
        supports = [:generation],
        default_model = "gemini-1.5-pro",
        description = "Google Gemini models"
    )
)

"""
    CellRange

Represents a range of cells in an Excel sheet.
"""
struct CellRange
    sheet_name::String
    start_row::Int
    end_row::Int
    start_col::Int
    end_col::Int
    description::String
    data_type::String  # "header", "data", "metadata", "summary"
end

"""
    SheetInfo

Information about a single Excel sheet.
"""
struct SheetInfo
    name::String
    row_count::Int
    col_count::Int
    has_merged_cells::Bool
    cell_ranges::Vector{CellRange}
    potential_headers::Vector{String}
    metadata_cells::Dict{String, Any}  # Cell references to metadata values
end

"""
    ExcelAnalysis

Comprehensive analysis of an Excel file structure.
"""
struct ExcelAnalysis
    file_path::String
    sheets::Vector{SheetInfo}
    recommended_sheet::String
    recommended_ranges::Vector{CellRange}
    pivoting_detected::Bool
    pivot_analysis::Union{Dict, Nothing}
    complexity_score::Float64
    transformation_hints::Vector{String}
end

# =================== UNIFIED SDMX LLM INTERFACE ===================

"""
    sdmx_aigenerate(prompt::String; provider::Symbol=:ollama, model::String="", kwargs...)

Generate text using specified provider with SDMX-optimized settings.
Uses PromptingTools.jl schemas for all supported providers.
"""
function sdmx_aigenerate(prompt::String; provider::Symbol=:ollama, model::String="", kwargs...)
    @assert !isempty(prompt) "Prompt cannot be empty"
    
    provider_info = get(SDMX_PROVIDERS, provider, nothing)
    @assert provider_info !== nothing "Unsupported provider: $provider. Available: $(keys(SDMX_PROVIDERS))"
    @assert :generation in provider_info.supports "Provider $provider doesn't support text generation"
    
    # Use default model if none specified
    model_name = isempty(model) ? provider_info.default_model : model
    
    return aigenerate(provider_info.schema, prompt; model=model_name, kwargs...)
end

"""
    sdmx_aiextract(return_type::Type, prompt::String; provider::Symbol=:openai, model::String="", kwargs...)

Extract structured data using specified provider. Falls back to text generation for unsupported providers.
"""
function sdmx_aiextract(return_type::Type, prompt::String; provider::Symbol=:openai, model::String="", kwargs...)
    @assert !isempty(prompt) "Prompt cannot be empty"
    
    provider_info = get(SDMX_PROVIDERS, provider, nothing)
    @assert provider_info !== nothing "Unsupported provider: $provider"
    
    model_name = isempty(model) ? provider_info.default_model : model
    
    if :extraction in provider_info.supports
        return aiextract(provider_info.schema, prompt; model=model_name, return_type=return_type, kwargs...)
    else
        @warn "Provider $provider doesn't support extraction, falling back to generation"
        # Format prompt for manual extraction
        structured_prompt = """
        $prompt
        
        Please format your response as valid Julia data that can be parsed into the requested type: $return_type
        """
        result = aigenerate(provider_info.schema, structured_prompt; model=model_name, kwargs...)
        return result.content  # Return raw content for manual parsing
    end
end

# =================== SDMX-SPECIFIC LLM FUNCTIONS ===================

"""
    infer_sdmx_column_mappings(source_columns::Vector{String}, target_schema; 
                              provider::Symbol=:ollama, model::String="")

Infer optimal column mappings from source data to SDMX schema using LLM analysis.
"""
function infer_sdmx_column_mappings(source_columns::Vector{String}, target_schema; 
                                   provider::Symbol=:ollama, model::String="")
    @assert !isempty(source_columns) "Source columns cannot be empty"
    @assert haskey(target_schema, :dimensions) || isa(target_schema, DataflowSchema) "Invalid target schema"
    
    # Extract schema information
    if isa(target_schema, DataflowSchema)
        dimensions = target_schema.dimensions.dimension_id
        measures = target_schema.measures.measure_id
    else
        dimensions = target_schema[:dimensions]
        measures = get(target_schema, :measures, ["OBS_VALUE"])
    end
    
    prompt = """
    You are an SDMX (Statistical Data and Metadata eXchange) expert. 
    
    Source data columns: $(join(source_columns, ", "))
    
    Target SDMX dimensions: $(join(dimensions, ", "))
    Target SDMX measures: $(join(measures, ", "))
    
    Suggest optimal column mappings from source to SDMX schema.
    Focus on standard SDMX patterns:
    - Geographic areas → GEO_PICT 
    - Time periods → TIME_PERIOD
    - Statistical indicators → INDICATOR
    - Observed values → OBS_VALUE
    - Units of measure → UNIT_MEASURE
    
    For each mapping, provide:
    1. Source column → Target SDMX dimension/measure
    2. Confidence level (high/medium/low)
    3. Required transformations or notes
    
    Be specific and actionable.
    """
    
    return sdmx_aigenerate(prompt; provider=provider, model=model)
end

"""
    generate_transformation_script(mappings::String, schema_info; 
                                 provider::Symbol=:ollama, model::String="", 
                                 style::Symbol=:tidier, excel_analysis::Union{ExcelAnalysis, Nothing}=nothing)

Generate Julia transformation script for SDMX data conversion using modern LLM providers.
"""
function generate_transformation_script(mappings::String, schema_info; 
                                       provider::Symbol=:ollama, model::String="", 
                                       style::Symbol=:tidier, excel_analysis::Union{ExcelAnalysis, Nothing}=nothing)
    @assert !isempty(mappings) "Column mappings cannot be empty"
    @assert style in [:tidier, :dataframes, :mixed] "Style must be :tidier, :dataframes, or :mixed"
    
    style_instruction = if style == :tidier
        "Use Tidier.jl syntax (@select, @mutate, @filter, @pivot_longer, etc.)"
    elseif style == :dataframes
        "Use DataFrames.jl syntax (select, transform, filter, etc.)"
    else
        "Use a mix of DataFrames.jl and Tidier.jl as appropriate"
    end
    
    # Add Excel analysis context if available
    excel_context = if excel_analysis !== nothing
        """
        
        Excel Analysis Context:
        - Complexity Score: $(excel_analysis.complexity_score)
        - Pivoting Detected: $(excel_analysis.pivoting_detected)
        - Recommended Sheet: $(excel_analysis.recommended_sheet)
        - Transformation Hints: $(join(excel_analysis.transformation_hints, "; "))
        """
    else
        ""
    end
    
    prompt = """
    Generate a complete Julia script to transform data for SDMX compliance.
    
    Column mappings identified:
    $mappings
    
    Target SDMX schema information:
    $(typeof(schema_info) == DataflowSchema ? "Dataflow: $(schema_info.dataflow_info.name)" : string(schema_info))
    $excel_context
    
    Requirements:
    1. $style_instruction
    2. Read source data (assume CSV format unless Excel context suggests otherwise)
    3. Apply all necessary transformations based on the mappings
    4. Handle missing data appropriately with @assert statements
    5. Validate output against SDMX requirements  
    6. Export clean SDMX-CSV format
    7. Include comprehensive error handling
    8. Add clear comments explaining each transformation step
    
    Generate complete, executable Julia code only.
    """
    
    return sdmx_aigenerate(prompt; provider=provider, model=model)
end

# Include the Excel analysis functions from the original file
# (analyze_excel_structure, detect_data_ranges, etc. would be included here)
# For brevity, showing the key refactored functions above