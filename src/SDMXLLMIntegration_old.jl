"""
Modern SDMX-LLM Integration using PromptingTools.jl

This module provides AI-powered analysis and transformation script generation using:
- All major LLM providers via PromptingTools.jl schemas
- Advanced Excel analysis (multi-sheet, header extraction, cell range detection)  
- Complex transformation scenarios (pivoting, wide-to-long, metadata extraction)
- Unified interface supporting Ollama, OpenAI, Anthropic, Mistral, Groq, etc.
"""

using PromptingTools, DataFrames, XLSX, CSV, Statistics
using EzXML, SDMX

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
    query_openai(config::LLMConfig, prompt::String, system_prompt::String) -> String

Queries OpenAI or Azure OpenAI API.
"""
function query_openai(config::LLMConfig, prompt::String, system_prompt::String)
    url = if config.provider == AZURE_OPENAI
        "$(config.base_url)/openai/deployments/$(config.model)/chat/completions?api-version=2023-12-01-preview"
    else
        "$(config.base_url)/chat/completions"
    end
    
    messages = []
    if !isempty(system_prompt)
        push!(messages, Dict("role" => "system", "content" => system_prompt))
    end
    push!(messages, Dict("role" => "user", "content" => prompt))
    
    payload = Dict(
        "model" => config.model,
        "messages" => messages,
        "temperature" => config.temperature,
        "max_tokens" => config.max_tokens
    )
    
    headers = ["Content-Type" => "application/json"]
    if config.provider == AZURE_OPENAI
        push!(headers, "api-key" => config.api_key)
    else
        push!(headers, "Authorization" => "Bearer $(config.api_key)")
    end
    
    response = HTTP.post(url, headers, JSON3.write(payload))
    result = JSON3.read(String(response.body))
    
    return String(result.choices[1].message.content)
end

"""
    query_anthropic(config::LLMConfig, prompt::String, system_prompt::String) -> String

Queries Anthropic Claude API.
"""
function query_anthropic(config::LLMConfig, prompt::String, system_prompt::String)
    url = "$(config.base_url)/messages"
    
    messages = [Dict("role" => "user", "content" => prompt)]
    
    payload = Dict(
        "model" => config.model,
        "max_tokens" => config.max_tokens,
        "temperature" => config.temperature,
        "messages" => messages
    )
    
    if !isempty(system_prompt)
        payload["system"] = system_prompt
    end
    
    headers = [
        "Content-Type" => "application/json",
        "x-api-key" => config.api_key,
        "anthropic-version" => "2023-06-01"
    ]
    
    response = HTTP.post(url, headers, JSON3.write(payload))
    result = JSON3.read(String(response.body))
    
    return String(result.content[1].text)
end

"""
    analyze_excel_structure(file_path::String) -> ExcelAnalysis

Performs comprehensive analysis of an Excel file to understand its structure,
identify data ranges, detect pivoting needs, and extract metadata.
"""
function analyze_excel_structure(file_path::String)
    if !isfile(file_path)
        error("File not found: $file_path")
    end
    
    xf = XLSX.readxlsx(file_path)
    sheets = Vector{SheetInfo}()
    
    for sheet_name in XLSX.sheetnames(xf)
        sheet = xf[sheet_name]
        
        # Get dimensions
        row_count, col_count = size(sheet[:])
        
        # Detect merged cells (simplified check)
        has_merged_cells = false  # We'll implement merged cell detection later if needed
        
        # Analyze cell ranges
        cell_ranges = detect_data_ranges(sheet, sheet_name)
        
        # Extract potential headers
        potential_headers = extract_potential_headers(sheet)
        
        # Extract metadata cells
        metadata_cells = extract_metadata_cells(sheet)
        
        sheet_info = SheetInfo(
            sheet_name,
            row_count,
            col_count,
            has_merged_cells,
            cell_ranges,
            potential_headers,
            metadata_cells
        )
        
        push!(sheets, sheet_info)
    end
    
    # Analyze pivoting needs
    pivot_analysis, pivoting_detected = analyze_pivoting_needs(sheets)
    
    # Calculate complexity score
    complexity_score = calculate_complexity_score(sheets, pivoting_detected)
    
    # Generate transformation hints
    transformation_hints = generate_transformation_hints(sheets, pivot_analysis)
    
    # Recommend primary sheet and ranges
    recommended_sheet, recommended_ranges = recommend_data_source(sheets)
    
    return ExcelAnalysis(
        file_path,
        sheets,
        recommended_sheet,
        recommended_ranges,
        pivoting_detected,
        pivot_analysis,
        complexity_score,
        transformation_hints
    )
end

"""
    detect_data_ranges(sheet::XLSX.Worksheet, sheet_name::String) -> Vector{CellRange}

Detects different types of data ranges within a sheet.
"""
function detect_data_ranges(sheet::XLSX.Worksheet, sheet_name::String)
    ranges = Vector{CellRange}()
    data_matrix = sheet[:]
    rows, cols = size(data_matrix)
    
    if rows == 0 || cols == 0
        return ranges
    end
    
    # Find header rows (typically first few rows with text)
    header_rows = []
    for r in 1:min(5, rows)
        row_data = data_matrix[r, :]
        non_missing = filter(!ismissing, row_data)
        if length(non_missing) > 0 && any(x -> isa(x, AbstractString), non_missing)
            push!(header_rows, r)
        end
    end
    
    if !isempty(header_rows)
        last_header = maximum(header_rows)
        push!(ranges, CellRange(sheet_name, 1, last_header, 1, cols, "Headers", "header"))
        
        # Data range starts after headers
        if last_header < rows
            # Find the actual data range by looking for non-empty cells
            data_start = last_header + 1
            data_end = rows
            
            # Trim empty rows from the end
            for r in rows:-1:data_start
                row_data = data_matrix[r, :]
                if any(!ismissing, row_data)
                    data_end = r
                    break
                end
            end
            
            if data_end >= data_start
                push!(ranges, CellRange(sheet_name, data_start, data_end, 1, cols, "Main Data", "data"))
            end
        end
    else
        # No clear headers, assume all data
        push!(ranges, CellRange(sheet_name, 1, rows, 1, cols, "All Data", "data"))
    end
    
    return ranges
end

"""
    extract_potential_headers(sheet::XLSX.Worksheet) -> Vector{String}

Extracts text that could be column headers or metadata labels.
"""
function extract_potential_headers(sheet::XLSX.Worksheet)
    headers = Vector{String}()
    data_matrix = sheet[:]
    rows, cols = size(data_matrix)
    
    # Look in first few rows for headers
    for r in 1:min(3, rows)
        for c in 1:cols
            cell_val = data_matrix[r, c]
            if !ismissing(cell_val) && isa(cell_val, AbstractString)
                header_text = strip(String(cell_val))
                if !isempty(header_text) && length(header_text) < 100  # Reasonable header length
                    push!(headers, header_text)
                end
            end
        end
    end
    
    return unique(headers)
end

"""
    extract_metadata_cells(sheet::XLSX.Worksheet) -> Dict{String, Any}

Extracts metadata information from cells that appear to contain labels and values.
"""
function extract_metadata_cells(sheet::XLSX.Worksheet)
    metadata = Dict{String, Any}()
    data_matrix = sheet[:]
    rows, cols = size(data_matrix)
    
    # Look for patterns like "Label: Value" or metadata in specific areas
    for r in 1:min(10, rows)  # Check first 10 rows for metadata
        for c in 1:min(cols-1, 5)  # Check first 5 columns for labels
            label_cell = data_matrix[r, c]
            value_cell = data_matrix[r, c+1]
            
            if !ismissing(label_cell) && isa(label_cell, AbstractString)
                label = strip(String(label_cell))
                
                # Check if this looks like a metadata label
                if occursin(r"(date|time|source|unit|period|frequency|country|region)", lowercase(label)) ||
                   endswith(label, ":") || 
                   length(split(label)) <= 3  # Short labels are likely metadata
                    
                    if !ismissing(value_cell)
                        metadata["$(r)_$(c)_$(label)"] = value_cell
                    end
                end
            end
        end
    end
    
    return metadata
end

"""
    analyze_pivoting_needs(sheets::Vector{SheetInfo}) -> Tuple{Union{Dict, Nothing}, Bool}

Analyzes whether the data needs pivoting (wide-to-long or long-to-wide transformation).
"""
function analyze_pivoting_needs(sheets::Vector{SheetInfo})
    pivot_analysis = Dict{String, Any}()
    pivoting_detected = false
    
    for sheet in sheets
        # Check if sheet has time periods as columns (wide format)
        time_columns = []
        for header in sheet.potential_headers
            # Look for year patterns, quarters, months, etc.
            if occursin(r"^(19|20)\d{2}$", header) ||  # Years like 2020, 2021
               occursin(r"^(19|20)\d{2}Q[1-4]$", header) ||  # Quarters like 2020Q1
               occursin(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", header) ||  # Month names
               occursin(r"^\d{4}-\d{2}$", header)  # YYYY-MM format
                push!(time_columns, header)
            end
        end
        
        if length(time_columns) > 1
            pivoting_detected = true
            pivot_analysis[sheet.name] = Dict(
                "type" => "wide_to_long",
                "time_columns" => time_columns,
                "pivot_columns" => time_columns,
                "description" => "Time periods are stored as separate columns - needs pivoting to long format"
            )
        end
        
        # Check for other pivot indicators
        # Multiple similar column names might indicate need for pivoting
        similar_patterns = Dict{String, Vector{String}}()
        for header in sheet.potential_headers
            # Group headers by base pattern
            base_pattern = replace(header, r"[0-9]+" => "X")  # Replace numbers with X
            base_pattern = replace(base_pattern, r"(Male|Female|M|F)" => "GENDER")
            base_pattern = replace(base_pattern, r"(Total|All)" => "TOTAL")
            
            if !haskey(similar_patterns, base_pattern)
                similar_patterns[base_pattern] = String[]
            end
            push!(similar_patterns[base_pattern], header)
        end
        
        # If we have patterns with multiple variations, might need pivoting
        for (pattern, headers) in similar_patterns
            if length(headers) > 2 && length(headers) < 20  # Reasonable range for pivot columns
                if !haskey(pivot_analysis, sheet.name)
                    pivoting_detected = true
                    pivot_analysis[sheet.name] = Dict(
                        "type" => "pattern_based",
                        "pattern" => pattern,
                        "columns" => headers,
                        "description" => "Multiple columns with similar patterns detected - may need pivoting"
                    )
                end
            end
        end
    end
    
    return isempty(pivot_analysis) ? nothing : pivot_analysis, pivoting_detected
end

"""
    calculate_complexity_score(sheets::Vector{SheetInfo}, pivoting_detected::Bool) -> Float64

Calculates a complexity score (0-1) for the Excel file transformation.
"""
function calculate_complexity_score(sheets::Vector{SheetInfo}, pivoting_detected::Bool)
    score = 0.0
    
    # Base complexity from number of sheets
    score += min(length(sheets) * 0.1, 0.3)
    
    for sheet in sheets
        # Complexity from merged cells
        if sheet.has_merged_cells
            score += 0.2
        end
        
        # Complexity from number of cell ranges
        score += min(length(sheet.cell_ranges) * 0.05, 0.2)
        
        # Complexity from metadata extraction needs
        score += min(length(sheet.metadata_cells) * 0.02, 0.1)
    end
    
    # Pivoting adds significant complexity
    if pivoting_detected
        score += 0.3
    end
    
    return min(score, 1.0)
end

"""
    generate_transformation_hints(sheets::Vector{SheetInfo}, pivot_analysis::Union{Dict, Nothing}) -> Vector{String}

Generates human-readable hints about the transformation complexity.
"""
function generate_transformation_hints(sheets::Vector{SheetInfo}, pivot_analysis::Union{Dict, Nothing})
    hints = String[]
    
    if length(sheets) > 1
        push!(hints, "Multiple sheets detected - may need to combine or select specific sheet")
    end
    
    for sheet in sheets
        if sheet.has_merged_cells
            push!(hints, "Sheet '$(sheet.name)' has merged cells - may need special handling")
        end
        
        if length(sheet.metadata_cells) > 0
            push!(hints, "Sheet '$(sheet.name)' contains metadata that may need to be extracted")
        end
    end
    
    if pivot_analysis !== nothing
        for (sheet_name, analysis) in pivot_analysis
            if analysis["type"] == "wide_to_long"
                push!(hints, "Sheet '$sheet_name' has time periods as columns - needs pivot_longer transformation")
            else
                push!(hints, "Sheet '$sheet_name' may need pivoting based on column patterns")
            end
        end
    end
    
    return hints
end

"""
    recommend_data_source(sheets::Vector{SheetInfo}) -> Tuple{String, Vector{CellRange}}

Recommends which sheet and ranges to use as the primary data source.
"""
function recommend_data_source(sheets::Vector{SheetInfo})
    if isempty(sheets)
        return "", CellRange[]
    end
    
    # Score each sheet based on data content
    sheet_scores = []
    for sheet in sheets
        score = 0.0
        
        # Prefer sheets with actual data ranges
        data_ranges = filter(r -> r.data_type == "data", sheet.cell_ranges)
        score += length(data_ranges) * 10
        
        # Prefer sheets with reasonable size
        if 10 <= sheet.row_count <= 10000
            score += 5
        end
        
        # Prefer sheets with multiple columns
        if sheet.col_count >= 3
            score += 3
        end
        
        # Avoid sheets with too much metadata
        if length(sheet.metadata_cells) > 10
            score -= 2
        end
        
        push!(sheet_scores, (sheet.name, score, sheet))
    end
    
    # Sort by score and return best sheet
    sort!(sheet_scores, by=x->x[2], rev=true)
    best_sheet = sheet_scores[1][3]
    
    # Return data ranges from best sheet
    data_ranges = filter(r -> r.data_type == "data", best_sheet.cell_ranges)
    
    return best_sheet.name, data_ranges
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

# =================== BACKWARD COMPATIBILITY ===================

"""
Legacy function for backward compatibility. Use generate_transformation_script instead.
"""
function build_transformation_prompt(args...)
    @warn "build_transformation_prompt is deprecated. Use generate_transformation_script instead."
    return "# This function has been deprecated. Please use the new generate_transformation_script function."
end
    
    prompt = """Please generate a Julia transformation script using Tidier.jl to convert the source data into SDMX-CSV format.

## Source Data Analysis
File: $(source_profile.file_path) ($(source_profile.file_type))
Dimensions: $(source_profile.row_count) rows × $(source_profile.column_count) columns
Data Quality: $(round(source_profile.data_quality_score * 100, digits=1))%

### Source Columns:
"""
    
    for col in source_profile.columns
        prompt *= "- **$(col.name)**: $(col.type)"
        
        if col.is_temporal
            prompt *= " (Time/Date: $(col.temporal_format))"
        elseif col.is_categorical
            prompt *= " (Categorical: $(col.unique_count) categories)"
        elseif col.numeric_stats !== nothing
            prompt *= " (Numeric: $(col.numeric_stats.min) - $(col.numeric_stats.max))"
        end
        
        if col.missing_count > 0
            prompt *= " [$(col.missing_count) missing values]"
        end
        
        prompt *= "\n"
    end
    
    # Add Excel analysis if available
    if excel_analysis !== nothing
        prompt *= "\n### Excel Structure Analysis\n"
        prompt *= "Complexity Score: $(round(excel_analysis.complexity_score, digits=2))\n"
        prompt *= "Recommended Sheet: $(excel_analysis.recommended_sheet)\n"
        
        if excel_analysis.pivoting_detected
            prompt *= "**PIVOTING REQUIRED**: $(excel_analysis.pivot_analysis)\n"
        end
        
        if !isempty(excel_analysis.transformation_hints)
            prompt *= "\nTransformation Hints:\n"
            for hint in excel_analysis.transformation_hints
                prompt *= "- $hint\n"
            end
        end
        
        # Include sheet details
        for sheet in excel_analysis.sheets
            if sheet.name == excel_analysis.recommended_sheet
                prompt *= "\n**Primary Sheet Details ($(sheet.name))**:\n"
                prompt *= "- Dimensions: $(sheet.row_count)×$(sheet.col_count)\n"
                prompt *= "- Merged cells: $(sheet.has_merged_cells)\n"
                
                if !isempty(sheet.metadata_cells)
                    prompt *= "- Metadata found: $(keys(sheet.metadata_cells))\n"
                end
                
                for range in sheet.cell_ranges
                    prompt *= "- $(range.description): rows $(range.start_row)-$(range.end_row), cols $(range.start_col)-$(range.end_col)\n"
                end
                break
            end
        end
    end
    
    # Add target schema information
    prompt *= "\n## Target SDMX Schema\n"
    prompt *= "Dataflow: $(target_schema.dataflow_info.id) - $(target_schema.dataflow_info.name)\n"
    prompt *= "Agency: $(target_schema.dataflow_info.agency)\n\n"
    
    prompt *= "### Required Columns:\n"
    required_cols = get_required_columns(target_schema)
    for col in required_cols
        # Find the column info
        if col in target_schema.dimensions.dimension_id
            dim_info = filter(row -> row.dimension_id == col, target_schema.dimensions)[1]
            prompt *= "- **$col** (Dimension): $(dim_info.codelist_id)\n"
        elseif target_schema.time_dimension !== nothing && col == target_schema.time_dimension.dimension_id
            prompt *= "- **$col** (Time Dimension): $(target_schema.time_dimension.data_type)\n"
        elseif col in target_schema.measures.measure_id
            measure_info = filter(row -> row.measure_id == col, target_schema.measures)[1]
            prompt *= "- **$col** (Measure): $(measure_info.data_type)\n"
        elseif col in target_schema.attributes.attribute_id
            attr_info = filter(row -> row.attribute_id == col, target_schema.attributes)[1]
            prompt *= "- **$col** (Attribute): $(attr_info.assignment_status)\n"
        end
    end
    
    optional_cols = get_optional_columns(target_schema)
    if !isempty(optional_cols)
        prompt *= "\n### Optional Columns:\n"
        for col in optional_cols
            attr_info = filter(row -> row.attribute_id == col, target_schema.attributes)[1]
            prompt *= "- **$col** (Optional Attribute): $(attr_info.codelist_id)\n"
        end
    end
    
    # Add column mappings if available
    if !isempty(column_mappings)
        prompt *= "\n### Suggested Column Mappings:\n"
        for (target_col, source_cols) in column_mappings
            prompt *= "- **$target_col** ← $(join(source_cols, " | "))\n"
        end
    end
    
    # Add specific instructions
    prompt *= "\n## Instructions\n"
    prompt *= "1. Generate complete Julia code using Tidier.jl to transform the source data\n"
    prompt *= "2. Handle the file reading (CSV.jl or XLSX.jl) based on source file type\n"
    prompt *= "3. Include all necessary data transformations, pivoting, and cleaning\n"
    prompt *= "4. Ensure all required SDMX columns are present with correct data types\n"
    prompt *= "5. Add validation checks for data quality and SDMX compliance\n"
    prompt *= "6. Include error handling and informative output messages\n"
    prompt *= "7. Save the result as SDMX-compliant CSV\n"
    
    if excel_analysis !== nothing && excel_analysis.pivoting_detected
        prompt *= "\n**IMPORTANT**: This data requires pivoting transformation. Use @pivot_longer or @pivot_wider as needed.\n"
    end
    
    prompt *= "\nProvide the complete Julia script with clear comments explaining each transformation step."
    
    return prompt
end