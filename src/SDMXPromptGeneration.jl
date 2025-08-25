"""
Comprehensive SDMX Prompt Generation for LLM-Assisted Transformation Scripts

This module creates rich, detailed prompts that combine SDMX schema context with 
source data analysis to generate high-quality transformation script drafts.
"""

# Dependencies loaded at package level


# =================== PROMPT TEMPLATES ===================

"""
    PromptTemplate

Structure for storing reusable prompt templates for different Julia syntax styles.
"""
struct PromptTemplate
    name::String
    description::String
    syntax_style::Symbol  # :tidier, :dataframes, :mixed
    header_template::String
    requirements_template::String
    validation_template::String
    output_template::String
end

"""
    TransformationScenario

Structure for different transformation scenarios (Excel ranges, pivoting, etc.)
"""
struct TransformationScenario
    name::String
    description::String
    specific_instructions::String
    additional_requirements::Vector{String}
    example_code_snippet::String
end

"""
    ComprehensivePrompt

Complete prompt structure combining all context and templates.
"""
struct ComprehensivePrompt
    sdmx_context::String
    source_analysis::String
    transformation_requirements::String
    code_templates::String
    validation_rules::String
    full_prompt::String
end

# =================== BUILT-IN TEMPLATES ===================

const TIDIER_TEMPLATE = PromptTemplate(
    "Tidier.jl Style",
    "Modern Julia data transformation using Tidier.jl syntax",
    :tidier,
    """
    You are an expert Julia programmer specializing in SDMX data transformations using Tidier.jl.
    Generate a complete, executable Julia script that transforms source data into SDMX-compliant format.
    
    Use Tidier.jl syntax throughout:
    - @select() for column selection
    - @mutate() for column creation and transformation
    - @filter() for data filtering
    - @pivot_longer() / @pivot_wider() for reshaping
    - @group_by() and @summarise() for aggregation
    - @arrange() for sorting
    """,
    """
    ## SCRIPT REQUIREMENTS
    1. Use only Tidier.jl syntax - no base DataFrames.jl operations
    2. Include comprehensive error handling with @assert statements
    3. Add detailed comments explaining each transformation step
    4. Implement robust data validation throughout the pipeline
    5. Handle missing data appropriately
    6. Create informative error messages for data quality issues
    """,
    """
    ## VALIDATION PATTERNS
    ```julia
    # Validate input data structure
    @assert ncol(data) >= expected_cols "Source data missing required columns"
    @assert nrow(data) > 0 "Source data cannot be empty"
    
    # Validate SDMX compliance
    @assert all(required_cols .∈ Ref(names(transformed_data))) "Missing required SDMX columns"
    @assert all(!ismissing(transformed_data.GEO_PICT)) "Geographic dimension cannot have missing values"
    ```
    """,
    """
    ## OUTPUT REQUIREMENTS
    - Export final dataset as SDMX-compliant CSV
    - Include data quality summary report
    - Provide transformation log with statistics
    - Save intermediate datasets for debugging if needed
    """
)

const DATAFRAMES_TEMPLATE = PromptTemplate(
    "DataFrames.jl Style", 
    "Traditional Julia data manipulation using DataFrames.jl",
    :dataframes,
    """
    You are an expert Julia programmer specializing in SDMX data transformations using DataFrames.jl.
    Generate a complete, executable Julia script that transforms source data into SDMX-compliant format.
    
    Use DataFrames.jl syntax throughout:
    - select() for column selection
    - transform() for column creation
    - filter() for data filtering
    - stack() / unstack() for reshaping
    - groupby() and combine() for aggregation
    - sort() for sorting
    """,
    """
    ## SCRIPT REQUIREMENTS
    1. Use DataFrames.jl syntax with clear, readable operations
    2. Include comprehensive error handling with @assert statements
    3. Add detailed comments explaining each transformation step
    4. Implement robust data validation throughout the pipeline
    5. Handle missing data appropriately using DataFrames patterns
    6. Create informative error messages for data quality issues
    """,
    """
    ## VALIDATION PATTERNS
    ```julia
    # Validate input data structure
    @assert ncol(data) >= expected_cols "Source data missing required columns"
    @assert nrow(data) > 0 "Source data cannot be empty"
    
    # Validate SDMX compliance
    required_cols = ["GEO_PICT", "TIME_PERIOD", "INDICATOR", "OBS_VALUE"]
    @assert all(col -> col ∈ names(transformed_data), required_cols) "Missing required SDMX columns"
    ```
    """,
    """
    ## OUTPUT REQUIREMENTS
    - Export final dataset as SDMX-compliant CSV using CSV.write()
    - Include data quality summary report
    - Provide transformation log with statistics
    - Save intermediate datasets for debugging if needed
    """
)

const MIXED_TEMPLATE = PromptTemplate(
    "Mixed Style",
    "Combine Tidier.jl and DataFrames.jl as appropriate for each operation",
    :mixed,
    """
    You are an expert Julia programmer specializing in SDMX data transformations.
    Generate a complete, executable Julia script that transforms source data into SDMX-compliant format.
    
    Use the most appropriate syntax for each operation:
    - Tidier.jl for complex transformations and pipelines
    - DataFrames.jl for performance-critical operations
    - Choose the clearest syntax for each specific task
    """,
    """
    ## SCRIPT REQUIREMENTS
    1. Mix Tidier.jl and DataFrames.jl syntax optimally
    2. Include comprehensive error handling with @assert statements
    3. Add detailed comments explaining syntax choices
    4. Implement robust data validation throughout the pipeline
    5. Handle missing data using best practices from both packages
    6. Optimize for both readability and performance
    """,
    """
    ## VALIDATION PATTERNS
    ```julia
    # Use most appropriate validation style for each check
    @assert ncol(data) >= expected_cols "Source data missing required columns"
    
    # Tidier-style validation for pipelines
    data |> @assert(nrow(_) > 0, "Data cannot be empty after filtering")
    
    # DataFrames-style for complex conditions
    @assert all(x -> x ∈ valid_codes, data.GEO_PICT) "Invalid geographic codes found"
    ```
    """,
    """
    ## OUTPUT REQUIREMENTS
    - Export using most appropriate method (CSV.jl, Tidier export, etc.)
    - Include comprehensive data quality report
    - Provide clear transformation documentation
    - Balance performance and readability in final code
    """
)

# =================== TRANSFORMATION SCENARIOS ===================

const EXCEL_SCENARIO = TransformationScenario(
    "Excel Range Selection",
    "Handle Excel files with specific sheet and range selection",
    """
    ## EXCEL-SPECIFIC HANDLING
    The source data comes from an Excel file with the following characteristics:
    - Multi-sheet workbook structure detected
    - Specific data range selection required
    - Potential header rows and metadata cells to skip
    
    Include these Excel-specific steps:
    1. Read from the correct sheet and range
    2. Handle merged cells appropriately  
    3. Skip metadata rows at the top
    4. Validate data starts at the expected location
    """,
    [
        "Use XLSX.jl for robust Excel reading",
        "Implement range validation before processing",
        "Handle potential formatting issues in Excel data",
        "Add checks for unexpected sheet structure changes"
    ],
    """
    # Excel reading with range selection
    xlsx_file = XLSX.readxlsx("source_file.xlsx")
    data = DataFrame(XLSX.gettable(xlsx_file["Sheet1"], "A5:E100"))
    @assert ncol(data) == expected_cols "Excel range returned unexpected columns"
    """
)

const CSV_SCENARIO = TransformationScenario(
    "CSV Processing",
    "Handle standard CSV file transformations",
    """
    ## CSV FILE HANDLING
    The source data is in CSV format with the following considerations:
    - Standard delimiter and encoding
    - Potential header row variations
    - Missing data patterns to handle
    
    Include these CSV-specific steps:
    1. Robust CSV reading with error handling
    2. Header validation and normalization
    3. Missing data detection and treatment
    4. Encoding and delimiter validation
    """,
    [
        "Use CSV.jl with appropriate parsing options",
        "Validate CSV structure before processing", 
        "Handle various missing data representations",
        "Check for delimiter and encoding issues"
    ],
    """
    # Robust CSV reading
    data = CSV.read("source_file.csv", DataFrame; 
                   header=1, normalizenames=true, stringtype=String)
    @assert nrow(data) > 0 "CSV file appears to be empty"
    """
)

const PIVOTING_SCENARIO = TransformationScenario(
    "Data Pivoting",
    "Transform between wide and long format data structures",
    """
    ## PIVOTING REQUIREMENTS
    The data transformation requires pivoting operations:
    - Source data structure needs reshaping
    - Multiple value columns to be converted to indicator/value pairs
    - Maintain all dimensional information during pivoting
    
    Key pivoting considerations:
    1. Identify value columns that need to become indicators
    2. Preserve all dimensional columns during pivot
    3. Create proper INDICATOR and OBS_VALUE columns
    4. Validate data integrity after pivoting
    """,
    [
        "Use appropriate pivoting function for the syntax style",
        "Ensure all dimension columns are preserved",
        "Validate pivot results for completeness",
        "Handle missing values appropriately during pivot"
    ],
    """
    # Tidier.jl pivoting example
    transformed_data = data |>
        @pivot_longer(cols = ["GDP_Value", "Population"], 
                     names_to = "INDICATOR", 
                     values_to = "OBS_VALUE")
    """
)

const CODE_MAPPING_SCENARIO = TransformationScenario(
    "Code Mapping",
    "Map source values to SDMX codelist codes",
    """
    ## CODE MAPPING REQUIREMENTS
    The transformation requires mapping source values to SDMX standard codes:
    - Country names to geographic codes
    - Indicator names to standard indicator codes
    - Units to standard unit codes
    - Time formats to SDMX time periods
    
    Code mapping approach:
    1. Create mapping dictionaries from SDMX codelists
    2. Implement fuzzy matching for approximate matches
    3. Handle unmapped values with appropriate warnings
    4. Validate all mapped codes exist in target codelists
    """,
    [
        "Build comprehensive mapping dictionaries",
        "Implement fallback strategies for unmapped values",
        "Add validation that all mapped codes are valid",
        "Provide clear error messages for mapping failures"
    ],
    """
    # Code mapping example
    geo_mapping = Dict(
        "United States" => "US",
        "China" => "CN",
        "Australia" => "AU"
    )
    
    data = data |> @mutate(GEO_PICT = geo_mapping[Country_Name])
    @assert all(!ismissing(data.GEO_PICT)) "Some countries could not be mapped"
    """
)

# =================== PROMPT GENERATION FUNCTIONS ===================

"""
    create_prompt_template(style::Symbol) -> PromptTemplate

Create a prompt template for the specified syntax style.
"""
function create_prompt_template(style::Symbol)
    @assert style ∈ [:tidier, :dataframes, :mixed] "Style must be :tidier, :dataframes, or :mixed"
    
    if style == :tidier
        return TIDIER_TEMPLATE
    elseif style == :dataframes
        return DATAFRAMES_TEMPLATE
    else
        return MIXED_TEMPLATE
    end
end

"""
    build_sdmx_context_section(sdmx_context::SDMXStructuralContext) -> String

Build the SDMX schema context section of the prompt with rich codelist information.
"""
function build_sdmx_context_section(sdmx_context::SDMXStructuralContext)
    sections = String[]
    push!(sections, "## COMPREHENSIVE SDMX SCHEMA CONTEXT")
    push!(sections, "")
    push!(sections, "### Dataflow Information")
    push!(sections, "- ID: " * string(sdmx_context.dataflow_info.id))
    push!(sections, "- Name: " * string(sdmx_context.dataflow_info.name))
    push!(sections, "- Agency: " * string(sdmx_context.dataflow_info.agency))
    
    # Handle required columns separately to avoid Unicode comma issues
    required_cols_str = join(sdmx_context.required_columns, ", ")
    push!(sections, "- Required Columns: " * required_cols_str)
    push!(sections, "")
    push!(sections, "### Dimensions with Codelists")
    
    # Add detailed information for each dimension
    for row in eachrow(sdmx_context.dimensions)
        dim_id = row.dimension_id
        codelist_id = row.codelist_id
        
        push!(sections, "")
        push!(sections, "**" * string(dim_id) * "**")
        
        if haskey(sdmx_context.codelist_summary, codelist_id)
            cl_info = sdmx_context.codelist_summary[codelist_id]
            push!(sections, "- Codelist: " * string(codelist_id))
            push!(sections, "- Total codes: " * string(cl_info.total_codes))
            push!(sections, "- Sample codes: " * join(cl_info.sample_codes, ", "))
            push!(sections, "- Hierarchical: " * string(cl_info.has_hierarchy))
            
            # Add available codes if we have them
            if haskey(sdmx_context.available_codes, dim_id)
                available = sdmx_context.available_codes[dim_id]
                available_sample = join(first(available, min(5, length(available))), ", ")
                push!(sections, "- Available in data: " * string(length(available)) * " codes")
                push!(sections, "- Available sample: " * available_sample)
            end
        else
            push!(sections, "- Codelist: " * string(codelist_id) * " (details not available)")
        end
    end
    
    # Add measures information
    if nrow(sdmx_context.measures) > 0
        push!(sections, "")
        push!(sections, "### Measures")
        for row in eachrow(sdmx_context.measures)
            measure_desc = get(row, :description, "Observed values")
            push!(sections, "- **" * string(row.measure_id) * "**: " * string(measure_desc))
        end
    end
    
    # Add time dimension information
    if sdmx_context.time_dimension !== nothing
        push!(sections, "")
        push!(sections, "### Time Dimension")
        push!(sections, "- Time dimension: " * string(sdmx_context.time_dimension.dimension_id))
        if sdmx_context.time_coverage !== nothing
            push!(sections, "- Coverage: " * string(sdmx_context.time_coverage))
        end
    end
    
    return join(sections, "\n")
end

"""
    build_source_analysis_section(source_context::DataSourceContext) -> String

Build the source data analysis section without exposing actual data.
"""
function build_source_analysis_section(source_context::DataSourceContext)
    sections = String[]
    push!(sections, "## SOURCE DATA ANALYSIS")
    push!(sections, "")
    push!(sections, "### File Information")
    push!(sections, "- File: " * string(source_context.file_info.path))
    push!(sections, "- Size: " * string(source_context.file_info.size_mb) * " MB")
    push!(sections, "- Dimensions: " * string(source_context.source_profile.row_count) * " rows × " * string(source_context.source_profile.column_count) * " columns")
    push!(sections, "")
    
    # Add pattern detection summary
    push!(sections, "### Detected Patterns")
    push!(sections, "- Time patterns: " * string(length(source_context.time_patterns)) * " (temporal columns detected)")
    push!(sections, "- Geographic patterns: " * string(length(source_context.geographic_patterns)) * " (country/region columns)")
    push!(sections, "- Value patterns: " * string(length(source_context.value_patterns)) * " (numeric data columns)")
    push!(sections, "- Hierarchical patterns: " * string(length(source_context.hierarchical_patterns)) * " (parent-child relationships)")
    push!(sections, "")
    
    # Add column analysis
    push!(sections, "### Column Analysis (Structure Only - No Actual Data)")
    for (col_name, pattern) in source_context.column_patterns
        col_desc = "- **" * string(col_name) * "**: " * string(pattern.data_type)
        
        if pattern.is_temporal
            col_desc *= " (Temporal data detected)"
        elseif pattern.is_categorical
            col_desc *= " (Categorical: " * string(pattern.categorical_info.categories) * " categories)"
        elseif pattern.is_numeric
            col_desc *= " (Numeric data)"
        end
        
        if pattern.missing_ratio > 0
            missing_pct = round(pattern.missing_ratio*100, digits=1)
            col_desc *= " [" * string(missing_pct) * "% missing]"
        end
        
        push!(sections, col_desc)
    end
    
    # Add data shape analysis
    push!(sections, "")
    push!(sections, "### Data Shape Analysis")
    push!(sections, "- Wide format: " * string(source_context.data_shape.is_wide_format))
    push!(sections, "- Needs pivoting: " * string(source_context.data_shape.needs_pivoting))
    push!(sections, "- Complexity score: " * string(source_context.data_shape.complexity_score))
    
    if source_context.data_shape.needs_pivoting
        pivot_candidates_str = join(source_context.data_shape.pivot_candidates, ", ")
        push!(sections, "- Pivot candidates: " * pivot_candidates_str)
    end
    
    # Add Excel analysis if applicable
    if source_context.excel_structure !== nothing
        push!(sections, "")
        push!(sections, "### Excel Structure Analysis")
        push!(sections, "- Recommended sheet: " * string(source_context.excel_structure.recommended_sheet))
        push!(sections, "- Complexity score: " * string(source_context.excel_structure.complexity_score))
        hints_str = join(source_context.excel_structure.transformation_hints, "; ")
        push!(sections, "- Transformation hints: " * hints_str)
    end
    
    return join(sections, "\n")
end

"""
    build_code_mapping_section(sdmx_context::SDMXStructuralContext, source_context::DataSourceContext) -> String

Build specific code mapping instructions based on detected patterns and available codelists.
"""
function build_code_mapping_section(sdmx_context::SDMXStructuralContext, source_context::DataSourceContext)
    sections = [
        "## CODE MAPPING REQUIREMENTS",
        ""
    ]
    
    # Geographic mapping
    if !isempty(source_context.geographic_patterns)
        geo_dims = filter(row -> occursin("GEO", uppercase(row.dimension_id)) || 
                               occursin("AREA", uppercase(row.dimension_id)), 
                         sdmx_context.dimensions)
        
        if nrow(geo_dims) > 0
            geo_dim = geo_dims[1, :dimension_id]
            codelist_id = geo_dims[1, :codelist_id]
            
            push!(sections, "### Geographic Code Mapping")
            geo_patterns_str = join(source_context.geographic_patterns, ", ")
            push!(sections, "**Source → Target**: " * geo_patterns_str * " → " * string(geo_dim))
            
            if haskey(sdmx_context.codelist_summary, codelist_id)
                cl_info = sdmx_context.codelist_summary[codelist_id]
                push!(sections, "- Target codelist: " * string(codelist_id))
                codes_str = join(cl_info.sample_codes, ", ")
                push!(sections, "- Available codes: " * codes_str * " (" * string(cl_info.total_codes) * " total)")
                
                # Generate mapping dictionary template
                push!(sections, "```julia")
                push!(sections, "# Geographic code mapping dictionary")
                push!(sections, "geo_mapping = Dict(")
                for code in cl_info.sample_codes[1:min(3, length(cl_info.sample_codes))]
                    # Create example mappings
                    example_name = if code == "US"
                        "\"United States\" => \"US\","
                    elseif code == "AU"  
                        "\"Australia\" => \"AU\","
                    elseif code == "CN"
                        "\"China\" => \"CN\","
                    else
                        "\"" * string(code) * "_full_name\" => \"" * string(code) * "\","
                    end
                    push!(sections, "    " * example_name)
                end
                push!(sections, "    # Add all required mappings")
                push!(sections, ")")
                push!(sections, "```")
            end
            push!(sections, "")
        end
    end
    
    # Value column mapping for pivoting
    if source_context.data_shape.needs_pivoting && !isempty(source_context.data_shape.pivot_candidates)
        push!(sections, "### Value Column Mapping (Pivoting Required)")
        push!(sections, "The following columns need to be pivoted into INDICATOR/OBS_VALUE structure:")
        
        for candidate in source_context.data_shape.pivot_candidates
            # Try to suggest indicator codes based on column names
            suggested_code = generate_indicator_code_suggestion(candidate)
            push!(sections, "- **" * string(candidate) * "** → INDICATOR=\"" * string(suggested_code) * "\", OBS_VALUE=value")
        end
        push!(sections, "")
    end
    
    # Time format mapping
    if !isempty(source_context.time_patterns)
        push!(sections, "### Time Format Mapping")
        for time_pattern in source_context.time_patterns
            push!(sections, "**Source**: " * string(time_pattern.column) * " (format: " * string(time_pattern.format) * ")")
            push!(sections, "**Target**: TIME_PERIOD (SDMX time format)")
            push!(sections, "- Convert to appropriate SDMX time period format")
            push!(sections, "- Validate time periods are within data coverage range")
        end
        push!(sections, "")
    end
    
    return join(sections, "\n")
end

"""
    generate_indicator_code_suggestion(column_name::String) -> String

Generate a suggested indicator code based on column name patterns.
"""
function generate_indicator_code_suggestion(column_name::String)
    name_lower = lowercase(column_name)
    
    if occursin("gdp", name_lower)
        return "GDP_PC"
    elseif occursin("population", name_lower) || occursin("pop", name_lower)
        return "POP_TOT"
    elseif occursin("inflation", name_lower) || occursin("infl", name_lower)
        return "INF_RATE"
    elseif occursin("unemployment", name_lower) || occursin("unemp", name_lower)
        return "UNEMP_RATE"
    elseif occursin("export", name_lower)
        return "EXP_VAL"
    elseif occursin("import", name_lower)
        return "IMP_VAL"
    else
        # Generate generic code from column name
        clean_name = replace(uppercase(column_name), r"[^A-Z0-9]" => "_")
        return clean_name[1:min(10, length(clean_name))]
    end
end

"""
    build_comprehensive_prompt(sdmx_context::SDMXStructuralContext, 
                              source_context::DataSourceContext,
                              template::PromptTemplate,
                              scenarios::Vector{TransformationScenario}=TransformationScenario[]) -> ComprehensivePrompt

Build a complete comprehensive prompt combining all context and templates.
"""
function build_comprehensive_prompt(sdmx_context::SDMXStructuralContext, 
                                   source_context::DataSourceContext,
                                   template::PromptTemplate,
                                   scenarios::Vector{TransformationScenario}=TransformationScenario[])
    
    # Build individual sections
    sdmx_section = build_sdmx_context_section(sdmx_context)
    source_section = build_source_analysis_section(source_context)
    mapping_section = build_code_mapping_section(sdmx_context, source_context)
    
    # Add scenario-specific instructions
    scenario_section = ""
    if !isempty(scenarios)
        scenario_parts = ["## SCENARIO-SPECIFIC REQUIREMENTS", ""]
        for scenario in scenarios
            append!(scenario_parts, [
                "### " * string(scenario.name),
                scenario.description,
                "",
                scenario.specific_instructions,
                "",
                "**Additional Requirements:**"
            ])
            for req in scenario.additional_requirements
                push!(scenario_parts, "- " * string(req))
            end
            push!(scenario_parts, "")
            if !isempty(scenario.example_code_snippet)
                push!(scenario_parts, "**Example Code Pattern:**")
                push!(scenario_parts, "```julia")
                push!(scenario_parts, scenario.example_code_snippet)
                push!(scenario_parts, "```")
                push!(scenario_parts, "")
            end
        end
        scenario_section = join(scenario_parts, "\n")
    end
    
    # Combine all sections into final prompt
    full_prompt = join([
        template.header_template,
        "",
        sdmx_section,
        "",
        source_section,
        "",
        mapping_section,
        "",
        scenario_section,
        "",
        template.requirements_template,
        "",
        template.validation_template,
        "",
        template.output_template,
        "",
        "Generate complete, executable Julia code with detailed comments explaining each transformation step."
    ], "\n")
    
    return ComprehensivePrompt(
        sdmx_section,
        source_section,
        mapping_section,
        scenario_section,
        template.validation_template,
        full_prompt
    )
end

"""
    generate_transformation_prompt(dataflow_url::String, 
                                  source_file::String;
                                  style::Symbol=:tidier,
                                  scenarios::Vector{TransformationScenario}=TransformationScenario[],
                                  provider::Symbol=:ollama,
                                  model::String="") -> String

Complete end-to-end function to generate transformation script from SDMX schema and source data.
"""
function generate_transformation_prompt(dataflow_url::String, 
                                       source_file::String;
                                       style::Symbol=:tidier,
                                       scenarios::Vector{TransformationScenario}=TransformationScenario[],
                                       provider::Symbol=:ollama,
                                       model::String="")
    
    @assert !isempty(dataflow_url) "Dataflow URL cannot be empty"
    @assert !isempty(source_file) "Source file path cannot be empty"
    @assert isfile(source_file) "Source file does not exist: $source_file"
    @assert style ∈ [:tidier, :dataframes, :mixed] "Style must be :tidier, :dataframes, or :mixed"
    
    # Extract comprehensive contexts
    println("Extracting SDMX structural context...")
    sdmx_context = extract_structural_context(dataflow_url)
    
    println("Analyzing source data structure...")
    source_context = extract_data_source_context(source_file)
    
    # Create template and build prompt
    template = create_prompt_template(style)
    
    # Auto-detect scenarios based on source analysis
    auto_scenarios = detect_scenarios(source_context)
    all_scenarios = vcat(scenarios, auto_scenarios)
    
    println("Building comprehensive prompt...")
    comprehensive_prompt = build_comprehensive_prompt(sdmx_context, source_context, template, all_scenarios)
    
    # Generate script using LLM
    println("Generating transformation script with " * string(provider) * "...")
    script_result = sdmx_aigenerate(comprehensive_prompt.full_prompt; provider=provider, model=model)
    
    return script_result.content
end

"""
    detect_scenarios(source_context::DataSourceContext) -> Vector{TransformationScenario}

Automatically detect relevant transformation scenarios based on source data analysis.
"""
function detect_scenarios(source_context::DataSourceContext)
    scenarios = TransformationScenario[]
    
    # Detect Excel scenario
    if source_context.excel_structure !== nothing
        push!(scenarios, EXCEL_SCENARIO)
    else
        push!(scenarios, CSV_SCENARIO)
    end
    
    # Detect pivoting scenario
    if source_context.data_shape.needs_pivoting
        push!(scenarios, PIVOTING_SCENARIO)
    end
    
    # Detect code mapping scenario
    if !isempty(source_context.geographic_patterns) || 
       length(source_context.value_patterns) > 1
        push!(scenarios, CODE_MAPPING_SCENARIO)
    end
    
    return scenarios
end