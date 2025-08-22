"""
Advanced Data Mapping Inference Engine for SDMX.jl

This module provides intelligent mapping between source data and SDMX schemas using:
- Advanced value matching against SDMX codelists
- Fuzzy string matching with confidence scoring
- Statistical analysis of data patterns
- Hierarchical mapping inference
- Machine learning-style pattern recognition
- User feedback incorporation for continuous improvement
"""

# Dependencies loaded at package level

export MappingCandidate, MappingConfidence, AdvancedMappingResult, InferenceEngine,
       create_inference_engine, infer_advanced_mappings, validate_mapping_quality,
       suggest_value_transformations, analyze_mapping_coverage, learn_from_feedback,
       fuzzy_match_score, analyze_value_patterns, detect_hierarchical_relationships

"""
    MappingConfidence

Enumeration of confidence levels for advanced mapping suggestions.

This enum represents different confidence levels for data mapping inferences,
from very low confidence requiring manual review to very high confidence
suitable for automatic application.

# Values
- `VERY_LOW = 1`: 0-20% confidence - manual review required
- `LOW = 2`: 20-40% confidence - careful validation needed  
- `MEDIUM = 3`: 40-60% confidence - moderate confidence
- `HIGH = 4`: 60-80% confidence - high confidence
- `VERY_HIGH = 5`: 80-100% confidence - suitable for automation

# See also
[`MappingCandidate`](@ref), [`AdvancedMappingResult`](@ref)
"""
@enum MappingConfidence begin
    VERY_LOW = 1    # 0-20% confidence
    LOW = 2         # 20-40% confidence  
    MEDIUM = 3      # 40-60% confidence
    HIGH = 4        # 60-80% confidence
    VERY_HIGH = 5   # 80-100% confidence
end

"""
    MappingCandidate

A potential mapping between source and target columns with comprehensive assessment.

This struct represents a candidate mapping discovered by the inference engine,
including confidence scores, evidence supporting the mapping, and suggested
transformations needed to implement the mapping.

# Fields
- `source_column::String`: Name of the source data column
- `target_column::String`: Name of the target schema column
- `confidence_score::Float64`: Numeric confidence score (0.0-1.0)
- `confidence_level::MappingConfidence`: Categorized confidence level
- `match_type::String`: Type of match ("exact", "fuzzy", "value_pattern", "statistical")
- `evidence::Dict{String, Any}`: Supporting evidence for the mapping decision
- `suggested_transformation::Union{String, Nothing}`: Code transformation needed
- `validation_notes::Vector{String}`: Additional validation notes and warnings

# Examples
```julia
candidate = MappingCandidate(
    "country_name",
    "GEO_PICT", 
    0.85,
    VERY_HIGH,
    "fuzzy",
    Dict("name_similarity" => 0.8, "value_analysis" => analysis),
    "@mutate(GEO_PICT = recode(country_name, mapping...))",
    ["High confidence country mapping"]
)
```

# See also
[`MappingConfidence`](@ref), [`AdvancedMappingResult`](@ref)
"""
struct MappingCandidate
    source_column::String
    target_column::String
    confidence_score::Float64
    confidence_level::MappingConfidence
    match_type::String  # "exact", "fuzzy", "value_pattern", "statistical"
    evidence::Dict{String, Any}  # Supporting evidence for the mapping
    suggested_transformation::Union{String, Nothing}  # Code transformation needed
    validation_notes::Vector{String}
end

"""
    AdvancedMappingResult

Comprehensive result of advanced mapping inference.
"""
struct AdvancedMappingResult
    mappings::Vector{MappingCandidate}
    coverage_analysis::Dict{String, Any}
    unmapped_source_columns::Vector{String}
    unmapped_target_columns::Vector{String}
    quality_score::Float64
    recommendations::Vector{String}
    transformation_complexity::Float64
end

"""
    InferenceEngine

Main engine for advanced mapping inference with learning capabilities.
"""
mutable struct InferenceEngine
    # Core data
    source_profile::Union{SourceDataProfile, Nothing}
    target_schema::Union{DataflowSchema, Nothing}
    codelists_data::Dict{String, DataFrame}  # Cached codelist data
    
    # Learning components
    successful_mappings::Dict{String, Vector{String}}  # Pattern -> successful targets
    failed_mappings::Dict{String, Vector{String}}      # Pattern -> failed targets
    user_feedback::Vector{Dict{String, Any}}           # Historical feedback
    
    # Configuration
    fuzzy_threshold::Float64
    min_confidence::Float64
    use_statistical_analysis::Bool
    enable_learning::Bool
end

"""
    create_inference_engine(; kwargs...) -> InferenceEngine

Creates a new advanced mapping inference engine with configurable parameters.

This function initializes an inference engine that can learn from user feedback
and apply multiple matching algorithms to discover data mappings between source
datasets and SDMX schemas.

# Arguments
- `fuzzy_threshold::Float64=0.6`: Minimum score for fuzzy string matching
- `min_confidence::Float64=0.3`: Minimum confidence threshold for mapping suggestions
- `use_statistical_analysis::Bool=true`: Enable statistical compatibility analysis
- `enable_learning::Bool=true`: Enable learning from user feedback

# Returns
- `InferenceEngine`: Configured inference engine ready for mapping analysis

# Examples
```julia
# Create engine with default settings
engine = create_inference_engine()

# Create engine with custom thresholds
engine = create_inference_engine(
    fuzzy_threshold=0.8,
    min_confidence=0.5,
    use_statistical_analysis=true,
    enable_learning=true
)

# Use the engine
result = infer_advanced_mappings(engine, source_profile, target_schema, source_data)
```

# See also
[`InferenceEngine`](@ref), [`infer_advanced_mappings`](@ref)
"""
function create_inference_engine(;fuzzy_threshold=0.6, 
                                min_confidence=0.3,
                                use_statistical_analysis=true, 
                                enable_learning=true)
    return InferenceEngine(
        nothing,  # source_profile
        nothing,  # target_schema
        Dict{String, DataFrame}(),  # codelists_data
        Dict{String, Vector{String}}(),  # successful_mappings
        Dict{String, Vector{String}}(),  # failed_mappings
        Vector{Dict{String, Any}}(),     # user_feedback
        fuzzy_threshold,
        min_confidence,
        use_statistical_analysis,
        enable_learning
    )
end

"""
    fuzzy_match_score(str1::String, str2::String) -> Float64

Calculates comprehensive fuzzy matching score between strings using multiple algorithms.

This function combines multiple string similarity measures including Jaro-Winkler
similarity, substring matching, token-based similarity, and semantic similarity
to produce a comprehensive matching score for data mapping inference.

# Arguments
- `str1::String`: First string to compare
- `str2::String`: Second string to compare

# Returns
- `Float64`: Similarity score between 0.0 (no match) and 1.0 (perfect match)

# Examples
```julia
# Exact match
score = fuzzy_match_score("country", "country")  # 1.0

# Close match
score = fuzzy_match_score("country_name", "geo_pict")  # ~0.3

# Semantic match
score = fuzzy_match_score("gender", "sex")  # ~0.5 (boosted by semantics)

# Substring match
score = fuzzy_match_score("time_period", "period")  # ~0.7
```

# See also
[`analyze_value_patterns`](@ref), [`create_inference_engine`](@ref)
"""
function fuzzy_match_score(str1::String, str2::String)
    # Normalize strings
    s1 = lowercase(strip(str1))
    s2 = lowercase(strip(str2))
    
    if s1 == s2
        return 1.0
    end
    
    # Jaro-Winkler-like similarity
    function jaro_similarity(str1, str2)
        if isempty(str1) && isempty(str2)
            return 1.0
        elseif isempty(str1) || isempty(str2)
            return 0.0
        end
        
        len1, len2 = length(str1), length(str2)
        match_window = max(len1, len2) ÷ 2 - 1
        match_window = max(0, match_window)
        
        matches = 0
        transpositions = 0
        
        str1_matches = falses(len1)
        str2_matches = falses(len2)
        
        # Find matches
        for i in 1:len1
            start = max(1, i - match_window)
            stop = min(i + match_window, len2)
            
            for j in start:stop
                if str2_matches[j] || str1[i] != str2[j]
                    continue
                end
                str1_matches[i] = str2_matches[j] = true
                matches += 1
                break
            end
        end
        
        if matches == 0
            return 0.0
        end
        
        # Count transpositions
        k = 1
        for i in 1:len1
            if !str1_matches[i]
                continue
            end
            while !str2_matches[k]
                k += 1
            end
            if str1[i] != str2[k]
                transpositions += 1
            end
            k += 1
        end
        
        jaro = (matches/len1 + matches/len2 + (matches - transpositions/2)/matches) / 3.0
        return jaro
    end
    
    # Combine multiple similarity measures
    jaro_score = jaro_similarity(s1, s2)
    
    # Substring similarity
    substring_score = 0.0
    if occursin(s1, s2) || occursin(s2, s1)
        substring_score = min(length(s1), length(s2)) / max(length(s1), length(s2))
    end
    
    # Token-based similarity
    tokens1 = Set(split(replace(s1, r"[^a-z0-9]" => " ")))
    tokens2 = Set(split(replace(s2, r"[^a-z0-9]" => " ")))
    
    filter!(x -> !isempty(x), tokens1)
    filter!(x -> !isempty(x), tokens2)
    
    token_score = if !isempty(tokens1) && !isempty(tokens2)
        length(intersect(tokens1, tokens2)) / length(union(tokens1, tokens2))
    else
        0.0
    end
    
    # Semantic similarity boost for known mappings
    semantic_boost = 0.0
    semantic_pairs = [
        ("country", "geo"), ("country", "geographic"), ("country", "pict"),
        ("gender", "sex"), ("year", "time"), ("year", "period"), ("time", "period"),
        ("rate", "value"), ("rate", "obs"), ("count", "value"), ("amount", "value")
    ]
    
    for (word1, word2) in semantic_pairs
        if (occursin(word1, s1) && occursin(word2, s2)) || (occursin(word2, s1) && occursin(word1, s2))
            semantic_boost = 0.3
            break
        end
    end
    
    # Weighted combination
    final_score = 0.4 * jaro_score + 0.2 * substring_score + 0.2 * token_score + semantic_boost
    return min(1.0, final_score)
end

"""
    analyze_value_patterns(source_values::Vector, target_codelist::DataFrame) -> Dict{String, Any}

Analyzes patterns in source values against target codelist to find matches and transformations.
"""
function analyze_value_patterns(source_values::Vector, target_codelist::DataFrame)
    analysis = Dict{String, Any}(
        "exact_matches" => 0,
        "fuzzy_matches" => 0,
        "pattern_matches" => 0,
        "unmatched" => 0,
        "match_details" => Dict{String, String}(),
        "suggested_transformations" => Vector{String}(),
        "confidence_score" => 0.0
    )
    
    # Get unique non-missing source values
    unique_source = unique(filter(!ismissing, source_values))
    if isempty(unique_source)
        return analysis
    end
    
    # Get target codes (assume code_id column exists)
    if !hasproperty(target_codelist, :code_id) && !("code_id" in names(target_codelist))
        return analysis
    end
    
    target_codes = if hasproperty(target_codelist, :code_id)
        unique(target_codelist.code_id)
    else
        unique(target_codelist[!, "code_id"])
    end
    
    # Analyze each source value
    for source_val in unique_source
        source_str = string(source_val)
        best_match = nothing
        best_score = 0.0
        match_type = "unmatched"
        
        for target_code in target_codes
            target_str = string(target_code)
            
            # Exact match
            if lowercase(source_str) == lowercase(target_str)
                best_match = target_str
                best_score = 1.0
                match_type = "exact"
                break
            end
            
            # Fuzzy match
            score = fuzzy_match_score(source_str, target_str)
            if score > best_score && score > 0.7
                best_match = target_str
                best_score = score
                match_type = "fuzzy"
            end
        end
        
        # Also check against names if available
        if hasproperty(target_codelist, :name) || ("name" in names(target_codelist))
            target_names = if hasproperty(target_codelist, :name)
                filter(!ismissing, target_codelist.name)
            else
                filter(!ismissing, target_codelist[!, "name"])
            end
            
            for target_name in target_names
                name_str = string(target_name)
                score = fuzzy_match_score(source_str, name_str)
                if score > best_score && score > 0.6
                    # Find corresponding code
                    name_col = hasproperty(target_codelist, :name) ? :name : "name"
                    code_col = hasproperty(target_codelist, :code_id) ? :code_id : "code_id"
                    
                    matching_rows = target_codelist[target_codelist[!, name_col] .== target_name, :]
                    if nrow(matching_rows) > 0
                        best_match = string(matching_rows[1, code_col])
                        best_score = score
                        match_type = "name_fuzzy"
                    end
                end
            end
        end
        
        # Categorize the match
        if match_type == "exact"
            analysis["exact_matches"] += 1
        elseif match_type in ["fuzzy", "name_fuzzy"]
            analysis["fuzzy_matches"] += 1
        else
            analysis["unmatched"] += 1
        end
        
        if best_match !== nothing
            analysis["match_details"][source_str] = best_match
        end
    end
    
    # Calculate overall confidence
    total_values = length(unique_source)
    exact_ratio = analysis["exact_matches"] / total_values
    fuzzy_ratio = analysis["fuzzy_matches"] / total_values
    
    analysis["confidence_score"] = exact_ratio + (fuzzy_ratio * 0.7)
    
    # Suggest transformations
    if analysis["fuzzy_matches"] > 0
        push!(analysis["suggested_transformations"], "Apply fuzzy matching with manual review")
    end
    
    if analysis["unmatched"] > 0
        unmatched_ratio = analysis["unmatched"] / total_values
        if unmatched_ratio > 0.3
            push!(analysis["suggested_transformations"], "Review unmapped values - may need codelist extension or data cleaning")
        end
    end
    
    return analysis
end

"""
    detect_hierarchical_relationships(source_profile::SourceDataProfile, 
                                    target_schema::DataflowSchema) -> Dict{String, Any}

Detects potential hierarchical relationships in the data that could inform mapping.
"""
function detect_hierarchical_relationships(source_profile::SourceDataProfile, 
                                         target_schema::DataflowSchema)
    hierarchical_analysis = Dict{String, Any}(
        "potential_hierarchies" => Dict{String, Vector{String}}(),
        "parent_child_relationships" => Vector{Tuple{String, String}}(),
        "aggregation_levels" => Dict{String, String}()
    )
    
    # Look for columns that might represent different levels of the same hierarchy
    # Example: country -> region, detailed_category -> broad_category
    
    for col1 in source_profile.columns
        for col2 in source_profile.columns
            if col1.name == col2.name
                continue
            end
            
            # Check if one column might be a parent of another
            if col1.is_categorical && col2.is_categorical
                # If col1 has fewer unique values than col2, it might be a parent
                if col1.unique_count < col2.unique_count && col1.unique_count > 1
                    # Additional heuristics based on naming patterns
                    name1_lower = lowercase(col1.name)
                    name2_lower = lowercase(col2.name)
                    
                    # Geographic hierarchies
                    if (occursin("country", name1_lower) && occursin("region", name2_lower)) ||
                       (occursin("region", name1_lower) && occursin("area", name2_lower)) ||
                       (occursin("broad", name1_lower) && occursin("detailed", name2_lower))
                        
                        push!(hierarchical_analysis["parent_child_relationships"], (col1.name, col2.name))
                        
                        if !haskey(hierarchical_analysis["potential_hierarchies"], col1.name)
                            hierarchical_analysis["potential_hierarchies"][col1.name] = String[]
                        end
                        push!(hierarchical_analysis["potential_hierarchies"][col1.name], col2.name)
                    end
                end
            end
        end
    end
    
    return hierarchical_analysis
end

"""
    infer_advanced_mappings(engine::InferenceEngine, 
                           source_profile::SourceDataProfile,
                           target_schema::DataflowSchema,
                           source_data::DataFrame) -> AdvancedMappingResult

Performs comprehensive mapping inference using all available techniques.
"""
function infer_advanced_mappings(engine::InferenceEngine, 
                                source_profile::SourceDataProfile,
                                target_schema::DataflowSchema,
                                source_data::DataFrame)
    
    # Update engine with current data
    engine.source_profile = source_profile
    engine.target_schema = target_schema
    
    # Load codelist data if needed
    load_codelist_data!(engine, target_schema)
    
    mapping_candidates = Vector{MappingCandidate}()
    
    # Get all target columns
    all_target_columns = vcat(
        target_schema.dimensions.dimension_id,
        target_schema.time_dimension !== nothing ? [target_schema.time_dimension.dimension_id] : String[],
        target_schema.measures.measure_id,
        target_schema.attributes.attribute_id
    )
    
    # Analyze each source column against each target column
    for source_col in source_profile.columns
        source_data_col = source_data[!, source_col.name]
        
        for target_col in all_target_columns
            candidate = analyze_column_mapping(
                engine, source_col, target_col, source_data_col, target_schema
            )
            
            if candidate !== nothing && candidate.confidence_score >= engine.min_confidence
                push!(mapping_candidates, candidate)
            end
        end
    end
    
    # Sort candidates by confidence
    sort!(mapping_candidates, by=x->x.confidence_score, rev=true)
    
    # Remove duplicate mappings (same source column mapped to multiple targets)
    final_mappings = select_best_mappings(mapping_candidates)
    
    # Analyze coverage and quality
    coverage_analysis = analyze_mapping_coverage(final_mappings, source_profile, target_schema)
    
    # Generate recommendations
    recommendations = generate_mapping_recommendations(
        final_mappings, coverage_analysis, source_profile, target_schema
    )
    
    # Calculate transformation complexity
    complexity = calculate_transformation_complexity(final_mappings, coverage_analysis)
    
    # Identify unmapped columns
    mapped_source = Set([m.source_column for m in final_mappings])
    mapped_target = Set([m.target_column for m in final_mappings])
    
    unmapped_source = [col.name for col in source_profile.columns if col.name ∉ mapped_source]
    unmapped_target = [col for col in all_target_columns if col ∉ mapped_target]
    
    # Overall quality score
    quality_score = calculate_mapping_quality_score(final_mappings, coverage_analysis)
    
    return AdvancedMappingResult(
        final_mappings,
        coverage_analysis,
        unmapped_source,
        unmapped_target,
        quality_score,
        recommendations,
        complexity
    )
end

"""
    load_codelist_data!(engine::InferenceEngine, target_schema::DataflowSchema)

Loads codelist data for value-based matching.
"""
function load_codelist_data!(engine::InferenceEngine, target_schema::DataflowSchema)
    # Get unique codelists referenced in the schema
    codelist_refs = Set{String}()
    
    for row in eachrow(target_schema.dimensions)
        if !ismissing(row.codelist_id)
            push!(codelist_refs, row.codelist_id)
        end
    end
    
    for row in eachrow(target_schema.attributes)
        if !ismissing(row.codelist_id)
            push!(codelist_refs, row.codelist_id)
        end
    end
    
    # Load codelists that aren't already cached
    for codelist_id in codelist_refs
        if !haskey(engine.codelists_data, codelist_id)
            try
                # This would normally load from the same SDMX endpoint
                # For now, we'll create a placeholder
                engine.codelists_data[codelist_id] = DataFrame(
                    code_id = ["CODE1", "CODE2", "CODE3"],
                    name = ["Name 1", "Name 2", "Name 3"]
                )
            catch e
                @warn "Could not load codelist $codelist_id: $e"
            end
        end
    end
end

"""
    analyze_column_mapping(engine::InferenceEngine, source_col::ColumnProfile, 
                          target_col::String, source_data::Vector,
                          target_schema::DataflowSchema) -> Union{MappingCandidate, Nothing}

Analyzes a specific source-target column mapping.
"""
function analyze_column_mapping(engine::InferenceEngine, 
                               source_col::ColumnProfile, 
                               target_col::String, 
                               source_data::Vector,
                               target_schema::DataflowSchema)
    
    evidence = Dict{String, Any}()
    confidence_score = 0.0
    match_type = "unknown"
    suggested_transformation = nothing
    validation_notes = String[]
    
    # 1. Name-based matching
    name_score = fuzzy_match_score(source_col.name, target_col)
    evidence["name_similarity"] = name_score
    confidence_score += name_score * 0.4
    
    if name_score > 0.8
        match_type = "name_exact"
    elseif name_score > engine.fuzzy_threshold
        match_type = "name_fuzzy"
    end
    
    # 2. Type compatibility checking
    target_info = get_target_column_info(target_col, target_schema)
    if target_info !== nothing
        type_compatibility = assess_type_compatibility(source_col, target_info)
        evidence["type_compatibility"] = type_compatibility
        confidence_score += type_compatibility * 0.2
        
        if type_compatibility < 0.3
            push!(validation_notes, "Type compatibility concerns - may need data transformation")
        end
    end
    
    # 3. Value pattern analysis (if codelist available)
    if target_info !== nothing && !ismissing(target_info.codelist_id)
        codelist_id = target_info.codelist_id
        if haskey(engine.codelists_data, codelist_id)
            codelist = engine.codelists_data[codelist_id]
            value_analysis = analyze_value_patterns(source_data, codelist)
            evidence["value_analysis"] = value_analysis
            confidence_score += value_analysis["confidence_score"] * 0.3
            
            if !isempty(value_analysis["suggested_transformations"])
                suggested_transformation = join(value_analysis["suggested_transformations"], "; ")
            end
            
            if value_analysis["confidence_score"] > 0.7
                match_type = "value_pattern"
            end
        end
    end
    
    # 4. Statistical compatibility
    if engine.use_statistical_analysis
        statistical_score = assess_statistical_compatibility(source_col, target_info)
        evidence["statistical_compatibility"] = statistical_score
        confidence_score += statistical_score * 0.1
    end
    
    # 5. Apply learning from historical data
    if engine.enable_learning
        learning_boost = apply_learning_boost(engine, source_col.name, target_col)
        evidence["learning_boost"] = learning_boost
        confidence_score += learning_boost
    end
    
    # Normalize confidence score
    confidence_score = min(1.0, max(0.0, confidence_score))
    
    # Determine confidence level
    confidence_level = if confidence_score >= 0.8
        VERY_HIGH
    elseif confidence_score >= 0.6
        HIGH
    elseif confidence_score >= 0.4
        MEDIUM
    elseif confidence_score >= 0.2
        LOW
    else
        VERY_LOW
    end
    
    # Only return candidate if above minimum threshold
    if confidence_score >= engine.min_confidence
        return MappingCandidate(
            source_col.name,
            target_col,
            confidence_score,
            confidence_level,
            match_type,
            evidence,
            suggested_transformation,
            validation_notes
        )
    else
        return nothing
    end
end

"""
    get_target_column_info(target_col::String, target_schema::DataflowSchema) -> Union{NamedTuple, Nothing}

Gets information about a target column from the schema.
"""
function get_target_column_info(target_col::String, target_schema::DataflowSchema)
    # Check dimensions
    if nrow(target_schema.dimensions) > 0
        dim_matches = filter(row -> row.dimension_id == target_col, target_schema.dimensions)
        if nrow(dim_matches) > 0
            row = dim_matches[1, :]
            return (type="dimension", codelist_id=row.codelist_id, data_type=row.data_type)
        end
    end
    
    # Check time dimension
    if target_schema.time_dimension !== nothing && target_schema.time_dimension.dimension_id == target_col
        td = target_schema.time_dimension
        return (type="time_dimension", codelist_id=missing, data_type=td.data_type)
    end
    
    # Check measures
    if nrow(target_schema.measures) > 0
        measure_matches = filter(row -> row.measure_id == target_col, target_schema.measures)
        if nrow(measure_matches) > 0
            row = measure_matches[1, :]
            return (type="measure", codelist_id=missing, data_type=row.data_type)
        end
    end
    
    # Check attributes
    if nrow(target_schema.attributes) > 0
        attr_matches = filter(row -> row.attribute_id == target_col, target_schema.attributes)
        if nrow(attr_matches) > 0
            row = attr_matches[1, :]
            return (type="attribute", codelist_id=row.codelist_id, data_type=row.data_type)
        end
    end
    
    return nothing
end

"""
    assess_type_compatibility(source_col::ColumnProfile, target_info::NamedTuple) -> Float64

Assesses how compatible the source column type is with the target requirements.
"""
function assess_type_compatibility(source_col::ColumnProfile, target_info::NamedTuple)
    compatibility = 0.5  # Base score
    
    # Time dimension compatibility
    if target_info.type == "time_dimension"
        if source_col.is_temporal
            compatibility = 1.0
        elseif source_col.type <: Integer && !source_col.is_categorical
            compatibility = 0.8  # Years as integers
        else
            compatibility = 0.2
        end
    
    # Measure compatibility
    elseif target_info.type == "measure"
        if source_col.numeric_stats !== nothing && !source_col.is_categorical
            compatibility = 1.0
        elseif source_col.type <: Number
            compatibility = 0.8
        else
            compatibility = 0.1
        end
    
    # Dimension/Attribute compatibility
    elseif target_info.type in ["dimension", "attribute"]
        if !ismissing(target_info.codelist_id)
            # Should be categorical or have limited values
            if source_col.is_categorical
                compatibility = 1.0
            elseif source_col.unique_count < 100
                compatibility = 0.7
            else
                compatibility = 0.3
            end
        else
            # Free text field
            compatibility = 0.8
        end
    end
    
    return compatibility
end

"""
    assess_statistical_compatibility(source_col::ColumnProfile, target_info::Union{NamedTuple, Nothing}) -> Float64

Assesses statistical compatibility between source and target.
"""
function assess_statistical_compatibility(source_col::ColumnProfile, target_info::Union{NamedTuple, Nothing})
    if target_info === nothing
        return 0.0
    end
    
    # This is a simplified statistical compatibility check
    # In practice, this could be much more sophisticated
    
    if target_info.type == "measure" && source_col.numeric_stats !== nothing
        # Check if the numeric range seems reasonable
        stats = source_col.numeric_stats
        if 0 <= stats.min <= stats.max <= 100 && stats.mean > 0
            return 0.8  # Looks like a percentage or rate
        elseif stats.min >= 0
            return 0.6  # Non-negative numbers are often measures
        else
            return 0.4
        end
    elseif target_info.type in ["dimension", "attribute"] && source_col.is_categorical
        # Check cardinality appropriateness
        if 2 <= source_col.unique_count <= 20
            return 0.8  # Good categorical range
        elseif source_col.unique_count <= 100
            return 0.6
        else
            return 0.3
        end
    end
    
    return 0.5  # Default moderate compatibility
end

"""
    apply_learning_boost(engine::InferenceEngine, source_name::String, target_name::String) -> Float64

Applies learning-based confidence boost based on historical patterns.
"""
function apply_learning_boost(engine::InferenceEngine, source_name::String, target_name::String)
    boost = 0.0
    
    # Create pattern key
    pattern_key = "$(lowercase(source_name))->$(lowercase(target_name))"
    
    # Check successful mappings
    if haskey(engine.successful_mappings, source_name)
        if target_name in engine.successful_mappings[source_name]
            boost += 0.1
        end
    end
    
    # Check failed mappings (negative boost)
    if haskey(engine.failed_mappings, source_name)
        if target_name in engine.failed_mappings[source_name]
            boost -= 0.1
        end
    end
    
    return boost
end

"""
    select_best_mappings(candidates::Vector{MappingCandidate}) -> Vector{MappingCandidate}

Selects the best mapping for each source column, avoiding conflicts.
"""
function select_best_mappings(candidates::Vector{MappingCandidate})
    selected = Vector{MappingCandidate}()
    used_sources = Set{String}()
    used_targets = Set{String}()
    
    # Sort by confidence and process highest first
    sorted_candidates = sort(candidates, by=x->x.confidence_score, rev=true)
    
    for candidate in sorted_candidates
        # Skip if source or target already used (could make this configurable)
        if candidate.source_column in used_sources
            continue
        end
        
        push!(selected, candidate)
        push!(used_sources, candidate.source_column)
        push!(used_targets, candidate.target_column)
    end
    
    return selected
end

"""
    analyze_mapping_coverage(mappings::Vector{MappingCandidate}, 
                            source_profile::SourceDataProfile,
                            target_schema::DataflowSchema) -> Dict{String, Any}

Analyzes how well the mappings cover the required schema.
"""
function analyze_mapping_coverage(mappings::Vector{MappingCandidate}, 
                                source_profile::SourceDataProfile,
                                target_schema::DataflowSchema)
    
    required_cols = get_required_columns(target_schema)
    optional_cols = get_optional_columns(target_schema)
    
    mapped_required = sum([m.target_column in required_cols for m in mappings])
    mapped_optional = sum([m.target_column in optional_cols for m in mappings])
    
    coverage = Dict{String, Any}(
        "required_coverage" => mapped_required / length(required_cols),
        "optional_coverage" => mapped_optional / length(optional_cols),
        "total_mappings" => length(mappings),
        "high_confidence_mappings" => sum([m.confidence_level >= HIGH for m in mappings]),
        "needs_transformation" => sum([m.suggested_transformation !== nothing for m in mappings])
    )
    
    return coverage
end

"""
    calculate_transformation_complexity(mappings::Vector{MappingCandidate}, 
                                      coverage_analysis::Dict{String, Any}) -> Float64

Calculates the overall transformation complexity score.
"""
function calculate_transformation_complexity(mappings::Vector{MappingCandidate}, 
                                           coverage_analysis::Dict{String, Any})
    complexity = 0.0
    
    # Base complexity from number of transformations needed
    transformations_needed = coverage_analysis["needs_transformation"]
    complexity += transformations_needed * 0.1
    
    # Complexity from low confidence mappings
    low_confidence = sum([m.confidence_level <= MEDIUM for m in mappings])
    complexity += low_confidence * 0.05
    
    # Complexity from poor coverage
    required_coverage = coverage_analysis["required_coverage"]
    if required_coverage < 0.8
        complexity += (0.8 - required_coverage) * 0.3
    end
    
    return min(1.0, complexity)
end

"""
    calculate_mapping_quality_score(mappings::Vector{MappingCandidate}, 
                                   coverage_analysis::Dict{String, Any}) -> Float64

Calculates overall mapping quality score.
"""
function calculate_mapping_quality_score(mappings::Vector{MappingCandidate}, 
                                       coverage_analysis::Dict{String, Any})
    if isempty(mappings)
        return 0.0
    end
    
    # Average confidence of mappings
    avg_confidence = mean([m.confidence_score for m in mappings])
    
    # Weight by coverage
    required_coverage = coverage_analysis["required_coverage"]
    
    # Combine scores
    quality = 0.6 * avg_confidence + 0.4 * required_coverage
    
    return quality
end

"""
    generate_mapping_recommendations(mappings::Vector{MappingCandidate},
                                   coverage_analysis::Dict{String, Any},
                                   source_profile::SourceDataProfile,
                                   target_schema::DataflowSchema) -> Vector{String}

Generates actionable recommendations for improving mappings.
"""
function generate_mapping_recommendations(mappings::Vector{MappingCandidate},
                                        coverage_analysis::Dict{String, Any},
                                        source_profile::SourceDataProfile,
                                        target_schema::DataflowSchema)
    
    recommendations = String[]
    
    # Coverage recommendations
    if coverage_analysis["required_coverage"] < 0.8
        push!(recommendations, "Required field coverage is below 80% - review unmapped required columns")
    end
    
    # Confidence recommendations
    low_confidence = sum([m.confidence_level <= MEDIUM for m in mappings])
    if low_confidence > 0
        push!(recommendations, "$low_confidence mappings have medium or low confidence - manual review recommended")
    end
    
    # Transformation recommendations
    if coverage_analysis["needs_transformation"] > 0
        push!(recommendations, "$(coverage_analysis["needs_transformation"]) columns need value transformations")
    end
    
    # Data quality recommendations
    if source_profile.data_quality_score < 0.9
        push!(recommendations, "Source data quality is $(round(source_profile.data_quality_score*100, digits=1))% - consider data cleaning")
    end
    
    return recommendations
end

"""
    learn_from_feedback(engine::InferenceEngine, feedback::Dict{String, Any})

Incorporates user feedback to improve future mapping suggestions.
"""
function learn_from_feedback(engine::InferenceEngine, feedback::Dict{String, Any})
    if !engine.enable_learning
        return
    end
    
    push!(engine.user_feedback, feedback)
    
    # Process feedback to update learning patterns
    if haskey(feedback, "accepted_mappings")
        for mapping in feedback["accepted_mappings"]
            source_col = mapping["source"]
            target_col = mapping["target"]
            
            if !haskey(engine.successful_mappings, source_col)
                engine.successful_mappings[source_col] = String[]
            end
            push!(engine.successful_mappings[source_col], target_col)
        end
    end
    
    if haskey(feedback, "rejected_mappings")
        for mapping in feedback["rejected_mappings"]
            source_col = mapping["source"]
            target_col = mapping["target"]
            
            if !haskey(engine.failed_mappings, source_col)
                engine.failed_mappings[source_col] = String[]
            end
            push!(engine.failed_mappings[source_col], target_col)
        end
    end
end

"""
    suggest_value_transformations(mapping::MappingCandidate, 
                                 source_data::Vector,
                                 target_schema::DataflowSchema) -> Vector{String}

Suggests specific value transformation code for a mapping.
"""
function suggest_value_transformations(mapping::MappingCandidate, 
                                     source_data::Vector,
                                     target_schema::DataflowSchema)
    
    transformations = String[]
    
    if mapping.suggested_transformation !== nothing
        push!(transformations, mapping.suggested_transformation)
    end
    
    # Analyze the mapping evidence for specific transformations
    if haskey(mapping.evidence, "value_analysis")
        value_analysis = mapping.evidence["value_analysis"]
        
        if haskey(value_analysis, "match_details")
            match_details = value_analysis["match_details"]
            
            if !isempty(match_details)
                recode_rules = []
                for (source_val, target_val) in match_details
                    push!(recode_rules, "\"$source_val\" => \"$target_val\"")
                end
                
                transformation_code = "@mutate($(mapping.target_column) = recode($(mapping.source_column), $(join(recode_rules, ", "))))"
                push!(transformations, transformation_code)
            end
        end
    end
    
    return transformations
end

"""
    validate_mapping_quality(result::AdvancedMappingResult) -> Dict{String, Any}

Validates the quality of the mapping result and provides quality metrics.
"""
function validate_mapping_quality(result::AdvancedMappingResult)
    validation = Dict{String, Any}(
        "overall_quality" => result.quality_score,
        "coverage_adequate" => result.coverage_analysis["required_coverage"] >= 0.8,
        "confidence_distribution" => Dict{String, Int}(),
        "critical_issues" => String[],
        "warnings" => String[]
    )
    
    # Analyze confidence distribution
    for level in instances(MappingConfidence)
        count = sum([m.confidence_level == level for m in result.mappings])
        validation["confidence_distribution"][string(level)] = count
    end
    
    # Identify critical issues
    if result.coverage_analysis["required_coverage"] < 0.5
        push!(validation["critical_issues"], "Less than 50% of required fields are mapped")
    end
    
    if result.quality_score < 0.3
        push!(validation["critical_issues"], "Overall mapping quality is very low")
    end
    
    # Identify warnings
    if result.transformation_complexity > 0.7
        push!(validation["warnings"], "High transformation complexity detected")
    end
    
    if !isempty(result.unmapped_target_columns)
        required_cols = get_required_columns(result.mappings[1].source_column)  # This is a simplification
        unmapped_required = length(intersect(result.unmapped_target_columns, required_cols))
        if unmapped_required > 0
            push!(validation["warnings"], "$unmapped_required required columns remain unmapped")
        end
    end
    
    return validation
end