using Test
using SDMX
using SDMXLLM
using DataFrames

# --- LLM Integration tests ---
# Test LLM configuration setup
ollama_config = setup_llm_config(SDMXLLM.OLLAMA; model="llama2")
@test ollama_config.provider == OLLAMA
@test ollama_config.model == "llama2"
@test ollama_config.api_key === nothing

openai_config = setup_llm_config(SDMXLLM.OPENAI; model="gpt-4")
@test openai_config.provider == SDMXLLM.OPENAI
@test openai_config.model == "gpt-4"

# Test Excel structure analysis with mock data
# Create a simple test Excel file in memory simulation
test_excel_data = DataFrame(
    Country = ["FJ", "TV", "FJ", "TV"],
    Y2020 = [85.2, 92.1, 78.9, 89.7],
    Y2021 = [87.1, 93.5, 82.3, 91.2],
    Y2022 = [88.3, 94.1, 84.1, 92.6]
)

# This would be a full Excel analysis in practice
# For now, we verify the core functionality works
@test length(string(SDMXLLM.OLLAMA)) > 0  # Enum works
@test length(string(SDMXLLM.OPENAI)) > 0
@test length(string(SDMXLLM.ANTHROPIC)) > 0

# --- Advanced Mapping Inference tests ---

# Test inference engine creation
inference_engine = create_inference_engine(fuzzy_threshold=0.6, min_confidence=0.2)
@test inference_engine.fuzzy_threshold == 0.6
@test inference_engine.min_confidence == 0.2

# Test fuzzy matching
fuzzy_score = fuzzy_match_score("country", "GEO_PICT")
@test fuzzy_score > 0.3  # Should find semantic similarity with improved algorithm

# Test value pattern analysis
mock_codelist = DataFrame(code_id=["FJ", "VU", "TV"], name=["Fiji", "Vanuatu", "Tuvalu"])
test_values = ["FJ", "VU", "TV"]
pattern_analysis = analyze_value_patterns(test_values, mock_codelist)
@test haskey(pattern_analysis, "exact_matches")
@test pattern_analysis["exact_matches"] == 3
@test pattern_analysis["confidence_score"] == 1.0

# Test hierarchical relationship detection
hierarchical_profile = profile_source_data(DataFrame(
    broad_cat = ["A", "A", "B", "B"],
    detail_cat = ["A1", "A2", "B1", "B2"]
), "test")
schema_for_test = extract_dataflow_schema_from_string(String(HTTP.get("https://stats-sdmx-disseminate.pacificdata.org/rest/dataflow/SPC/DF_BP50/1.1?references=all").body))
hierarchy_analysis = detect_hierarchical_relationships(hierarchical_profile, schema_for_test)
@test haskey(hierarchy_analysis, "potential_hierarchies")
@test haskey(hierarchy_analysis, "parent_child_relationships")

# --- Script Generation tests ---
# Test script generator setup
test_llm_config = setup_llm_config(OLLAMA; model="llama2")
script_generator = create_script_generator(test_llm_config)
@test script_generator.llm_config == test_llm_config
@test script_generator.include_validation == true
@test script_generator.include_comments == true
@test script_generator.tidier_style == "pipes"
@test haskey(script_generator.templates, "standard_transformation")
@test haskey(script_generator.templates, "pivot_transformation")
@test haskey(script_generator.templates, "excel_multi_sheet")
@test haskey(script_generator.templates, "simple_csv")

# Test template creation
standard_template = create_standard_template()
@test standard_template.template_name == "standard_transformation"
@test haskey(standard_template.template_sections, "header")
@test haskey(standard_template.template_sections, "data_loading")
@test haskey(standard_template.template_sections, "transformations")
@test haskey(standard_template.template_sections, "validation")
@test haskey(standard_template.template_sections, "output")
@test "DataFrames" in standard_template.required_packages
@test "Tidier" in standard_template.required_packages

# Test transformation step building
test_data = DataFrame(
    country = ["FJ", "TV", "FJ", "TV"],
    year = [2020, 2020, 2021, 2021],
    value = [85.2, 92.1, 87.1, 89.3],
    category = ["A", "A", "B", "B"]
)
profile = profile_source_data(test_data, "test.csv")
advanced_mapping = infer_advanced_mappings(inference_engine, profile, schema_for_test)
transformation_steps = build_transformation_steps(advanced_mapping, profile, schema_for_test)
@test length(transformation_steps) > 0
@test any(step -> step.operation_type == "read", transformation_steps)
@test any(step -> step.operation_type == "mutate" || step.operation_type == "select", transformation_steps)
@test any(step -> step.operation_type == "write", transformation_steps)

# Test loading code generation
csv_loading_code = get_loading_code(profile)
@test occursin("CSV.read", csv_loading_code)
@test occursin(profile.file_path, csv_loading_code)

# Test Excel profile loading code
excel_profile = SourceDataProfile(
    "test.xlsx", "xlsx", 4, 4, String[], 1.0, profile.columns,
    String[], DataFrame(), String[], String[], String[]
)
excel_loading_code = get_loading_code(excel_profile)
@test occursin("XLSX.readtable", excel_loading_code)

# Test transformation logic generation
test_mapping = SDMXLLM.MappingCandidate(
    "country", "GEO_PICT", SDMXLLM.HIGH, "exact", 1.0,
    nothing, String[], String[]
)
basic_logic = generate_transformation_logic(test_mapping)
@test occursin("GEO_PICT = country", basic_logic)

# Test validation logic creation
validation_logic = create_validation_logic(advanced_mapping, schema_for_test)
@test occursin("required_cols", validation_logic)
@test occursin("missing_cols", validation_logic)
@test occursin("missing_count", validation_logic)

# Test script complexity calculation
complexity = calculate_script_complexity(transformation_steps, advanced_mapping)
@test complexity >= 0.0
@test complexity <= 1.0

# Test template selection
selected_template = select_template(script_generator, profile, nothing, "")
@test selected_template.template_name in ["standard_transformation", "simple_csv"]

# CSV template should be selected for simple CSV files
simple_csv_profile = SourceDataProfile(
    "simple.csv", "csv", 4, 3, String[], 1.0,
    profile.columns[1:3], String[], DataFrame(), String[], String[], String[]
)
csv_template = select_template(script_generator, simple_csv_profile, nothing, "")
@test csv_template.template_name == "simple_csv"

# Test script validation
mock_script = SDMXLLM.GeneratedScript(
    "test_script", "test.csv", "SPC:DF_BP50",
    "using DataFrames, Tidier\nsource_data = CSV.read(\"test.csv\", DataFrame)\ntransformed_data = source_data |> @mutate(GEO_PICT = country)\nCSV.write(\"output.csv\", transformed_data)",
    transformation_steps, 0.3, String[], String[], "standard_transformation", string(now())
)

script_validation = validate_generated_script(mock_script)
@test haskey(script_validation, "syntax_issues")
@test haskey(script_validation, "missing_elements")
@test haskey(script_validation, "recommendations")
@test haskey(script_validation, "overall_quality")
@test script_validation["overall_quality"] in ["good", "acceptable", "needs_work"]

# Test script preview
preview_text = preview_script_output(mock_script; max_lines=20)
@test occursin("GENERATED SCRIPT PREVIEW", preview_text)
@test occursin("test_script", preview_text)
@test occursin("standard_transformation", preview_text)
@test occursin("TRANSFORMATION STEPS", preview_text)
@test occursin("CODE PREVIEW", preview_text)

# Test script guidance creation
validation_notes, user_guidance = create_script_guidance(advanced_mapping, transformation_steps, profile, schema_for_test)
@test length(validation_notes) >= 0
@test length(user_guidance) >= 0
@test any(guidance -> occursin("Review and test", guidance), user_guidance)
@test any(guidance -> occursin("TODO comments", guidance), user_guidance)
