module SDMXLLM

# Load all dependencies at package level
using SDMX
using PromptingTools
using HTTP, JSON3, DataFrames, XLSX, CSV, Statistics, StatsBase, Dates, EzXML

# Include all module files
include("SDMXDataSources.jl")
include("SDMXPromptingIntegration.jl")
include("SDMXLLMIntegration.jl")
include("SDMXMetadataContext.jl")
include("SDMXEnhancedTransformation.jl")
include("SDMXMappingInference.jl")
include("SDMXScriptGeneration.jl")
include("SDMXWorkflow.jl")
include("SDMXLLMPipelineOps.jl")
include("SDMXPromptGeneration.jl")

# Data source abstractions (moved from SDMX.jl)
export DataSource, FileSource, NetworkSource, MemorySource
export CSVSource, ExcelSource, URLSource, DataFrameSource
export read_data, source_info, validate_source, data_source

# PromptingTools.jl integration (recommended)
export LLMProvider, ScriptStyle, llm_provider, script_style_enum
export OPENAI, ANTHROPIC, OLLAMA, MISTRAL, AZURE_OPENAI
export DATAFRAMES, TIDIER, MIXED
export SDMXMappingResult, SDMXTransformationScript, ExcelStructureAnalysis
export analyze_excel_with_ai, infer_column_mappings, generate_transformation_script
export setup_sdmx_llm, create_mapping_template, create_script_template, create_excel_analysis_template, ai_sdmx_workflow
export prepare_data_preview, prepare_schema_context, prepare_source_structure

# Modern LLM integration using PromptingTools.jl
export ExcelAnalysis, SheetInfo, CellRange, SDMX_PROVIDERS
export sdmx_aigenerate, sdmx_aiextract, infer_sdmx_column_mappings
export analyze_excel_structure, extract_excel_metadata, detect_data_ranges, analyze_pivoting_needs

# Enhanced metadata context and transformation
export SDMXStructuralContext, DataSourceContext, TransformationContext
export extract_structural_context, extract_data_source_context, build_transformation_context
export generate_enhanced_transformation_script

# Advanced mapping inference
export MappingCandidate, MappingConfidence, AdvancedMappingResult, InferenceEngine
export create_inference_engine, infer_advanced_mappings, validate_mapping_quality
export suggest_value_transformations, analyze_mapping_coverage, learn_from_feedback
export fuzzy_match_score, analyze_value_patterns, detect_hierarchical_relationships

# Script generation
export ScriptTemplate, TransformationStep, GeneratedScript, ScriptGenerator
export create_script_generator, get_script_templates
export build_transformation_steps, create_template_prompt, validate_generated_script
export add_custom_template, preview_script_output

# Workflow orchestration
export SDMXWorkflow, WorkflowStep, WorkflowConfig, WorkflowResult
export create_workflow, execute_workflow, add_custom_step, get_workflow_status
export save_workflow_state, load_workflow_state, generate_workflow_report
export batch_process_datasets, create_automated_pipeline

# Comprehensive prompt generation
export PromptTemplate, TransformationScenario, ComprehensivePrompt
export create_prompt_template, build_comprehensive_prompt, generate_transformation_prompt
export TIDIER_TEMPLATE, DATAFRAMES_TEMPLATE, MIXED_TEMPLATE
export EXCEL_SCENARIO, CSV_SCENARIO, PIVOTING_SCENARIO, CODE_MAPPING_SCENARIO
export build_sdmx_context_section, build_source_analysis_section, build_code_mapping_section

end # module SDMXLLM
