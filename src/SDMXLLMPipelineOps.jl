"""
LLM-specific Pipeline Operators for SDMXLLM.jl
"""

# Dependencies loaded at package level

export map_with, generate_with

"""
    map_with(engine::InferenceEngine, schema::DataflowSchema) -> Function

Creates a mapping function that can be used in pipelines.

# Example
```julia
mappings = profile |> map_with(engine, schema)
```
"""
function map_with(engine::InferenceEngine, schema::DataflowSchema)
    return profile -> infer_advanced_mappings(engine, profile, schema)
end

"""
    generate_with(generator::ScriptGenerator, schema::DataflowSchema; kwargs...) -> Function

Creates a script generation function that can be used in pipelines.

# Example
```julia
script = mappings |> generate_with(generator, schema)
```
"""
function generate_with(generator::ScriptGenerator, schema::DataflowSchema; kwargs...)
    return mappings -> begin
        # We need a profile for script generation, so this is a bit more complex
        # This function expects a tuple (profile, mappings) as input
        if isa(mappings, Tuple)
            profile, mappings_result = mappings
            return generator(profile, schema, mappings_result; kwargs...)
        else
            throw(ArgumentError("generate_with expects a tuple (profile, mappings) as input"))
        end
    end
end

"""
    ⇒(data::DataFrame, generator::ScriptGenerator) -> GeneratedScript
    data ⇒ generator

Data flow operator for script generation (requires additional context).
Note: This is a simplified version - full generation requires schema and mappings.

# Example
```julia
# This would need to be extended with proper context
# script = my_data ⇒ generator
```
"""
function ⇒(data::DataFrame, generator::ScriptGenerator)
    # This is a simplified version - in practice you'd need schema and mappings
    throw(ArgumentError("Script generation requires schema and mappings. Use the full generator interface."))
end
