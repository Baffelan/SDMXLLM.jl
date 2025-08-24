"""
Extension for GoogleGenAI support in SDMXLLM

This extension is loaded automatically when GoogleGenAI is imported,
enabling Google Gemini models in SDMXLLM workflows.
"""

module SDMXGoogleGenAIExt

using SDMXLLM
using GoogleGenAI

# This extension ensures GoogleGenAI is available for PromptingTools.jl
# when using :google provider in SDMXLLM functions

# No additional code needed - the mere loading of GoogleGenAI
# activates the PromptingTools.jl extension for Google models

end # module SDMXGoogleGenAIExt