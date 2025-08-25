using Test
using SDMX
using SDMXLLM
using DataFrames

@testset "SDMXLLM Basic Tests" begin
    
    @testset "LLM Setup" begin
        # Test that setup_sdmx_llm works
        provider = setup_sdmx_llm(:ollama; model="llama2")
        @test provider == SDMXLLM.OLLAMA
        
        # Test that enums are exported
        @test SDMXLLM.OLLAMA isa SDMXLLM.LLMProvider
        @test SDMXLLM.OPENAI isa SDMXLLM.LLMProvider
    end
    
    @testset "Data Sources" begin
        # Test CSV source with actual file
        test_file = "/home/gvdr/reps/julia_sdmx/SDMX.jl/test/fixtures/sample_data.csv"
        if isfile(test_file)
            csv_source = CSVSource(test_file)
            @test csv_source.path == test_file
            @test csv_source isa DataSource
        else
            @test_skip "Test file not found"
        end
        
        # Test DataFrame source
        df = DataFrame(a=[1,2,3], b=[4,5,6])
        df_source = DataFrameSource(df, "test_df")
        @test df_source.data == df
        @test df_source.name == "test_df"
    end
    
    @testset "Basic Functions" begin
        # Test that key functions are exported and accessible
        @test isdefined(SDMXLLM, :setup_sdmx_llm)
        @test isdefined(SDMXLLM, :create_workflow)
        @test isdefined(SDMXLLM, :generate_transformation_script)
        @test isdefined(SDMXLLM, :infer_column_mappings)
        @test isdefined(SDMXLLM, :create_inference_engine)
    end
    
    @testset "Inference Engine" begin
        # Test inference engine creation
        engine = create_inference_engine(fuzzy_threshold=0.6, min_confidence=0.2)
        @test engine.fuzzy_threshold == 0.6
        @test engine.min_confidence == 0.2
    end
    
    @testset "Fuzzy Matching" begin
        # Test fuzzy match scoring
        score = fuzzy_match_score("country", "COUNTRY")
        @test score > 0.9  # Should be very high for case differences
        
        score2 = fuzzy_match_score("geo", "GEO_PICT")
        @test score2 > 0.3  # Should find some similarity
    end
    
end

println("\n✓ All basic tests passed!")