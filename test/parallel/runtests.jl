# Do not put into a module because processes are spawned
@time @testset "Parallelism Tests" begin
    @time @testset "Theophylline" begin
        include("theophylline.jl")
    end

    @time @testset "Simple model" begin
        include("simple_model.jl")
    end
end
