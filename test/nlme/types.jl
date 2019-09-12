using Test
using Pumas, LinearAlgebra

@testset "args and kwargs" begin
data = read_pumas(example_data("sim_data_model1"))
#-----------------------------------------------------------------------# Test 1
mdsl1 = @model begin
    @param begin
        θ ∈ VectorDomain(1, init=[0.5])
        Ω ∈ ConstDomain(Diagonal([0.04]))
        Σ ∈ ConstDomain(0.1)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ[1] * exp(η[1])
        V  = 1.0
    end

    @vars begin
        conc = Central / V
    end

    @dynamics ImmediateAbsorptionModel

    @derived begin
        dv ~ @. Normal(conc,conc*sqrt(Σ)+eps())
    end
end

param = init_param(mdsl1)

ft_no_args = fit(mdsl1, data, param, Pumas.FOCEI())
@test isempty(pairs(ft_no_args.args))
@test isempty(pairs(ft_no_args.kwargs))

ft_alg_kwargs = fit(mdsl1, data, param, Pumas.FOCEI(); alg=Rosenbrock23())
@test isempty(pairs(ft_alg_kwargs.args))
kwarg_pairs = pairs(ft_alg_kwargs.kwargs)
@test keys(kwarg_pairs) == (:alg,)
@test kwarg_pairs[1] == Rosenbrock23()

end
