using Test
using Pumas

@testset "likelihood tests from NLME.jl" begin
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

for (ηstar, dt) in zip([-0.1007, 0.0167, -0.0363, -0.0820, 0.1061, 0.0473, -0.1007, -0.0361, -0.0578, -0.0181], data)
    @test (sqrt(param.Ω)*Pumas._orth_empirical_bayes(mdsl1, dt, param, Pumas.Laplace()))[1] ≈ ηstar rtol=1e-2
end
for (ηstar, dt) in zip([-0.114654,0.0350263,-0.024196,-0.0870518,0.0750881,0.059033,-0.114679,-0.023992,-0.0528146,-0.00185361], data)
    @test (sqrt(param.Ω)*Pumas._orth_empirical_bayes(mdsl1, dt, param, Pumas.LaplaceI()))[1] ≈ ηstar rtol=1e-3
end

@test deviance(mdsl1, data, param, Pumas.FO())        ≈ 56.474912258255571 rtol=1e-6
@test deviance(mdsl1, data, param, Pumas.FOCE())      ≈ 56.476216665029462 rtol=1e-6
@test deviance(mdsl1, data, param, Pumas.FOCEI())     ≈ 56.410938825140313 rtol=1e-6
@test deviance(mdsl1, data, param, Pumas.Laplace())   ≈ 56.613069180382027 rtol=1e-6
@test deviance(mdsl1, data, param, Pumas.LaplaceI())  ≈ 56.810343602063618 rtol=1e-6
@test deviance(mdsl1, data, param, Pumas.LLQuad()) ≈ 56.92491372848633  rtol=1e-6 #regression test

ft = fit(mdsl1, data, param, Pumas.FOCEI())
@test sprint((io, t) -> show(io, MIME"text/plain"(), t), ft) ==
"""
FittedPumasModel

Successful minimization:                true

Likelihood approximation:        Pumas.FOCEI
Objective function value:          45.623789
Total number of observation records:      20
Number of active observation records:     20
Number of subjects:                       10

------------------
         Estimate
------------------
θ₁        0.36476
Ω₁,₁      0.04
Σ         0.1
------------------
"""
end# testset
