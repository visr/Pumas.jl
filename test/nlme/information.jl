using Pumas, LinearAlgebra, Test

# Make sure that PUMASMODELS dict is loaded
if !isdefined(Main, :PUMASMODELS)
  include("testmodels/testmodels.jl")
end

@testset "Test informationmatrix with warfarin data" begin

  warfarin = PUMASMODELS["1cpt"]["oral"]["normal_additive"]["warfarin"]
  data, model, param = warfarin["data"], warfarin["analytical"], warfarin["param"]

  @test logdet(
    sum(
      Pumas._expected_information(
        model,
        d,
        param,
        Pumas._orth_empirical_bayes(model, d, param, Pumas.FO()),
        Pumas.FO()
      ) for d in data)) ≈ 53.8955 rtol=1e-6

  ft = fit(model, data, param, Pumas.FO())

  @test logdet(informationmatrix(ft)) isa Number

end

@testset "Multiple dvs. (The HCV model)" begin

  hcv = PUMASMODELS["misc"]["HCV"]

  @test logdet(Pumas._expected_information_fd(hcv["solver"], hcv["data"], hcv["param"], zeros(7), Pumas.FO())*30) ≈ 92.21128100630904 rtol=1e-7
end
