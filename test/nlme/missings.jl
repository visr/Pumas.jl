using Pumas, Test

@testset "Test with missing values" begin

  data         = read_pumas(example_data("sim_data_model1"))
  data_missing = read_pumas(example_data("sim_data_model1"))

  # Make a missing observation
  push!(data_missing[1].observations.dv, missing)
  push!(data_missing[1].time, 2)

  model = Dict()

  model["additive"] = @model begin
    @param begin
        θ ∈ RealDomain()
        Ω ∈ PDiagDomain(1)
        σ ∈ RealDomain(lower=0.0001)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ * exp(η[1])
        V  = 1.0
    end

    @vars begin
        conc = Central / V
    end

    @dynamics ImmediateAbsorptionModel

    @derived begin
        dv ~ @. Normal(conc, σ)
    end
  end

  model["proportional"] = @model begin
    @param begin
        θ ∈ RealDomain()
        Ω ∈ PDiagDomain(1)
        σ ∈ RealDomain(lower=0.0001)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ * exp(η[1])
        V  = 1.0
    end

    @vars begin
        conc = Central / V
    end

    @dynamics ImmediateAbsorptionModel

    @derived begin
        dv ~ @. Normal(conc, conc*σ)
    end
  end

  model["exponential"] = @model begin
    @param begin
        θ ∈ RealDomain()
        Ω ∈ PDiagDomain(1)
        σ ∈ RealDomain(lower=0.0001)
    end

    @random begin
        η ~ MvNormal(Ω)
    end

    @pre begin
        CL = θ * exp(η[1])
        V  = 1.0
    end

    @vars begin
        conc = Central / V
    end

    @dynamics ImmediateAbsorptionModel

    @derived begin
        dv ~ @. LogNormal(log(conc), σ)
    end
  end

  param = (θ=0.5, Ω=Diagonal([0.04]), σ=0.01)

  @testset "testing model: $_model, with $_approx approximation" for
    _model in ("additive", "proportional", "exponential"),
      _approx in (Pumas.FO(), Pumas.FOCE(), Pumas.FOCEI(), Pumas.Laplace(), Pumas.LaplaceI())
    # LaplaceI and proportional is very unstable and succeeds/fails depending on architecture
    # so we can't mark this as @test_broken
    if _model != "proportional" || _approx != Pumas.LaplaceI()
      @test deviance(fit(model[_model], data, param, _approx)) == deviance(fit(model[_model], data_missing, param, _approx))
    end
  end
end
