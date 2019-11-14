using Test
using Pumas
using Random
Random.seed!(4)
data = read_pumas(example_data("sim_data_model1"))

#likelihood tests from NLME.jl
#-----------------------------------------------------------------------# Test 1
mdsl1 = @model begin
    @param begin
        θ ∈ VectorDomain(1, init=[0.5])
        Ω ∈ PDiagDomain(init=[0.04])
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

pnpde = [Pumas.npde(mdsl1, data[i], param, 10000) for i in 1:10]

pnpde_ref = [[0.18962882237487352, 1.7201784995140674],
 [-1.3773631497105263, -0.252570666427693],
 [0.2976111022334799, 0.6489046638504296],
 [0.39153746690120006, 1.383211580197489],
 [0.617962588415399, -1.6901461375274704],
 [0.8405501419786955, -0.7461116042451951],
 [0.23939410585994175, 1.696453654440568],
 [-0.17662876259615526, 0.6610192009497576],
 [-1.3943895566974136, 0.9928152433591684],
 [0.9047459346629684, 0.3496518412286771]]

for (_pnpde, _ref) in zip(pnpde, pnpde_ref)
  @test _pnpde.dv == _ref
end

[Pumas.epred(
  mdsl1,
  data[i],
  param,
  Pumas.TransformVariables.transform(
    Pumas.totransform(
      mdsl1.random(param)
    ),
    Pumas._orth_empirical_bayes(
      mdsl1,
      data[i],
      param,
      Pumas.FOCE()
    )
  ),
  10000
) for i in 1:10]
[Pumas.cpred(mdsl1, data[i], param) for i in 1:10]
[Pumas.cpredi(mdsl1, data[i], param) for i in 1:10]

@testset "pred" for
    (sub_pred, dt) in zip([[10.0000000, 6.06530660],
                           [10.0000000, 6.06530660],
                           [10.0000000, 6.06530660],
                           [10.0000000, 6.06530660],
                           [10.0000000, 6.06530660],
                           [10.0000000, 6.06530660],
                           [10.0000000, 6.06530660],
                           [10.0000000, 6.06530660],
                           [10.0000000, 6.06530660],
                           [10.0000000, 6.06530660]], data)

    @test Pumas.pred(mdsl1, dt, param).dv ≈ sub_pred rtol=1e-6
end

@testset "wres" for
    (sub_wres, dt) in zip([[ 0.180566054, 1.74797817 ],
                           [-1.35845124 ,-0.274456699],
                           [ 0.310535666, 0.611240923],
                           [ 0.394652252, 1.41153536 ],
                           [ 0.607473539,-1.68539881 ],
                           [ 0.858874613,-0.769228457],
                           [ 0.245708974, 1.74827643 ],
                           [-0.169086986, 0.608506828],
                           [-1.38172560 , 0.984121759],
                           [ 0.905043866, 0.302785305]], data)

    @test Pumas.wres(mdsl1, dt, param).dv ≈ sub_wres
end

@testset "cwres" for
    (sub_cwres, dt) in zip([[ 0.180566054, 1.75204867 ],
                            [-1.35845124 ,-0.274353057],
                            [ 0.310535666, 0.611748221],
                            [ 0.394652252, 1.41420526 ],
                            [ 0.607473539,-1.68142230 ],
                            [ 0.858874613,-0.768408883],
                            [ 0.245708974, 1.75234831 ],
                            [-0.169086986, 0.609009620],
                            [-1.38172560 , 0.985428904],
                            [ 0.905043866, 0.302910385]], data)

    @test Pumas.cwres(mdsl1, dt, param).dv ≈ sub_cwres
end

@testset "cwresi" for
    (sub_cwresi, dt) in zip([[ 0.180566054, 1.6665779  ],
                             [-1.35845124 ,-0.278938663],
                             [ 0.310535666, 0.605059261],
                             [ 0.394652252, 1.36101861 ],
                             [ 0.607473539,-1.74177468 ],
                             [ 0.858874613,-0.789814478],
                             [ 0.245708974, 1.6668457  ],
                             [-0.169086986, 0.602404841],
                             [-1.3817256  , 0.962485383],
                             [ 0.905043866, 0.302554671]], data)

   @test Pumas.cwresi(mdsl1, dt, param).dv ≈ sub_cwresi rtol=1e-6
end

@testset "iwres" for
    (sub_iwres, dt) in zip([[ 0.180566054, 1.83329497 ],
                            [-1.35845124 ,-0.287852614],
                            [ 0.310535666, 0.641074888],
                            [ 0.394652252, 1.48043078 ],
                            [ 0.607473539,-1.76766118 ],
                            [ 0.858874613,-0.806773612],
                            [ 0.245708974, 1.83360779 ],
                            [-0.169086986, 0.638207345],
                            [-1.38172560 , 1.03215561 ],
                            [ 0.905043866, 0.317563907]], data)

    @test Pumas.iwres(mdsl1, dt, param).dv ≈ sub_iwres
end

@testset "icwres" for
    (sub_icwres, dt) in zip([[ 0.180566054, 1.67817359 ],
                             [-1.35845124 ,-0.261387432],
                             [ 0.310535666, 0.584242548],
                             [ 0.394652252, 1.35343113 ],
                             [ 0.607473539,-1.59554270 ],
                             [ 0.858874613,-0.731080208],
                             [ 0.245708974, 1.67846183 ],
                             [-0.169086986, 0.581622858],
                             [-1.38172560 , 0.942045331],
                             [ 0.905043866, 0.289051786]], data)

    @test Pumas.icwres(mdsl1, dt, param).dv ≈ sub_icwres rtol=1e-5
end

@testset "icwresi" for
    (sub_icwresi, dt) in zip([[ 0.180566054, 1.56991766 ],
                              [-1.35845124 ,-0.236161082],
                              [ 0.310535666, 0.595884676],
                              [ 0.394652252, 1.29087676 ],
                              [ 0.607473539,-1.71221172 ],
                              [ 0.858874613,-0.734054331],
                              [ 0.245708974, 1.57016202 ],
                              [-0.169086986, 0.593425217],
                              [-1.38172560 , 0.925641802],
                              [ 0.905043866, 0.314343255]], data)

    @test Pumas.icwresi(mdsl1, dt, param).dv ≈ sub_icwresi rtol=1e-5
end

[Pumas.eiwres(mdsl1, data[i], param, 10000).dv for i in 1:10]

param = (θ = [0.340689], Ω = Diagonal([0.000004]), Σ = 0.0752507)
@test ηshrinkage(mdsl1, data, param, Pumas.FOCEI()).η ≈ [0.997574] rtol=1e-6
ϵshrinkage(mdsl1, data, param, Pumas.FOCEI())
@test aic(mdsl1, data, param, Pumas.FOCEI()) ≈ 94.30968177483996 rtol=1e-6 #regression test
@test bic(mdsl1, data, param, Pumas.FOCEI()) ≈ 96.30114632194794 rtol=1e-6 #regression test
