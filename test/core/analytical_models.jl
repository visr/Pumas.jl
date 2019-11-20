using Pumas, Test

@test ImmediateAbsorptionModel().syms == [:Central]
@test Pumas.DiffEqBase.has_syms(ImmediateAbsorptionModel())
@test OneCompartmentModel().syms == [:Depot, :Central]
@test Pumas.DiffEqBase.has_syms(OneCompartmentModel())
@test OneCompartmentParallelModel().syms == [:Depot1, :Depot2, :Central]
@test Pumas.DiffEqBase.has_syms(OneCompartmentParallelModel())

# test for #732
# If two doses were given at the same time or if a rate dosage regimen was
# specified after an instant dosage regimen, the instant dosage would be overwritten.
# We simply test that it is accumulated at the time of dose.
model732 = @model begin
  @pre begin
    Ka = 0.01
    CL = 1.0
    V = 3.0
  end
  @dynamics OneCompartmentModel
  @derived begin
    dv ~ @. Normal(Central/V)
  end
end


doses_R = DosageRegimen(43, cmt=1, time=3, ii=12, addl=0, rate=5)
doses_D = DosageRegimen(43, cmt=1, time=3, ii=12, addl=0, rate=0)

doses_DD = DosageRegimen(doses_D, doses_D)
doses_DR = DosageRegimen(doses_D, doses_R)
doses_RD = DosageRegimen(doses_R, doses_D)

dose = doses_RD
pop     = Population(map(i -> Subject(id=i, evs=dose),1:15))
_simobs = simobs(model732, pop, NamedTuple())
simdf = DataFrame(_simobs)
data = read_pumas(simdf)
for i = 1:15
  sol = solve(model732, data[i], NamedTuple())

  @test sol(3.0)[1] ≈ 43.0
end

dose = doses_DD
pop     = Population(map(i -> Subject(id=i, evs=dose),1:15))
_simobs = simobs(model732, pop, NamedTuple())
simdf = DataFrame(_simobs)
data = read_pumas(simdf)
for i = 1:15
  sol = solve(model732, data[i], NamedTuple())

  @test sol(3.0)[1] ≈ 86.0
end


dose = doses_DR
pop     = Population(map(i -> Subject(id=i, evs=dose),1:15))
_simobs = simobs(model732, pop, NamedTuple())
simdf = DataFrame(_simobs)
data = read_pumas(simdf)
for i = 1:15
  sol = solve(model732, data[i], NamedTuple())

  @test sol(3.0)[1] ≈ 43.0
end
