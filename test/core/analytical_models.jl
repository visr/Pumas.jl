using Pumas

@test ImmediateAbsorptionModel().syms == [:Central]
@test Pumas.DiffEqBase.has_syms(ImmediateAbsorptionModel())
@test OneCompartmentModel().syms == [:Depot, :Central]
@test Pumas.DiffEqBase.has_syms(OneCompartmentModel())
@test OneCompartmentParallelModel().syms == [:Depot1, :Depot2, :Central]
@test Pumas.DiffEqBase.has_syms(OneCompartmentParallelModel())
