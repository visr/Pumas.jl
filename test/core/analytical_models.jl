using Pumas

ImmediateAbsorptionModel().syms == [:Central]
OneCompartmentModel().syms == [:Depot, :Central]
OneCompartmentParallelModel().syms == [:Depot1, :Depot2, :Central]
