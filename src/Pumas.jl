module Pumas

using DiffEqDiffTools, Reexport, StatsBase,
      StaticArrays, Distributed, LabelledArrays, GLM,
      TreeViews, CSV, ForwardDiff, DiffResults, Optim, PDMats,
      Missings, RecipesBase, RecursiveArrayTools, Quadrature,
      Statistics, DiffEqSensitivity
using LinearAlgebra
using AdvancedHMC: DiagEuclideanMetric, Hamiltonian, NUTS, Leapfrog, find_good_eps, StanHMCAdaptor, Preconditioner, NesterovDualAveraging
import MCMCChains: Chains

import DiffResults: DiffResult

import DataInterpolations

@reexport using OrdinaryDiffEq, Unitful
@reexport using Distributions, DataFrames

const Numeric = Union{AbstractVector{<:Number}, Number}

function opt_minimizer end
include("nca/NCA.jl")
include("ivivc/IVIVC.jl")

include("data_parsing/data_types.jl")
include("data_parsing/data_read.jl")

include("dsl/model_macro.jl")

include("models/params.jl")
include("models/simulated_observations.jl")
include("models/model_api.jl")
include("models/model_utils.jl")

include("estimation/transforms.jl")
include("estimation/likelihoods.jl")
include("estimation/bayes.jl")
include("estimation/diagnostics.jl")
include("estimation/gsa.jl")
include("estimation/show.jl")

include("analytical_solutions/standard_models.jl")
include("analytical_solutions/analytical_problem.jl")
include("analytical_solutions/analytical_solution_type.jl")

include("simulate_methods/utils.jl")
include("simulate_methods/diffeqs.jl")
include("simulate_methods/analytical.jl")

include("uq/expectation.jl")

include("plotting/plotting.jl")

@reexport using .NCA

example_data(filename) = joinpath(joinpath(@__DIR__, ".."),"examples/"*filename*".csv")

export Subject, Population, DosageRegimen
export PumasModel, init_param, init_randeffs, sample_randeffs
export simobs, pre
export tad, eventnum
export conditional_nll
export predict, residuals, wresiduals, empirical_bayes
export ηshrinkage, ϵshrinkage
export read_pumas, example_data
export @model, @nca, @tvcov
# From StatsBase
export fit, stderror, vcov, aic, bic, deviance, informationmatrix
export infer, inspect
export gsa
export mean, std, var, coef
export expectation, KoopmanExpectation, MonteCarloExpectation
# From LinearAlgebra
export diagm, Diagonal, I

end # module
