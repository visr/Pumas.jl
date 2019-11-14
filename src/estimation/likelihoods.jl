# Some PDMats piracy. Should be possible to remove once we stop using MvNormal
PDMats.unwhiten(C::PDiagMat, x::StridedVector) = sqrt.(C.diag) .* x
PDMats.unwhiten(C::PDiagMat, x::AbstractVector) = sqrt.(C.diag) .* x

abstract type LikelihoodApproximation end
struct NaivePooled <: LikelihoodApproximation end
struct TwoStage <: LikelihoodApproximation end
struct FO <: LikelihoodApproximation end
struct FOI <: LikelihoodApproximation end
struct FOCE <: LikelihoodApproximation end
struct FOCEI <: LikelihoodApproximation end
struct Laplace <: LikelihoodApproximation end
struct LaplaceI <: LikelihoodApproximation end
struct LLQuad{T} <: LikelihoodApproximation
  quadalg::T
end
LLQuad() = LLQuad(HCubatureJL())

zval(d) = 0.0
zval(d::Distributions.Normal{T}) where {T} = zero(T)

"""
    _lpdf(d,x)

The log pdf: this differs from `Distributions.logdpf` definintion in a couple of ways:
- if `d` is a non-distribution it assumes the Dirac distribution.
- if `x` is `NaN` or `Missing`, it returns 0.
- if `d` is a `NamedTuple` of distributions, and `x` is a `NamedTuple` of observations, it computes the sum of the observed variables.
"""
_lpdf(d::Number, x::Number) = d == x ? 0.0 : -Inf
_lpdf(d::ConstDomain, x) = _lpdf(d.val, x)
_lpdf(d::Distributions.Sampleable, x::Missing) = zval(d)
_lpdf(d::Distributions.UnivariateDistribution, x::AbstractVector) = sum(t -> _lpdf(d, t), x)
_lpdf(d::Distributions.MultivariateDistribution, x::AbstractVector) = logpdf(d,x)
_lpdf(d::Distributions.Sampleable, x::PDMat) = logpdf(d,x)
_lpdf(d::Distributions.Sampleable, x::Number) = isnan(x) ? zval(d) : logpdf(d,x)
_lpdf(d::Constrained, x) = _lpdf(d.dist, x)
_lpdf(d::Domain, x) = 0.0
function _lpdf(ds::AbstractVector, xs::AbstractVector)
  if length(ds) != length(xs)
    throw(DimensionMismatch("vectors must have same length"))
  end
  l = _lpdf(ds[1], xs[1])
  @inbounds for i in 2:length(ds)
    l += _lpdf(ds[i], xs[i])
  end
  return l
end

Base.@pure function _intersect_names(an::Tuple{Vararg{Symbol}}, bn::Tuple{Vararg{Symbol}})
    names = Symbol[]
    for n in an
        if Base.sym_in(n, bn)
            push!(names, n)
        end
    end
    (names...,)
end

@generated function _lpdf(ds::NamedTuple{Nds}, xs::NamedTuple{Nxs}) where {Nds, Nxs}
  _names = _intersect_names(Nds, Nxs)
  quote
    names = $_names
    l = _lpdf(getindex(ds, names[1]), getindex(xs, names[1]))
    for i in 2:length(names)
      name = names[i]
      l += _lpdf(getindex(ds, name), getindex(xs, name))
    end
    return l
  end
end

"""
    conditional_nll(m::PumasModel, subject::Subject, param, randeffs, args...; kwargs...)

Compute the conditional negative log-likelihood of model `m` for `subject` with parameters `param` and
random effects `randeffs`. `args` and `kwargs` are passed to ODE solver. Requires that
the derived produces distributions.
"""
@inline function conditional_nll(m::PumasModel,
                                 subject::Subject,
                                 param::NamedTuple,
                                 randeffs::NamedTuple,
                                 args...;
                                 kwargs...)
    dist = _derived(m, subject, param, randeffs, args...; kwargs...)
    conditional_nll(m, subject, param, randeffs, dist)
end

@inline function conditional_nll(m::PumasModel,
                                 subject::Subject,
                                 param::NamedTuple,
                                 randeffs::NamedTuple,
                                 dist::NamedTuple)

  collated_numtype = numtype(m.pre(param, randeffs, subject))

  if any(d->d isa Nothing, dist)
    return collated_numtype(Inf)
  end

  clean_dist = NamedTuple{keys(subject.observations)}(dist)
  ll = _lpdf(clean_dist, subject.observations)::collated_numtype
  return -ll
end

conditional_nll(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                randeffs::NamedTuple,
                approx::Union{FOI,FOCEI,LaplaceI},
                args...; kwargs...) = conditional_nll(m, subject, param, randeffs, args...; kwargs...)

function conditional_nll(m::PumasModel,
                         subject::Subject,
                         param::NamedTuple,
                         randeffs::NamedTuple,
                         approx::Union{FO,FOCE,Laplace},
                         args...; kwargs...)

  collated_numtype = numtype(m.pre(param, randeffs, subject))
  dist = _derived(m, subject, param, randeffs, args...;kwargs...)

  if any(d->d isa Nothing, dist)
    return collated_numtype(Inf)
  end

  clean_dist = NamedTuple{keys(subject.observations)}(dist)

  # this can potentially be calculated multiple times (in ∂l∂η as well), is that
  # a performance concern?
  homoscedastic_check = map(_is_homoscedastic, clean_dist)
  # If homoscedastic, simply call the generic conditional_nll
  dist0 = all(homoscedastic_check) ? dist : _derived(m, subject, param, map(zero, randeffs), args...; kwargs...)
  σ_dists = map(NamedTuple{keys(clean_dist)}(keys(clean_dist))) do dv_key
              if homoscedastic_check[dv_key]
                return dist[dv_key]
              else
                return _ofdisttype.(dist[dv_key], dist0[dv_key])
              end
            end

  return conditional_nll(m, subject, param, randeffs, σ_dists)::collated_numtype
end

"""
    penalized_conditional_nll(m::PumasModel, subject::Subject, param::NamedTuple, randeffs::NamedTuple, args...; kwargs...)

Compute the penalized conditional negative log-likelihood. This is the same as
[`conditional_nll`](@ref), except that it incorporates the penalty from the prior
distribution of the random effects.

Here `param` can be either a `NamedTuple` or a vector (representing a transformation of the
random effects to Cartesian space).
"""
function penalized_conditional_nll(m::PumasModel,
                                   subject::Subject,
                                   param::NamedTuple,
                                   randeffs::NamedTuple,
                                   args...;kwargs...)


  randeffstransform = totransform(m.random(param))
  vrandeffsorth = TransformVariables.inverse(randeffstransform, vrandeffsorth)

  return penalized_conditional_nll(m, subject, param, vrandeffsorth, args...; kwargs...)
end

function penalized_conditional_nll(m::PumasModel,
                                   subject::Subject,
                                   param::NamedTuple,
                                   vrandeffsorth::AbstractVector,
                                   args...;kwargs...)

  # First evaluate the penalty (wihout the π term)
  nl_randeffs = vrandeffsorth'vrandeffsorth/2

  # If penalty is too large (likelihood would be Inf) then return without evaluating conditional likelihood
  if nl_randeffs > log(floatmax(Float64))
    return nl_randeffs
  else
    randeffstransform = totransform(m.random(param))
    randeffs = TransformVariables.transform(randeffstransform, vrandeffsorth)
    return conditional_nll(m, subject, param, randeffs, args...;kwargs...) + nl_randeffs
  end
end

function _initial_randeffs(m::PumasModel, param::NamedTuple)
  rfxset = m.random(param)
  p = TransformVariables.dimension(totransform(rfxset))

  # Temporary workaround for incorrect initialization of derivative storage in NLSolversBase
  # See https://github.com/JuliaNLSolvers/NLSolversBase.jl/issues/97
  T = promote_type(numtype(param), numtype(param))
  zeros(T, p)
end

#     _orth_empirical_bayes(model, subject, param, approx, ...)
# The point-estimate the orthogonalized random effects (being the mode of the empirical
# Bayes estimate) of a particular subject at a particular parameter values. The result is
# returned as a vector (transformed into Cartesian space).

function _orth_empirical_bayes(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  approx::LikelihoodApproximation,
  args...; kwargs...)

  initial_vrandefforth = _initial_randeffs(m, param)

  return _orth_empirical_bayes!(initial_vrandefforth, m, subject, param, approx, args...; kwargs...)
end

function _orth_empirical_bayes!(
  vrandeffsorth::AbstractVector,
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  ::Union{FO,FOI,LLQuad},
  args...; kwargs...)

  fill!(vrandeffsorth, 0)
  return vrandeffsorth
end

function _orth_empirical_bayes!(
  vrandeffsorth::AbstractVector,
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  approx::Union{FOCE,FOCEI,Laplace,LaplaceI},
  args...;
  # We explicitly use reltol to compute the right step size for finite difference based gradient
  reltol=DEFAULT_ESTIMATION_RELTOL,
  fdtype=Val{:central}(),
  fdrelstep=_fdrelstep(m, param, reltol, fdtype),
  kwargs...)

  cost = vηorth -> penalized_conditional_nll(
    m,
    subject,
    param,
    vηorth,
    approx,
    args...;
    reltol=reltol,
    kwargs...)

  vrandeffsorth .= Optim.minimizer(
    Optim.optimize(
      cost,
      vrandeffsorth,
      NewtonTrustRegion(),
      Optim.Options(
        show_trace=false,
        extended_trace=true,
        g_tol=1e-5
      );
      autodiff=:forward))
  return vrandeffsorth
end


"""
    empirical_bayes_dist(model, subject, param, vrandeffsorth::AbstractVector, approx, ...)
Estimate the distribution of random effects (typically a Normal approximation of the
empirical Bayes posterior) of a particular subject at a particular parameter values. The
result is returned as a `MvNormal`.
"""
empirical_bayes_dist

function empirical_bayes_dist(m::PumasModel,
                              subject::Subject,
                              param::NamedTuple,
                              vrandeffsorth::AbstractVector,
                              approx::Union{FOCE,FOCEI,Laplace,LaplaceI},
                              args...; kwargs...)

  parset = m.random(param)
  trf = totransform(parset)
  dv_∂²l∂η² = ∂²l∂η²(m, subject, param, vrandeffsorth, approx, args...; kwargs...)
  # 3 is W, should we make is a named tuple with l, g, W?
  V = inv(sum(_dv_∂²l∂η²[3] for _dv_∂²l∂η² in dv_∂²l∂η²) + I)

  i = 1
  tmp = map(trf.transformations) do t
    d = TransformVariables.dimension(t)
    v = view(vrandeffsorth, i:(i + d - 1))
    μ = TransformVariables.transform(t, v)
    Vᵢ = V[i:(i + d - 1), i:(i + d - 1)]
    i += d
    if t isa MvNormalTransform
      if t.d.Σ isa PDiagMat
        U = Diagonal(sqrt.(t.d.Σ.diag))
      else
        U = cholesky(t.d.Σ)
      end
      return MvNormal(μ, Symmetric(U'*(Vᵢ*U)))
    elseif t isa NormalTransform
      return Normal(μ, std(t.d)*sqrt(Vᵢ[1,1]))
    else
      throw("transformation not currently covered")
    end
  end

  return tmp
end

"""
    marginal_nll(model, subject, param[, param], approx, ...)
    marginal_nll(model, population, param, approx, ...)

Compute the marginal negative loglikelihood of a subject or dataset, using the integral
approximation `approx`. If no random effect (`param`) is provided, then this is estimated
from the data.

See also [`deviance`](@ref).
"""
function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      randeffs::NamedTuple,
                      approx::LikelihoodApproximation,
                      args...; kwargs...)

  if approx isa NaivePooled
    vrandeffsorth = []
  else
    rfxset = m.random(param)
    vrandeffsorth = TransformVariables.inverse(totransform(rfxset), randeffs)
  end
  return marginal_nll(m, subject, param, vrandeffsorth, approx, args...; kwargs...)
end

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      approx::LikelihoodApproximation,
                      args...;
                      kwargs...)
  vrandeffsorth = _orth_empirical_bayes(m, subject, param, approx, args...; kwargs...)
  marginal_nll(m, subject, param, vrandeffsorth, approx, args...; kwargs...)
end

function marginal_nll(m::PumasModel,
                      # restrict to Vector to avoid distributed arrays taking
                      # this path
                      population::Vector{<:Subject},
                      args...;
                      parallel_type::ParallelType=Threading,
                      kwargs...)

  nll1 = marginal_nll(m, population[1], args...; kwargs...)
  # Compute first subject separately to determine return type and to return
  # early in case the parameter values cause the likelihood to be Inf. This
  # can e.g. happen if the ODE solver can't solve the ODE for the chosen
  # parameter values.
  if isinf(nll1)
    return nll1
  end

  # The different parallel computations are separated out into functions
  # to make it easier to infer the return types
  if parallel_type === Serial
    return sum(subject -> marginal_nll(m, subject, args...; kwargs...), population)
  elseif parallel_type === Threading
    return _marginal_nll_threads(nll1, m, population, args...; kwargs...)
  elseif parallel_type === Distributed # Distributed
    return _marginal_nll_pmap(nll1, m, population, args...; kwargs...)
  else
    throw(ArgumentError("parallel type $parallel_type not implemented"))
  end
end

function _marginal_nll_threads(nll1::T,
                               m::PumasModel,
                               population::Vector{<:Subject},
                               args...;
                               kwargs...)::T where T

  # Allocate array to store likelihood values for each subject in the threaded
  # for loop
  nlls = fill(T(Inf), length(population) - 1)

  # Run threaded for loop for the remaining subjects
  Threads.@threads for i in 2:length(population)
    nlls[i - 1] = marginal_nll(m, population[i], args...; kwargs...)
  end

  return nll1 + sum(nlls)
end

function _marginal_nll_pmap(nll1::T,
                            m::PumasModel,
                            population::Vector{<:Subject},
                            args...;
                            kwargs...)::T where T

  nlls = convert(Vector{T},
                 pmap(subject -> marginal_nll(m, subject, args...; kwargs...),
                 population[2:length(population)]))

    return nll1 + sum(nlls)
end

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffsorth::AbstractVector,
                      ::NaivePooled,
                      args...; kwargs...)::promote_type(numtype(param), numtype(vrandeffsorth))

  # The negative loglikelihood function. There are no random effects.
  conditional_nll(m, subject, param, NamedTuple(), args...;kwargs...)

end
function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffsorth::AbstractVector,
                      approx::FO,
                      args...; kwargs...)::promote_type(numtype(param), numtype(vrandeffsorth))

  # For FO, the conditional likelihood must be evaluated at η=0
  @assert iszero(vrandeffsorth)

  # Compute the gradient of the likelihood and Hessian approxmation in the random effect vector η
  dv_∂²l∂η² = ∂²l∂η²(m, subject, param, vrandeffsorth, approx, args...; kwargs...)

  sum(map(_dv_∂²l∂η² -> _marginal_nll(_dv_∂²l∂η², approx), dv_∂²l∂η²))
end
# this is the marginal_nll calculation for a single DV
function _marginal_nll(_dv_∂²l∂η², approx::FO)
    nl, dldη, W = _dv_∂²l∂η²
    if isfinite(nl)
      FIW = cholesky(Symmetric(Matrix(I + W)))
      return nl + (- dldη'*(FIW\dldη) + logdet(FIW))/2
    else # conditional likelihood return Inf
      return typeof(nl)(Inf)
    end
end

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffsorth::AbstractVector,
                      approx::Union{FOCE,FOCEI,Laplace,LaplaceI},
                      args...; kwargs...)::promote_type(numtype(param), numtype(vrandeffsorth))

  dv_∂²l∂η² = ∂²l∂η²(m, subject, param, vrandeffsorth, approx, args...; kwargs...)

  sum(map(_dv_∂²l∂η² -> _marginal_nll(_dv_∂²l∂η², vrandeffsorth, approx), dv_∂²l∂η²))
end

function _marginal_nll(_dv_∂²l∂η², vrandeffsorth, approx::Union{FOCE,FOCEI,Laplace,LaplaceI})
  nl, _, W = _dv_∂²l∂η²
  if isfinite(nl)
    # If the factorization succeeded then compute the approximate marginal likelihood. Otherwise, return Inf.
    # FIXME. For now we have to convert to matrix to have the check=false version available. Eventually,
    # this should also work with a StaticMatrix
    FIW = cholesky(Symmetric(Matrix(I + W)), check=false)
    if issuccess(FIW)
      return nl + (vrandeffsorth'vrandeffsorth + logdet(FIW))/2
    end
  end
  # conditional likelihood return Inf
  return typeof(nl)(Inf)
end

function marginal_nll(m::PumasModel,
                      subject::Subject,
                      param::NamedTuple,
                      vrandeffsorth::AbstractVector,
                      approx::LLQuad,
                      # Since the random effect is scaled to be standard normal we can just hardcode the integration domain
                      low::AbstractVector=fill(-4.0, length(vrandeffsorth)),
                      high::AbstractVector=fill(4.0, length(vrandeffsorth)),
                      args...; batch = 0,
                      ireltol = 1e-12, iabstol=1e-12, imaxiters = 100_000,
                      kwargs...)

  randeffstransform = totransform(m.random(param))

  integrand = (_vrandeffsorth,nothing) -> exp(
    -conditional_nll(
      m,
      subject,
      param,
      TransformVariables.transform(
        randeffstransform,
        _vrandeffsorth
      ),
      args...;
      kwargs...
    ) - _vrandeffsorth'_vrandeffsorth/2 - log(2π)*length(vrandeffsorth)/2
  )

  intprob = QuadratureProblem(integrand,low,
                                 high,
                                 batch=batch)

  sol = solve(intprob,approx.quadalg,reltol=ireltol,
              abstol=iabstol,maxiters = imaxiters)
  -log(sol.u)
end

# deviance is NONMEM-equivalent marginal negative loglikelihood
"""
    deviance(model, subject, param[, param], approx, ...)
    deviance(model, data, param, approx, ...)

Compute the deviance of a subject or dataset:
this is scaled and shifted slightly from [`marginal_nll`](@ref).
"""
StatsBase.deviance(m::PumasModel,
                   subject::Subject,
                   args...; kwargs...) =
    2marginal_nll(m, subject, args...; kwargs...) - count(!ismissing, first(subject.observations))*log(2π)

StatsBase.deviance(m::PumasModel,
                   data::Population,
                   args...; kwargs...) =
    2marginal_nll(m, data, args...; kwargs...) - sum(subject->count(!ismissing, first(subject.observations)), data)*log(2π)
# NONMEM doesn't allow ragged, so this suffices for testing

# Compute the gradient of marginal_nll without solving inner optimization
# problem. This functions follows the approach of Almquist et al. (2015) by
# computing the gradient
#
# dℓᵐ/dθ = ∂ℓᵐ/∂θ + dη/dθ'*∂ℓᵐ/∂η
#
# where ℓᵐ is the marginal likelihood of the subject and dη/dθ is the Jacobian
# of the optimal value of η with respect to the population parameters θ. By
# exploiting that η is computed as the optimum in θ, the Jacobian can be
# computed as
#
# dη/dθ = -∂²ℓᵖ∂η² \ ∂²ℓᵖ∂η∂θ
#
# ℓᵖ is the penalized conditional likelihood function.
function marginal_nll_gradient!(g::AbstractVector,
                                model::PumasModel,
                                subject::Subject,
                                vparam::AbstractVector,
                                vrandeffsorth::AbstractVector,
                                approx::Union{FOCE,FOCEI,Laplace,LaplaceI},
                                trf::TransformVariables.TransformTuple,
                                args...;
                                # We explicitly use reltol to compute the right step size for finite difference based gradient
                                reltol=DEFAULT_ESTIMATION_RELTOL,
                                fdtype=Val{:central}(),
                                fdrelstep=_fdrelstep(model, vparam, reltol, fdtype),
                                fdabsstep=fdrelstep^2,
                                kwargs...
                                )

  param = TransformVariables.transform(trf, vparam)

  # Compute first order derivatives of the marginal likelihood function
  # with finite differencing to save compute time
  ∂ℓᵐ∂θ = DiffEqDiffTools.finite_difference_gradient(
    _vparam -> marginal_nll(
      model,
      subject,
      TransformVariables.transform(trf, _vparam),
      vrandeffsorth,
      approx,
      args...;
      reltol=reltol,
      kwargs...
    ),
    vparam,
    relstep=fdrelstep,
    absstep=fdabsstep
  )

  ∂ℓᵐ∂η = DiffEqDiffTools.finite_difference_gradient(
    vηorth -> marginal_nll(
      model,
      subject,
      param,
      vηorth,
      approx,
      args...;
      reltol=reltol,
      kwargs...
    ),
    vrandeffsorth,
    relstep=fdrelstep,
    absstep=fdabsstep
  )

  # Compute second order derivatives in high precision with ForwardDiff
  ∂²ℓᵖ∂η² = ForwardDiff.hessian(
    vηorth -> penalized_conditional_nll(
      model,
      subject,
      param,
      vηorth,
      approx,
      args...;
      reltol=reltol,
      kwargs...),
    vrandeffsorth
  )

  ∂²ℓᵖ∂η∂θ = ForwardDiff.jacobian(
    _vparam -> begin
      _param = TransformVariables.transform(trf, _vparam)
      ForwardDiff.gradient(
        vηorth -> penalized_conditional_nll(
          model,
          subject,
          _param,
          vηorth,
          approx,
          args...;
          reltol=reltol,
          kwargs...),
        vrandeffsorth
      )
    end,
    vparam
  )

  dηdθ = -∂²ℓᵖ∂η² \ ∂²ℓᵖ∂η∂θ

  g .= ∂ℓᵐ∂θ .+ dηdθ'*∂ℓᵐ∂η

  return g
end

function _fdrelstep(model::PumasModel, param, reltol, ::Val{:forward})
  if model.prob isa ExplicitModel
    return sqrt(eps(numtype(param)))
  else
    return max(norm(reltol), sqrt(eps(numtype(param))))
  end
end
function _fdrelstep(model::PumasModel, param, reltol, ::Val{:central})
  if model.prob isa ExplicitModel
    return cbrt(eps(numtype(param)))
  else
    return max(norm(reltol), cbrt(eps(numtype(param))))
  end
end

# Similar to the version for FOCE, FOCEI, Laplace, and LaplaceI
# but much simpler since the expansion point in η is fixed. Hence,
# the gradient is simply the partial derivative in θ
function marginal_nll_gradient!(g::AbstractVector,
                                model::PumasModel,
                                subject::Subject,
                                vparam::AbstractVector,
                                vrandeffsorth::AbstractVector,
                                approx::Union{NaivePooled,FO,FOI,LLQuad},
                                trf::TransformVariables.TransformTuple,
                                args...;
                                # We explicitly use reltol to compute the right step size for finite difference based gradient
                                reltol=DEFAULT_ESTIMATION_RELTOL,
                                fdtype=Val{:central}(),
                                fdrelstep=_fdrelstep(model, vparam, reltol, fdtype),
                                fdabsstep=fdrelstep^2,
                                kwargs...
                                )

  # Compute first order derivatives of the marginal likelihood function
  # with finite differencing to save compute time
  g .= DiffEqDiffTools.finite_difference_gradient(
    _vparam -> marginal_nll(
      model,
      subject,
      TransformVariables.transform(trf, _vparam),
      vrandeffsorth,
      approx,
      args...;
      reltol=reltol,
      kwargs...
    ),
    vparam,
    typeof(fdtype);
    relstep=fdrelstep,
    absstep=fdabsstep
  )

  return g
end

function marginal_nll_gradient!(g::AbstractVector,
                                model::PumasModel,
                                population::Population,
                                vparam::AbstractVector,
                                vvrandeffsorth::AbstractVector,
                                approx::LikelihoodApproximation,
                                trf::TransformVariables.TransformTuple,
                                args...; kwargs...
                                )

  # Zero the gradient
  fill!(g, 0)

  # FIXME! Avoid this allocation
  _g = similar(g)

  for (subject, vrandeffsorth) in zip(population, vvrandeffsorth)
    marginal_nll_gradient!(
      _g,
      model,
      subject,
      vparam,
      vrandeffsorth,
      approx,
      trf, args...; kwargs...)
    g .+= _g
  end

  return g
end

function _derived_vηorth_gradient(
  m::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  args...; kwargs...)
  # Costruct closure for calling conditional_nll_ext as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  _transform_derived =  vηorth -> begin
    randeffs = TransformVariables.transform(totransform(m.random(param)), vηorth)
    return _derived(m, subject, param, randeffs, args...; kwargs...)
  end
  # Construct vector of dual numbers for the random effects to track the partial derivatives
  cfg = ForwardDiff.JacobianConfig(_transform_derived, vrandeffsorth, ForwardDiff.Chunk{length(vrandeffsorth)}())

  ForwardDiff.seed!(cfg.duals, vrandeffsorth, cfg.seeds)

  return _transform_derived(cfg.duals)
end


function _mean_derived_vηorth_jacobian(m::PumasModel,
                                       subject::Subject,
                                       param::NamedTuple,
                                       vrandeffsorth::AbstractVector,
                                       args...; kwargs...)
  dual_derived = _derived_vηorth_gradient(m ,subject, param, vrandeffsorth, args...; kwargs...)
  # Loop through the distribution vector and extract derivative information
  nt = length(first(dual_derived))
  nrandeffs = length(vrandeffsorth)
  F = map(NamedTuple{keys(subject.observations)}(dual_derived)) do dv
        Ft = zeros(eltype(ForwardDiff.partials(first(dv).μ).values), nrandeffs, nt)
        for j in eachindex(dv)
          partial_values = ForwardDiff.partials(dv[j].μ).values
          for i = 1:nrandeffs
            Ft[i, j] = partial_values[i]
          end
        end
        return Ft'
      end
  return F
end

function ∂²l∂η²(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffsorth::AbstractVector,
                approx::Union{FO,FOI,FOCE,FOCEI},
                args...; kwargs...)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable
  # per observation while tracking partial derivatives of the random effects
  _derived_dist = _derived_vηorth_gradient(m, subject, param, vrandeffsorth, args...; kwargs...)

  if any(d->d isa Nothing, _derived_dist)
    return map(x->(Inf, nothing, nothing), subject.observations)
  end

  dv_keys = keys(subject.observations)

  dv_dist = NamedTuple{dv_keys}(_derived_dist)
  dv_zip = NamedTuple{dv_keys}(zip(dv_keys, subject.observations, dv_dist))

  if approx isa FOCE
    return map(d -> _∂²l∂η²(d[1], d[2], d[3], m, subject, param, vrandeffsorth, approx, args...; kwargs...), dv_zip)
  else
    return map(d -> _∂²l∂η²(d[2], d[3], approx), dv_zip)
  end
end

function _∂²l∂η²(obsdv::AbstractVector, dv::AbstractVector{<:Normal}, ::FO)
  # The dimension of the random effect vector
  nrfx = length(ForwardDiff.partials(first(dv).μ))

  # Initialize Hessian matrix and gradient vector
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  dldη = @SVector zeros(nrfx)
  nl   = 0.0

  # Loop through the distribution vector and extract derivative information
  for j in eachindex(dv)
    obsdvj = obsdv[j]

    # We ignore missing observations when estimating the model
    if ismissing(obsdvj)
      continue
    end

    dvj = dv[j]
    r = ForwardDiff.value(dvj.σ)^2
    f = SVector(ForwardDiff.partials(dvj.μ).values)
    fdr = f/r

    H    += fdr*f'
    dldη += fdr*(obsdvj - ForwardDiff.value(dvj.μ))
    nl   -= ForwardDiff.value(_lpdf(dvj, obsdvj))
  end

  return nl, dldη, H
end

function _∂²l∂η²(obsdv::AbstractVector, dv::AbstractVector{<:LogNormal}, ::FO)
  # The dimension of the random effect vector
  nrfx = length(ForwardDiff.partials(first(dv).μ))

  # Initialize Hessian matrix and gradient vector
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  dldη = @SVector zeros(nrfx)
  nl   = 0.0

  # Loop through the distribution vector and extract derivative information
  for j in eachindex(dv)
    obsdvj = obsdv[j]

    # We ignore missing observations when estimating the model
    if ismissing(obsdvj)
      continue
    end

    dvj = dv[j]
    r = ForwardDiff.value(dvj.σ)^2
    f = SVector(ForwardDiff.partials(dvj.μ).values)
    fdr = f/r

    H    += fdr*f'
    dldη += fdr*(log(obsdvj) - ForwardDiff.value(dvj.μ))
    nl   -= ForwardDiff.value(_lpdf(dvj, obsdvj))
  end

  return nl, dldη, H
end

# Helper function to detect homoscedasticity. For now, it is assumed the input dv vecotr containing
# a vector of distributions with ForwardDiff element types.
function _is_homoscedastic(dv::AbstractVector{<:Union{Normal,LogNormal}})
  # FIXME! Eventually we should support more dependent variables instead of hard coding for dv
  v1 = ForwardDiff.value(first(dv).σ)
  return all(t -> ForwardDiff.value(t.σ) == v1, dv)
end
_is_homoscedastic(::Any) = throw(ArgumentError("Distribution not supported"))

function _∂²l∂η²(dv_name::Symbol,
                 obsdv,
                 dv_d::AbstractVector{<:Union{Normal,LogNormal}},
                 m::PumasModel,
                 subject::Subject,
                 param::NamedTuple,
                 vrandeffsorth::AbstractVector,
                 ::FOCE,
                 args...; kwargs...)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation for η=0
  # If the model is homoscedastic, it is not necessary to recompute the variances at η=0
  if _is_homoscedastic(dv_d)
    return _∂²l∂η²(obsdv, dv_d, first(dv_d), FOCE())
  else # in the Heteroscedastic case, compute the variances at η=0
    randeffstransform = totransform(m.random(param))
    # should just pass this in from the outside to only calculate once
    dist_0 = _derived(
      m,
      subject,
      param,
      TransformVariables.transform(randeffstransform, zero(vrandeffsorth)),
      args...;
      kwargs...
      )

    # Compute the Hessian approxmation in the random effect vector η
    return _∂²l∂η²(obsdv, dv_d, dist_0[dv_name], FOCE())
  end
end

_ofdisttype(dμ, dσ) = _ofdisttype(dμ; σ=dσ.σ)
_ofdisttype(d::Normal; μ=d.μ, σ=d.σ)    = Normal(μ, σ)
_ofdisttype(d::LogNormal; μ=d.μ, σ=d.σ) = LogNormal(μ, σ)

# Homoscedastic case
function _∂²l∂η²(obsdv::AbstractVector, dv_d::AbstractVector{<:Union{Normal,LogNormal}}, dv0::Union{Normal,LogNormal}, ::FOCE)
  # Loop through the distribution vector and extract derivative information
  nrfx = length(ForwardDiff.partials(first(dv_d).μ))

  # Initialize Hessian matrix and gradient vector
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  nl   = 0.0
  σ = ForwardDiff.value(dv0.σ)

  for j in eachindex(dv_d)
    obj = obsdv[j]
    if ismissing(obj)
      continue
    end
    dvj = dv_d[j]
    f = SVector(ForwardDiff.partials(dvj.μ).values)/σ

    H  += f*f'
    nl -= _lpdf(_ofdisttype(dvj; μ=ForwardDiff.value(dvj.μ), σ=σ), obj)
  end

  return nl, nothing, H
end

# Heteroscedastic case
function _∂²l∂η²(obsdv::AbstractVector, dv_d::AbstractVector{<:Normal}, dv0::AbstractVector{<:Normal}, ::FOCE)
  # Loop through the distribution vector and extract derivative information
  nrfx = length(ForwardDiff.partials(first(dv_d).μ))

  # Initialize Hessian matrix and gradient vector
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  nl   = 0.0

  for j in eachindex(dv_d)
    obj = obsdv[j]
    if ismissing(obj)
      continue
    end
    dvj = dv_d[j]
    σ   = dv0[j].σ
    f   = SVector(ForwardDiff.partials(dvj.μ).values)/σ
    H  += f*f'
    nl -= _lpdf(_ofdisttype(dvj; μ=ForwardDiff.value(dvj.μ), σ=σ), obj)
  end

  return nl, nothing, H
end

function _∂²l∂η²(obsdv::AbstractVector, dv::AbstractVector{<:Union{Normal,LogNormal}}, ::FOCEI)
  # Loop through the distribution vector and extract derivative information
  nrfx = length(ForwardDiff.partials(first(dv).μ))

  # Initialize Hessian matrix and gradient vector
  ## FIXME! Careful about hardcoding for Float64 here
  H    = @SMatrix zeros(nrfx, nrfx)
  nl   = 0.0

  for j in eachindex(dv)
    obj = obsdv[j]
    if ismissing(obj)
      continue
    end
    dvj   = dv[j]
    r_inv = inv(ForwardDiff.value(dvj.σ^2))
    f     = SVector(ForwardDiff.partials(dvj.μ).values)
    del_r = SVector(ForwardDiff.partials(dvj.σ.^2).values)

    H  += f*r_inv*f' + (r_inv*del_r*r_inv*del_r')/2
    nl -= ForwardDiff.value(_lpdf(dvj, obj))
  end

  return nl, nothing, H
end

# FIXME Laplace(I) only support one dv.
function ∂²l∂η²(m::PumasModel,
                subject::Subject,
                param::NamedTuple,
                vrandeffsorth::AbstractVector,
                approx::Union{Laplace,LaplaceI},
                args...; kwargs...)

  if length(subject.observations) > 1
    throw("Laplace and LaplaceI currently do not support multiple DVs, use FOCEI instead.")
  end
  # Initialize HessianResult for computing Hessian, gradient and value of negative loglikelihood in one go
  diffres = DiffResults.HessianResult(vrandeffsorth)

  # Compute the derivates
  ForwardDiff.hessian!(diffres,
    vηorth -> conditional_nll(
      m,
      subject,
      param,
      TransformVariables.transform(totransform(m.random(param)), vηorth),
      approx,
      args...;
      kwargs...),
  vrandeffsorth)

#   # Extract the derivatives
  nl, W = DiffResults.value(diffres), DiffResults.hessian(diffres)

  return map(x -> (nl, nothing, W), subject.observations)
end

# Fallbacks for a usful error message when distribution isn't supported
_∂²l∂η²(dv_name::Symbol,
        dv_d::Any,
        obsdv,
        m::PumasModel,
        subject::Subject,
        param::NamedTuple,
        vrandeffsorth::AbstractVector,
        approx::LikelihoodApproximation,
        args...; kwargs...) = throw(ArgumentError("Distribution is current not supported for the $approx approximation. Please consider a different likelihood approximation."))
_∂²l∂η²(dv_d::Any,
        obsdv,
        approx::LikelihoodApproximation) = throw(ArgumentError("Distribution is current not supported for the $approx approximation. Please consider a different likelihood approximation."))

# Fitting methods
struct FittedPumasModel{T1<:PumasModel,T2<:Population,T3,T4<:LikelihoodApproximation, T5, T6, T7, T8}
  model::T1
  data::T2
  optim::T3
  approx::T4
  vvrandeffsorth::T5
  args::T6
  kwargs::T7
  fixedtrf::T8
end

function DEFAULT_OPTIMIZE_FN(cost, p, callback)
  Optim.optimize(
    cost,
    p,
    BFGS(
      linesearch=Optim.LineSearches.BackTracking(),
      # Make sure that step isn't too large by scaling initial Hessian by the norm of the initial gradient
      initial_invH=t -> Matrix(I/norm(Optim.NLSolversBase.gradient(cost)), length(p), length(p))
    ),
    Optim.Options(
      show_trace=false, # Print progress
      store_trace=true,
      extended_trace=true,
      g_tol=1e-3,
      allow_f_increases=true,
      callback=callback
    )
  )
end

"""
    _fixed_to_constanttransform(trf::TransformTuple, param::NamedTuple, fixed::NamedTuple)

Replace individual parameter transformations in `trf` with `ConstantTranform` if
the parameter has an entry in `fixed`. Return a new parameter `NamedTuple` with
the values in `fixed` in place of the values in input `param`.
"""
function _fixed_to_constanttransform(trf, param, fixed)
  fix_keys = keys(fixed)
  _keys = keys(trf.transformations)
  _vals = []
  _paramval = []
  for key in _keys
    if key ∈ fix_keys
      push!(_vals, ConstantTransform(fixed[key]))
      push!(_paramval, fixed[key])
    else
      push!(_vals, trf.transformations[key])
      push!(_paramval, param[key])
    end
  end
  new_param = NamedTuple{_keys}(_paramval)
  return new_param, TransformVariables.TransformTuple(NamedTuple{_keys}(_vals))
end
function Distributions.fit(m::PumasModel,
                           population::Population,
                           param::NamedTuple,
                           approx::LikelihoodApproximation,
                           args...;
                           # optimize_fn should take the arguments cost, p, and callback where cost is a
                           # NLSolversBase.OnceDifferentiable, p is a Vector, and cl is Function. Hence,
                           # optimize_fn should evaluate cost with the NLSolversBase.value and
                           # NLSolversBase.gradient interface. The cl callback should be called once per
                           # outer iteration when applicable but it is not required that the optimization
                           # procedure calls cl. In that case, the estimation of the EBEs will always begin
                           # in zero. In addition, the returned object should support a opt_minimizer method
                           # that returns the optimized parameters.
                           optimize_fn = DEFAULT_OPTIMIZE_FN,
                           constantcoef = NamedTuple(),
                           kwargs...)

  # Compute transform object defining the transformations from NamedTuple to Vector while applying any parameter restrictions and apply the transformations
  trf = totransform(m.param)
  fixedtrf=trf
  param, fixedtrf = _fixed_to_constanttransform(trf, param, constantcoef)
  vparam = TransformVariables.inverse(fixedtrf, param)

  # We'll store the orthogonalized random effects estimate in vvrandeffsorth which allows us to carry the estimates from last
  # iteration and use them as staring values in the next iteration. We also allocate a buffer to store the
  # random effect estimate during an iteration since it might be modified several times during a line search
  # before the new value has been found. We then define a callback which will store values of vvrandeffsorth_tmp
  # in vvrandeffsorth once the iteration is done.
  if approx isa NaivePooled
    vvrandeffsorth     = [[] for subject in population]
    vvrandeffsorth_tmp = [copy(vrandefforths) for vrandefforths in vvrandeffsorth]
    cb(state) = false
  else
    vvrandeffsorth     = [zero(_vecmean(m.random(param))) for subject in population]
    vvrandeffsorth_tmp = [copy(vrandefforths) for vrandefforths in vvrandeffsorth]
    cb = state -> begin
      for i in eachindex(vvrandeffsorth)
        copyto!(vvrandeffsorth[i], vvrandeffsorth_tmp[i])
      end
      return false
    end
  end
  # Define cost function for the optimization
  cost = Optim.NLSolversBase.OnceDifferentiable(
    Optim.NLSolversBase.only_fg!() do f, g, _vparam
      # The negative loglikelihood function
      # Update the Empirical Bayes Estimates explicitly after each iteration

      # Convert vector to NamedTuple
      _param = TransformVariables.transform(fixedtrf, _vparam)

      # Sum up loglikelihood contributions
      nll = sum(zip(population, vvrandeffsorth, vvrandeffsorth_tmp)) do (subject, vrandefforths, vrandefforths_tmp)
        # If not FO then compute EBE based on the estimates from last iteration stored in vvrandeffsorth
        # and store the retult in vvrandeffsorth_tmp
        if !(approx isa FO || approx isa NaivePooled)
          copyto!(vrandefforths_tmp, vrandefforths)
          _orth_empirical_bayes!(vrandefforths_tmp, m, subject, _param, approx, args...; kwargs...)
        end
        marginal_nll(m, subject, _param, vrandefforths_tmp, approx, args...; kwargs...)
      end

      # Update score
      if g !== nothing
        marginal_nll_gradient!(g, m, population, _vparam, vvrandeffsorth_tmp, approx, fixedtrf, args...; kwargs...)
      end

      return nll
    end,

    # The initial values
    vparam
  )

  # Run the optimization
  o = optimize_fn(cost, vparam, cb)

  # Update the random effects after optimization
  if !(approx isa FO || approx isa NaivePooled)
    for (vrandefforths, subject) in zip(vvrandeffsorth, population)
      _orth_empirical_bayes!(vrandefforths, m, subject, TransformVariables.transform(fixedtrf, opt_minimizer(o)), approx, args...; kwargs...)
    end
  end

  return FittedPumasModel(m, population, o, approx, vvrandeffsorth, args, kwargs, fixedtrf)
end

function Distributions.fit(m::PumasModel,
                           subject::Subject,
                           param::NamedTuple,
                           args...;
                           kwargs...)
  return fit(m, [subject,], param, NaivePooled(), args...; kwargs...)
end
function Distributions.fit(m::PumasModel,
                           population::Population,
                           param::NamedTuple,
                           ::TwoStage,
                           args...;
                           kwargs...)
  return map(x->fit(m, [x,], param, NaivePooled(), args...; kwargs...), population)
end

# error handling for fit(model, subject, param, args...; kwargs...)
function Distributions.fit(model::PumasModel, subject::Subject,
             param::NamedTuple, approx::LikelihoodApproximation, args...; kwargs...)
  throw(ArgumentError("Calling fit on a single subject is not allowed with a likelihood approximation method specified."))
end

opt_minimizer(o::Optim.OptimizationResults) = Optim.minimizer(o)

function StatsBase.coef(fpm::FittedPumasModel)
  # we need to use the transform that takes into account that the fixed param
  # are transformed according to the ConstantTransformations, and not the
  # transformations given in totransform(model.param)
  trf = fpm.fixedtrf
  TransformVariables.transform(trf, opt_minimizer(fpm.optim))
end
function Base.getproperty(f::FittedPumasModel{<:Any,<:Any,<:Optim.MultivariateOptimizationResults}, s::Symbol)
  if s === :param
    # deprecate?
    coef(f)
  else
    return getfield(f, s)
  end
end

marginal_nll(      f::FittedPumasModel) = marginal_nll(f.model, f.data, coef(f), f.approx)
StatsBase.deviance(f::FittedPumasModel) = deviance(    f.model, f.data, coef(f), f.approx)

function _observed_information(f::FittedPumasModel,
                                ::Val{Score},
                               args...;
                               # We explicitly use reltol to compute the right step size for finite difference based gradient
                               # The tolerance has to be stricter when computing the covariance than during estimation
                               reltol=abs2(DEFAULT_ESTIMATION_RELTOL),
                               kwargs...) where Score
  # Transformation the NamedTuple of parameters to a Vector
  # without applying any bounds (identity transform)
  trf = toidentitytransform(f.model.param)
  param = coef(f)
  vparam = TransformVariables.inverse(trf, param)

  fdrelstep_score = _fdrelstep(f.model, vparam, reltol, Val{:central}())
  fdrelstep_hessian = sqrt(_fdrelstep(f.model, vparam, reltol, Val{:central}()))

  # Initialize arrays
  H = zeros(eltype(vparam), length(vparam), length(vparam))
  _H = zeros(eltype(vparam), length(vparam), length(vparam))
  if Score
    S = copy(H)
    g = similar(vparam, length(vparam))
  else
    S = g = nothing
  end

  # Loop through subject and compute Hessian and score contributions
  for i in eachindex(f.data)
    subject = f.data[i]

    _f = function (_j, _vparam)
      _param = TransformVariables.transform(trf, _vparam)
      vrandeffsorth = _orth_empirical_bayes(f.model, subject, _param, f.approx, args...; kwargs...)
      marginal_nll_gradient!(
      _j,
      f.model,
      subject,
      _vparam,
      vrandeffsorth,
      f.approx,
      trf,
      args...;
      reltol=reltol,
      fdtype=Val{:central}(),
      fdrelstep=fdrelstep_hessian,
      fdabsstep=fdrelstep_hessian^2,
      kwargs...)
      return nothing
    end

    # Compute Hessian contribution and update Hessian
    DiffEqDiffTools.finite_difference_jacobian!(_H,_f,vparam,
                                                     Val{:central};
                                                     relstep=fdrelstep_hessian,
                                                     absstep=fdrelstep_hessian^2)

    H .+= _H

    if Score
      # Compute score contribution
      vrandeffsorth = _orth_empirical_bayes(f.model, subject, coef(f), f.approx, args...; kwargs...)
      marginal_nll_gradient!(
        g,
        f.model,
        subject,
        vparam,
        vrandeffsorth,
        f.approx,
        trf,
        args...;
        reltol=reltol,
        fdtype=Val{:central}(),
        fdrelstep=fdrelstep_score,
        fdabsstep=fdrelstep_hessian^2,
        kwargs...)

      # Update outer product of scores
      S .+= g .* g'
    end
  end
  return H, S
end

function _expected_information(m::PumasModel,
                               subject::Subject,
                               param::NamedTuple,
                               vrandeffsorth::AbstractVector,
                               ::FO,
                               args...; kwargs...)

  trf = toidentitytransform(m.param)
  vparam = TransformVariables.inverse(trf, param)

  # Costruct closure for calling _derived as a function
  # of a random effects vector. This makes it possible for ForwardDiff's
  # tagging system to work properly
  __E_and_V = _param -> _E_and_V(m, subject, TransformVariables.transform(trf, _param), vrandeffsorth, FO(), args...; kwargs...)

  # Construct vector of dual numbers for the population parameters to track the partial derivatives
  cfg = ForwardDiff.JacobianConfig(__E_and_V, vparam, ForwardDiff.Chunk{length(vparam)}())
  ForwardDiff.seed!(cfg.duals, vparam, cfg.seeds)

  # Compute the conditional likelihood and the conditional distributions of the dependent variable per observation while tracking partial derivatives of the random effects
  E_d, V_d = __E_and_V(cfg.duals)

  V⁻¹ = inv(cholesky(ForwardDiff.value.(V_d)))
  dEdθ = hcat((collect(ForwardDiff.partials(E_k).values) for E_k in E_d)...)

  m = size(dEdθ, 1)
  n = size(dEdθ, 2)
  dVpart = similar(dEdθ, m, m)
  for l in 1:m
    dVdθl = [ForwardDiff.partials(V_d[i,j]).values[l] for i in 1:n, j in 1:n]
    for k in 1:m
      dVdθk = [ForwardDiff.partials(V_d[i,j]).values[k] for i in 1:n, j in 1:n]
      # dVpart[l,k] = tr(dVdθk * V⁻¹ * dVdθl * V⁻¹)/2
      dVpart[l,k] = sum((V⁻¹ * dVdθk) .* (dVdθl * V⁻¹))/2
    end
  end

  return dEdθ*V⁻¹*dEdθ' + dVpart
end

function _expected_information_fd(
  model::PumasModel,
  subject::Subject,
  param::NamedTuple,
  vrandeffsorth::AbstractVector,
  ::FO,
  args...;
  blockdiag=true,
  reltol=DEFAULT_ESTIMATION_RELTOL,
  fdtype=Val{:central}(),
  fdrelstep=_fdrelstep(model, param, reltol, fdtype),
  fdabsstep=fdrelstep^2,
  kwargs...)

  trf = toidentitytransform(model.param)
  vparam = TransformVariables.inverse(trf, param)

  # Costruct closure for calling _derived as a function
  # of a random effects vector.
  __E_and_V = _param -> _E_and_V(
    model,
    subject,
    TransformVariables.transform(trf, _param),
    vrandeffsorth,
    FO(),
    args...; kwargs...)

  E, V = __E_and_V(vparam)

  ___E = _vparam -> __E_and_V(_vparam)[1]
  ___V = _vparam -> vec(__E_and_V(_vparam)[2])

  dEdθ = DiffEqDiffTools.finite_difference_jacobian(
    ___E,
    vparam,
    typeof(fdtype),
    eltype(vparam),
    relstep=fdrelstep,
    absstep=fdabsstep
  )
  JV = DiffEqDiffTools.finite_difference_jacobian(
    ___V,
    vparam,
    typeof(fdtype),
    eltype(vparam),
    relstep=fdrelstep,
    absstep=fdabsstep
  )

  V⁻¹ = inv(V)
  m = size(dEdθ, 1)
  n = size(dEdθ, 2)

  dVpart = similar(dEdθ, n, n)
  for l in 1:n
    dVdθl = reshape(JV[:, l], m, m)
    for k in 1:n
      dVdθk = reshape(JV[:, k], m, m)
      dVpart[l,k] = sum((V⁻¹ * dVdθk) .* (dVdθl * V⁻¹))/2
    end
  end

  Apart = dEdθ'*V⁻¹*dEdθ
  return Apart + dVpart
end

function StatsBase.informationmatrix(f::FittedPumasModel; expected::Bool=true)
  data          = f.data
  model         = f.model
  param         = coef(f)
  vrandeffsorth = f.vvrandeffsorth
  if expected
    return sum(_expected_information(model, data[i], param, vrandeffsorth[i], f.approx) for i in 1:length(data))
  else
    return first(_observed_information(f, Val(false)))
  end
end

"""
    vcov(f::FittedPumasModel) -> Matrix

Compute the covariance matrix of the population parameters
"""
function StatsBase.vcov(f::FittedPumasModel, args...; kwargs...)

  # Compute the observed information based on the Hessian (H) and the product of the outer scores (S)
  H, S = _observed_information(f, Val(true), args...; kwargs...)

  # Use generialized eigenvalue decomposition to compute inv(H)*S*inv(H)
  F = eigen(Symmetric(H), Symmetric(S))
  any(t -> t <= 0, F.values) && @warn("Hessian is not positive definite")
  return F.vectors*Diagonal(inv.(abs2.(F.values)))*F.vectors'
end

"""
    stderror(f::FittedPumasModel) -> NamedTuple

Compute the standard errors of the population parameters and return
the result as a `NamedTuple` matching the `NamedTuple` of population
parameters.
"""
StatsBase.stderror(f::FittedPumasModel) = stderror(infer(f))

function Statistics.mean(vfpm::Vector{<:FittedPumasModel})
  names = keys(coef(first(vfpm)))
  means = []
  for name in names
    push!(means, mean([coef(fpm)[name] for fpm in vfpm]))
  end
  NamedTuple{names}(means)
end
function Statistics.std(vfpm::Vector{<:FittedPumasModel})
  names = keys(coef(first(vfpm)))
  stds = []
  for name in names
    push!(stds, std([coef(fpm)[name] for fpm in vfpm]))
  end
  NamedTuple{names}(stds)
end
function Statistics.var(vfpm::Vector{<:FittedPumasModel})
  names = keys(coef(first(vfpm)))
  vars = []
  for name in names
    push!(vars, var([coef(fpm)[name] for fpm in vfpm]))
  end
  NamedTuple{names}(vars)
end

# This is called with dual numbered "param"
function _E_and_V(model::PumasModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffsorth::AbstractVector,
                  ::FO,
                  args...; kwargs...)

  randeffstransform = totransform(model.random(param))
  dist = _derived(
    model,
    subject,
    param,
    TransformVariables.transform(randeffstransform, vrandeffsorth),
    args...; kwargs...)

  _names = keys(subject.observations)

  E = vcat([mean.(dist[_name]) for _name in _names]...)

  F = _mean_derived_vηorth_jacobian(
    model,
    subject,
    param,
    vrandeffsorth,
    args...; kwargs...)

  FF = vcat([F[_name] for _name in _names]...)
  dd = vcat([var.(dist[_name]) for _name in _names]...)
  V = FF*FF' + Diagonal(dd)

  return E, V
end

function _E_and_V(m::PumasModel,
                  subject::Subject,
                  param::NamedTuple,
                  vrandeffsorth::AbstractVector,
                  ::FOCE,
                  args...; kwargs...)

  randeffstransform = totransform(m.random(param))
  y = subject.observations.dv
  dist0 = _derived(m, subject, param, TransformVariables.transform(randeffstransform, zero(vrandeffsorth)))
  dist  = _derived(m, subject, param, TransformVariables.transform(randeffstransform, vrandeffsorth))

  F = ForwardDiff.jacobian(
    _vrandeffs -> begin
      _randeffs = TransformVariable.transform(randeffstransform, _vrandeffs)
      return mean.(_derived(m, subject, param, _randeffs).dv)
    end,
    vrandeffsorth
  )
  V = Symmetric(F*F' + Diagonal(var.(dist0.dv)))

  return mean.(dist.dv) .- F*vrandeffsorth, V
end

# Some type piracy for the time being
Distributions.MvNormal(D::Diagonal) = MvNormal(PDiagMat(D.diag))

struct FittedPumasModelInference{T1, T2, T3}
  fpm::T1
  vcov::T2
  level::T3
end

"""
    infer(fpm::FittedPumasModel) -> FittedPumasModelInference

Compute the `vcov` matrix and return a struct used for inference
based on the fitted model `fpm`.
"""
function infer(fpm::FittedPumasModel, args...; level = 0.95, kwargs...)
  print("Calculating: variance-covariance matrix")
  _vcov = vcov(fpm, args...; kwargs...)
  println(". Done.")
  FittedPumasModelInference(fpm, _vcov, level)
end
function StatsBase.stderror(pmi::FittedPumasModelInference)
  ss = sqrt.(diag(pmi.vcov))
  trf = tostderrortransform(pmi.fpm.model.param)
  return TransformVariables.transform(trf, ss)
end

# empirical_bayes_dist for FittedPumasModel
function empirical_bayes_dist(fpm::FittedPumasModel)
  map(zip(fpm.data, fpm.vvrandeffsorth)) do (subject, vrandeffsorth)
      empirical_bayes_dist(fpm.model, subject, coef(fpm), vrandeffsorth, fpm.approx)
  end
end
