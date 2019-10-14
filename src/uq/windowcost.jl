abstract type UncertaintyQuantificationAlgorithm end
struct KoopmanQuant{T} <: UncertaintyQuantificationAlgorithm
    quadalg::T
end
struct MonteCarloQuant <: UncertaintyQuantificationAlgorithm end

function uq_windowcost(g, quant::KoopmanQuant,
                       m::PumasModel, subject::Subject,
                       param_dists,
                       args...;
                       ireltol=1e-3,iabstol=1e-3,imaxiters=10000,
                       batch = 0,
                       kwargs...)

    _f = function (x,nothing)
      k = g(simobs(m,subject,(θ=x,),args...;kwargs...))
      w = prod(pdf(a,b) for (a,b) in zip(param_dists.θ,x))
      k*w
    end

    intprob = QuadratureProblem(_f,m.param.params.θ.lower,
                                   m.param.params.θ.upper,
                                   batch=batch)

    sol = solve(intprob,quant.quadalg,reltol=ireltol,
                abstol=iabstol,maxiters = imaxiters)
end

function uq_windowcost(g, ::MonteCarloQuant,
                       m::PumasModel, subject::Subject,
                       param_dists,
                       args...;imaxiters=10000,
                       kwargs...)
   pop = [deepcopy(subject) for i in 1:imaxiters]
   results = g.(simobs(m,pop,param_dists,args...;kwargs...))
   mean(results),std(results)
end
