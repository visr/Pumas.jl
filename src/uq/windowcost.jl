abstract type UncertaintyQuantificationAlgorithm end
struct KoopmanQuant <: UncertaintyQuantificationAlgorithm end
struct MonteCarloQuant <: UncertaintyQuantificationAlgorithm end

function uq_windowcost(g, ::KoopmanQuant,
                       m::PumasModel, subject::Subject,
                       param_dists,
                       randeffs=init_randeffs(m, param),
                       ireltol=1e-3,iabstol=1e-3,imaxiters=10000,
                       batch = 0,
                       args...;kwargs...)

    _transform = totransform(param_dists)

    _f = function (x,nothing)
      k = g(simobs(m,subject,(θ=x,),randeffs,args...;kwargs...))
      w = _lpdf(param_dists,x)
      k*w
    end

    intprob = QuadratureProblem(_f,minimum.(m_diffeq.param.params.θ.lower),
                                maximum.(m_diffeq.param.params.θ.upper),
                                batch=batch)
    sol = solve(intprob,quadalg,reltol=ireltol,
                abstol=iabstol,maxiters = imaxiters)
end

function uq_windowcost(g, ::MonteCarloQuant,
                       m::PumasModel, subject::Subject,
                       param_dists,
                       imaxiters=10000,
                       args...;kwargs...)

   pop = [deepcopy(subject) for i in 1:imaxiters]
   mean(g.(simobs(m,pop,param_dists,args...;kwargs...)))
end
