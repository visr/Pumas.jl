abstract type ExpectationAlgorithm end
struct KoopmanExpectation{T} <: ExpectationAlgorithm
    quadalg::T
end
struct MonteCarloExpectation <: ExpectationAlgorithm end

function expectation(g, quant::KoopmanExpectation,
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

function expectation(g, ::MonteCarloExpectation,
                       m::PumasModel, subject::Subject,
                       param_dists,
                       args...;imaxiters=10000,
                       kwargs...)
   pop = [deepcopy(subject) for i in 1:imaxiters]
   results = g.(simobs(m,pop,param_dists,args...;kwargs...))
   mean(results)
end
