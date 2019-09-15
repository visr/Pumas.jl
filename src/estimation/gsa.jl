function DiffEqSensitivity.gsa(m::PumasModel,subject::Subject,params::NamedTuple,method::DiffEqSensitivity.GSAMethod,vars = [:dv],p_range_low=NamedTuple{keys(params)}([par.*0.05 for par in values(params)]),p_range_high=NamedTuple{keys(params)}([par.*1.95 for par in values(params)]),args...; kwargs...)
    vlowparam = TransformVariables.inverse(toidentitytransform(m.param),p_range_low)
    vhighparam = TransformVariables.inverse(toidentitytransform(m.param),p_range_high)
    p_range = [[vlowparam[i],vhighparam[i]] for i in 1:length(vlowparam)]
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,subject,param,args...;kwargs...)
        collect(Iterators.flatten([getproperty(sim.observed,key) for key in vars])) 
    end
    DiffEqSensitivity.gsa(f,p_range,method)
end

function DiffEqSensitivity.gsa(m::PumasModel,population::Population,params::NamedTuple,method::DiffEqSensitivity.GSAMethod,vars = [:dv],p_range_low=NamedTuple{keys(params)}([par.*0.05 for par in values(params)]),p_range_high=NamedTuple{keys(params)}([par.*1.95 for par in values(params)]),args...; kwargs...)
    vlowparam = TransformVariables.inverse(toidentitytransform(m.param),p_range_low)
    vhighparam = TransformVariables.inverse(toidentitytransform(m.param),p_range_high)
    p_range = [[vlowparam[i],vhighparam[i]] for i in 1:length(vlowparam)]
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sims = simobs(m,population,param,args...;kwargs...)
        collect(Iterators.flatten([collect(Iterators.flatten([getproperty(sims[i].observed,key) for key in vars])) for i in 1:length(sims)]))
    end
    DiffEqSensitivity.gsa(f,p_range,method)
end
