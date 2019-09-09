function DiffEqSensitivity.gsa(m::PumasModel,subject::Subject,params::NamedTuple,method::DiffEqSensitivity.GSAMethod,p_range=[[0.0,1.0] for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],args...;kwargs...)
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,subject,param,args...;kwargs...)
        collect(Iterators.flatten([getproperty(sim.observed,key) for key in keys(sim.observed)])) 
    end
    DiffEqSensitivity.gsa(f,p_range,method)
end

function DiffEqSensitivity.gsa(m::PumasModel,population::Population,params::NamedTuple,method::DiffEqSensitivity.GSAMethod,p_range=[[0.0,1.0] for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],args...; kwargs...)
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,population,param,args...;kwargs...)
        collect(Iterators.flatten([collect(Iterators.flatten([getproperty(sim.sims[i].observed,key) for key in keys(sim.sims[1].observed)])) for i in 1:length(sim.sims)]))
    end
    DiffEqSensitivity.gsa(f,p_range,method)
end