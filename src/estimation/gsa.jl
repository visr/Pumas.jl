function DiffEqSensitivity.gsa(m::PumasModel,subject::Subject,params::NamedTuple,method::DiffEqSensitivity.GSAMethod,p_range=[[0.0,1.0] for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],args...;kwargs...)
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,subject,param,args...;kwargs...)
        sim.observed.dv
    end
    DiffEqSensitivity.gsa(f,p_range,method)
end

function DiffEqSensitivity.gsa(m::PumasModel,population::Population,params::NamedTuple,method::DiffEqSensitivity.GSAMethod,p_range=[[0.0,1.0] for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],args...; kwargs...)
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,subject,param,args...;kwargs...)
        collect(Iterators.flatten([sim.sims[i].observed.dv for i in 1:length(sim.sims)]))
    end
    DiffEqSensitivity.gsa(f,p_range,method)
end