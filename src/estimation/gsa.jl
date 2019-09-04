function DiffEqSensitivity.gsa(m::PumasModel,subject::Subject,params::NamedTuple,method::DiffEqSensitivity.Sobol,p_range=[[0.0,1.0] for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],order=[0],args...; N=1000,kwargs...)
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,subject,param)
        sim.observed.dv
    end
    DiffEqSensitivity.gsa(f,p_range,method,N,order,args...; kwargs...)
end

function DiffEqSensitivity.gsa(m::PumasModel,subject::Subject,params::NamedTuple,method::DiffEqSensitivity.Morris,p_range=[[0.0,1.0] for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],
                p_steps=[100 for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],args...; kwargs...)
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,subject,param)
        sim.observed.dv
    end
    DiffEqSensitivity.gsa(f,p_range,method,p_steps,args...; kwargs...)
end

function DiffEqSensitivity.gsa(m::PumasModel,population::Population,params::NamedTuple,method::DiffEqSensitivity.Sobol,p_range=[[0.0,1.0] for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],N=1000,order=[0],args...; kwargs...)
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,population, param)
        collect(Iterators.flatten([sim.sims[i].observed.dv for i in 1:length(sim.sims)]))
    end
    DiffEqSensitivity.gsa(f,p_range,method,N,order,args...; kwargs...)
end

function DiffEqSensitivity.gsa(m::PumasModel,population::Population,params::NamedTuple,method::DiffEqSensitivity.Morris,p_range=[[0.0,1.0] for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],
                p_steps=[100 for i in 1:length(TransformVariables.inverse(toidentitytransform(m.param),params))],args...; kwargs...)
    trf_ident = toidentitytransform(m.param)
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,population, param)
        collect(Iterators.flatten([sim.sims[i].observed.dv for i in 1:length(sim.sims)]))
    end
    DiffEqSensitivity.gsa(f,p_range,method,p_steps,args...; kwargs...)
end