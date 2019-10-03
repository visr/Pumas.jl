function DiffEqSensitivity.gsa(m::PumasModel,subject::Subject,params::NamedTuple,method::DiffEqSensitivity.GSAMethod,vars = [:dv],p_range_low=NamedTuple{keys(params)}([par.*0.05 for par in values(params)]),p_range_high=NamedTuple{keys(params)}([par.*1.95 for par in values(params)]),args...; kwargs...)
    vlowparam = TransformVariables.inverse(toidentitytransform(m.param),p_range_low)
    vhighparam = TransformVariables.inverse(toidentitytransform(m.param),p_range_high)
    p_range = [[vlowparam[i],vhighparam[i]] for i in 1:length(vlowparam)]
    trf_ident = toidentitytransform(m.param)
    sim_ = simobs(m,subject,params,args...;kwargs...)
    length_vars = append!([0],cumsum([length(getproperty(sim_.observed,key)) for key in vars]))
    params_len = append!([0],cumsum([length(param) for param in params]))
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,subject,param,args...;kwargs...)
        collect(Iterators.flatten([getproperty(sim.observed,key) for key in vars])) 
    end
    sensitivity = DiffEqSensitivity.gsa(f,p_range,method)
    return sens_result(sensitivity,params,vars,length_vars,params_len,trf_ident)
end

function DiffEqSensitivity.gsa(m::PumasModel,population::Population,params::NamedTuple,method::DiffEqSensitivity.GSAMethod,vars = [:dv],p_range_low=NamedTuple{keys(params)}([par.*0.05 for par in values(params)]),p_range_high=NamedTuple{keys(params)}([par.*1.95 for par in values(params)]),args...; kwargs...)
    vlowparam = TransformVariables.inverse(toidentitytransform(m.param),p_range_low)
    vhighparam = TransformVariables.inverse(toidentitytransform(m.param),p_range_high)
    p_range = [[vlowparam[i],vhighparam[i]] for i in 1:length(vlowparam)]
    trf_ident = toidentitytransform(m.param)
    sim_ = simobs(m,population,params,args...;kwargs...)
    length_vars = append!([0],cumsum([length(getproperty(sim_.sims[1].observed,key)) for key in vars]))
    params_len = append!([0],cumsum([length(param) for param in params]))
    function f(p)
        param = TransformVariables.transform(trf_ident,p)
        sim = simobs(m,population,param,args...;kwargs...)
        mean([collect(Iterators.flatten([getproperty(sim.sims[i].observed,key) for key in vars])) for i in 1:length(sim.sims)])
    end
    sensitivity = DiffEqSensitivity.gsa(f,p_range,method)
    return sens_result(sensitivity,params,vars,length_vars,params_len,trf_ident)
end

function sens_result(sens::DiffEqSensitivity.SobolResult,params::NamedTuple,vars::AbstractVector,length_vars::AbstractVector,params_len,trf_ident)
    sensi = []
    if length(sens.S1) > 0
        for ind in 1:length(length_vars)-1
            val_par = []
            for sen_p in sens.S1
                push!(val_par,sen_p[length_vars[ind]+1:length_vars[ind+1]])
            end
            push!(sensi, TransformVariables.transform(trf_ident,collect(Iterators.flatten(val_par))))
        end
    end

    return NamedTuple{Tuple(vars)}(sensi)
end
