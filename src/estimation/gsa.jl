function DiffEqSensitivity.gsa(m::PumasModel, subject::Subject, params::NamedTuple, method::DiffEqSensitivity.GSAMethod, vars = [:dv], p_range_low=NamedTuple{keys(params)}([par.*0.05 for par in values(params)]), p_range_high=NamedTuple{keys(params)}([par.*1.95 for par in values(params)]), args...; kwargs...)
    vlowparam = TransformVariables.inverse(toidentitytransform(m.param), p_range_low)
    vhighparam = TransformVariables.inverse(toidentitytransform(m.param), p_range_high)
    p_range = [[vlowparam[i], vhighparam[i]] for i in 1:length(vlowparam)]
    trf_ident = toidentitytransform(m.param)
    sim_ = simobs(m, subject, params, args...; kwargs...)
    length_vars = append!([0], cumsum([length(sim_.observed[key]) for key in vars]))
    function f(p)
        param = TransformVariables.transform(trf_ident, p)
        sim = simobs(m, subject, param, args...; kwargs...)
        collect(Iterators.flatten([sim.observed[key] for key in vars])) 
    end
    sensitivity = DiffEqSensitivity.gsa(f, p_range, method)
    return sens_result(sensitivity, params, vars, length_vars, trf_ident)
end

function DiffEqSensitivity.gsa(m::PumasModel, population::Population, params::NamedTuple, method::DiffEqSensitivity.GSAMethod, vars = [:dv], p_range_low=NamedTuple{keys(params)}([par.*0.05 for par in values(params)]), p_range_high=NamedTuple{keys(params)}([par.*1.95 for par in values(params)]), args...; kwargs...)
    vlowparam = TransformVariables.inverse(toidentitytransform(m.param), p_range_low)
    vhighparam = TransformVariables.inverse(toidentitytransform(m.param), p_range_high)
    p_range = [[vlowparam[i], vhighparam[i]] for i in 1:length(vlowparam)]
    trf_ident = toidentitytransform(m.param)
    sim_ = simobs(m, population, params, args...; kwargs...)
    length_vars = append!([0], cumsum([length(sim_[1].observed[key]) for key in vars]))
    function f(p)
        param = TransformVariables.transform(trf_ident, p)
        sim = simobs(m, population, param, args...; kwargs...)
        mean([collect(Iterators.flatten([sim[i].observed[key] for key in vars])) for i in 1:length(sim)])
    end
    sensitivity = DiffEqSensitivity.gsa(f, p_range, method)
    return sens_result(sensitivity, params, vars, length_vars, trf_ident)
end

struct SobolOutput{T1, T2, T3, T4}
    first_order::T1
    total_order::T2
    first_order_conf_int::T3
    total_order_conf_int::T4
end

function sens_result(sens::DiffEqSensitivity.SobolResult, params::NamedTuple, vars::AbstractVector, length_vars::AbstractVector, trf_ident)
    first = NamedTuple()
    total = NamedTuple()
    first_ci_l = NamedTuple()
    first_ci_h = NamedTuple()
    tot_ci_l = NamedTuple()
    tot_ci_h = NamedTuple()
    sensi_1 = []
    sensi_ci_l = []
    sensi_ci_h = []
    sensi_T = []
    sensiT_ci_l = []
    sensiT_ci_h = []
    if length(sens.S1) > 0
        for ind in 1:length(length_vars)-1
            val_par = []
            conf_ints_low = []
            conf_ints_high = []
            for sen_p in sens.S1
                push!(val_par, sen_p[length_vars[ind]+1:length_vars[ind+1]])
            end
            push!(sensi_1, TransformVariables.transform(trf_ident, collect(Iterators.flatten(val_par))))
            if length(sens.S1_Conf_Int) > 0 
                for (conf_int_low, conf_int_high) in zip(sens.S1_Conf_Int[1], sens.S1_Conf_Int[2])
                    push!(conf_ints_low, conf_int_low[length_vars[ind]+1:length_vars[ind+1]])
                    push!(conf_ints_high, conf_int_high[length_vars[ind]+1:length_vars[ind+1]])
                end
                push!(sensi_ci_l, TransformVariables.transform(trf_ident, collect(Iterators.flatten(conf_ints_low))))
                push!(sensi_ci_h, TransformVariables.transform(trf_ident, collect(Iterators.flatten(conf_ints_high))))
            end
            
        end
        first = NamedTuple{Tuple(vars)}(sensi_1)
        if length(sens.S1_Conf_Int) > 0 
            first_ci_l = NamedTuple{Tuple(vars)}(sensi_ci_l)
            first_ci_h = NamedTuple{Tuple(vars)}(sensi_ci_h)
        end
    end
    if length(sens.ST) > 0
        for ind in 1:length(length_vars)-1
            val_par = []
            conf_ints_low = []
            conf_ints_high = []
            for sen_p in sens.ST
                push!(val_par, sen_p[length_vars[ind]+1:length_vars[ind+1]])
            end
            push!(sensi_T, TransformVariables.transform(trf_ident, collect(Iterators.flatten(val_par))))
            if length(sens.ST_Conf_Int) > 0 
                for (conf_int_low, conf_int_high) in zip(sens.ST_Conf_Int[1], sens.ST_Conf_Int[2])
                    push!(conf_ints_low, conf_int_low[length_vars[ind]+1:length_vars[ind+1]])
                    push!(conf_ints_high, conf_int_high[length_vars[ind]+1:length_vars[ind+1]])
                end
                push!(sensiT_ci_l, TransformVariables.transform(trf_ident, collect(Iterators.flatten(conf_ints_low))))
                push!(sensiT_ci_h, TransformVariables.transform(trf_ident, collect(Iterators.flatten(conf_ints_high))))
            end
            
        end
        total = NamedTuple{Tuple(vars)}(sensi_T)
        if length(sens.ST_Conf_Int) > 0 
            tot_ci_l = NamedTuple{Tuple(vars)}(sensiT_ci_l)
            tot_ci_h = NamedTuple{Tuple(vars)}(sensiT_ci_h)
        end
    end
    return SobolOutput(first, total, (max_conf_int = first_ci_h, min_conf_int = first_ci_l, ), (max_conf_int = tot_ci_h, min_conf_int = tot_ci_l, ))
end


struct MorrisOutput{T}
    μ::T
    variances::T
end

function sens_result(sens::DiffEqSensitivity.MorrisResult, params::NamedTuple, vars::AbstractVector, length_vars::AbstractVector, trf_ident)
    sensi_mean = []
    sensi_var = []
    for ind in 1:length(length_vars)-1
        means = []
        variances = []
        for (sen_p_mean, sen_p_var) in zip(sens.means, sens.variances)
            push!(means, sen_p_mean[length_vars[ind]+1:length_vars[ind+1]])
            push!(variances, sen_p_var[length_vars[ind]+1:length_vars[ind+1]])
        end
        push!(sensi_mean, TransformVariables.transform(trf_ident, collect(Iterators.flatten(means))))
        push!(sensi_var, TransformVariables.transform(trf_ident, collect(Iterators.flatten(variances))))
    end
    return MorrisOutput(NamedTuple{Tuple(vars)}(sensi_mean), NamedTuple{Tuple(vars)}(sensi_var))
end
