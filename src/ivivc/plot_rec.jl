function to_plottable(subj::Union{InVitroForm, UirData}; denseplot = true, plotdensity = 10_000)
  t = subj.time
  start = t[1]; stop = t[end]
  if denseplot
    plott = collect(range(start, stop=stop, length=plotdensity))
  else
    plott = t
  end
  plott, subj.m(plott, subj.pmin)
end

@recipe function f(subj::Union{InVitroForm, UirData}; denseplot = true, plotdensity = 10_000)
  t = subj.time
  start = t[1]; stop = t[end]
  if denseplot
    plott = collect(range(start, stop=stop, length=plotdensity))
  else
    plott = t
  end
  plott, subj.m(plott, subj.pmin)
end

@userplot Vitro_plot

@recipe function f(h::Vitro_plot; plotdensity = 10_000, denseplot = true)

  vitro_data = h.args[1]
  
  title  := "Estimated Vitro Model with Original data"
  xlabel := "time"
  ylabel := "Fraction Dissolved (Fdiss)" 
  legend := :bottomright

  for (form, prof) in vitro_data
    @series begin
      seriestype --> :scatter
      label --> "Original " * "$(form)"
      prof.time, prof.conc
    end
    @series begin
      seriestype --> :path
      label --> "Fitted " * "$(form)"
      x, y = to_plottable(prof, plotdensity = plotdensity, denseplot = denseplot)
      x, y
    end
  end
  primary := false
end

@userplot Ivivc_plot

@recipe function f(h::Ivivc_plot; plotdensity = 10_000, denseplot = true)

  model = h.args[1]
  @unpack vitro_data, vivo_data, fabs, pmin = model
  
  title  := "In Vitro In Vivo Plot"
  xlabel := "Fdiss(t * Tscale - Tshift)"
  ylabel := "FAbs Observed" 
  legend := :topleft

  for (form, prof) in vivo_data[1]
    @series begin
      seriestype --> :scatter
      label --> "$(form)"
      if length(pmin) > 2
        t = prof.time * pmin[2] .- pmin[3]
      else
        t = prof.time * pmin[2]
      end
      vitro_data[1][form](t), fabs[1][form]
    end
  end

  @series begin
    seriestype --> :path
    label --> "Estimated IVIVC Model"
    x = collect(range(0.0, stop=1.0, length=plotdensity))
    if length(pmin) == 2
      y = pmin[1] * x
    elseif length(pmin) == 3
      y = pmin[1] * x
    elseif length(pmin) == 4
      y = pmin[1] * x .- pmin[4]
    end
    x, y
  end
  primary := false
end


@userplot Levy_plot

@recipe function f(h::Levy_plot; plotdensity = 10_000, denseplot = true)

  model = h.args[1]
  @unpack vitro_data, vivo_data, fabs, pmin = model
  
  title  := "Levy Plot"
  xlabel := "TVitro"
  ylabel := "TVivo"
  legend := :topleft
  # i = 1
  for (form, prof) in vivo_data[1]
    conc_t  = collect(0.0:0.1:0.8)
    vitro_t = LinearInterpolation(vitro_data[1][form].time, vitro_data[1][form].conc)
    vivo_t  = LinearInterpolation(prof.time, fabs[1][form])
    # markers = [:hexagon, :rect, :circle]
    @series begin
      seriestype --> :scatter
      label --> "$(form)"
      # markershape --> markers[i]
      vitro_t.(conc_t), vivo_t.(conc_t)
    end
    # i = i + 1
  end
  primary := false
end

