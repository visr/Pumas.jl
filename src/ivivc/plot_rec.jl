function to_plottable(subj::Union{VitroForm, UirData}; denseplot = true, plotdensity = 10_000)
  t = subj.time
  start = t[1]; stop = t[end]
  if denseplot
    plott = collect(range(start, stop=stop, length=plotdensity))
  else
    plott = t
  end
  plott, subj.m(plott, subj.pmin)
end

@recipe function f(subj::Union{VitroForm, UirData}; denseplot = true, plotdensity = 10_000)
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
  
  # title  := "In Vitro In Vivo"
  xlabel := "Fdiss(t * Tscale)"
  ylabel := "FAbs Observed" 
  legend := :topleft

  for (form, prof) in vivo_data[1]
    @series begin
      seriestype --> :scatter
      label --> "$(form)"
      vitro_data[1][form](prof.time * pmin[2]), fabs[1][form]
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
