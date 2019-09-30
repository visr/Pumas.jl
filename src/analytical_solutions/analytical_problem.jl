struct PKPDAnalyticalProblem{uType,tType,isinplace,F,EV,T,P,B,K} <: DiffEqBase.AbstractAnalyticalProblem{uType,tType,isinplace}
  f::F
  u0::uType
  tspan::Tuple{tType,tType}
  events::EV
  times::T
  p::P
  bioav::B
  kwargs::K
  DiffEqBase.@add_kwonly function PKPDAnalyticalProblem{iip}(f,u0,tspan,
                                      events, times,
                                      p=DiffEqBase.NullParameters(),
                                      bioav = 1f0;
                                      kwargs...) where {iip}
    new{typeof(u0),promote_type(map(typeof,tspan)...),iip,
    typeof(f),typeof(events),typeof(times),typeof(p),typeof(bioav),
    typeof(kwargs)}(f,u0,tspan,events,times,p,bioav,kwargs)
  end
end

function PKPDAnalyticalProblem(f,u0,tspan,args...;kwargs...)
  iip = DiffEqBase.isinplace(f,7)
  PKPDAnalyticalProblem{iip}(f,u0,tspan,args...;kwargs...)
end

struct AnalyticalPKProblem{P1<:ExplicitModel,P2}
  pkprob::P1
  prob2::P2
end

struct PresetAnalyticalPKProblem{P,PK}
  numprob::P
  pksol::PK
end

struct NullDEProblem{P} <: DiffEqBase.DEProblem
  p::P
end

Base.summary(prob::NullDEProblem) = string(DiffEqBase.TYPE_COLOR, nameof(typeof(prob)),
                                                   DiffEqBase.NO_COLOR)

function Base.show(io::IO, A::NullDEProblem)
  println(io,summary(A.p))
  println(io)
end

export PKPDAnalyticalProblem, AnalyticalPKProblem
