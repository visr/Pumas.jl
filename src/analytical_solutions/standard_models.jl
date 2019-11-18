export OneCmtModel, OneCmtDepotModel, OneCmtTwoDepotModel,
       TwoCmtPeriModel, TwoCmtDepotPeriModel,
       Metabolite01, Metabolite011

abstract type ExplicitModel end
# Generic ExplicitModel solver. Uses an analytical eigen solution.
function _analytical_solve(m::M, t, t₀, amounts, doses, p, rates) where M<:ExplicitModel
  amt₀ = amounts + doses   # initial values for cmt's + new doses
  Λ, 𝕍 = eigen(m, p)

  # We avoid the extra exp calls, but could have written:
  # Dh  = Diagonal(@SVector(exp.(λ * (_t - _t₀)))
  # Dp  = Diagonal(@SVector(expm1.(λ * (_t - _t₀))./λ))
  # Instead we write:
  Dp = Diagonal(expm1.(Λ * (t - t₀)) ./ Λ)
  Dh = Dp .* Λ + I
  amtₜ = 𝕍*(Dp*(𝕍\rates) + Dh*(𝕍\amt₀)) # could derive inverse here

  return SLVector(NamedTuple{varnames(M)}(amtₜ))
end
DiffEqBase.has_syms(x::ExplicitModel) = true
Base.getproperty(x::ExplicitModel, symbol::Symbol) = symbol == :syms ? Pumas.varnames(typeof(x)) : getfield(x, symbol)

struct OneCmtModel <: ExplicitModel end
(m::OneCmtModel)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::OneCmtModel, p)
  Ke = p.CL/p.V
  T = typeof(Ke)

  Λ = @SVector([-Ke])
  𝕍 = @SMatrix([T(1)])

  return Λ, 𝕍
end
varnames(::Type{OneCmtModel}) = (:Central,)
pk_init(::OneCmtModel) = SLVector(Central=0.0)

struct OneCmtDepotModel <: ExplicitModel end
(m::OneCmtDepotModel)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::OneCmtDepotModel, p)
    a = p.Ka
    e = p.CL/p.V

    Λ = @SVector([-a, -e])
    v = e/a - 1
    𝕍 = @SMatrix([v 0;
                  1 1])

    return Λ, 𝕍
end
varnames(::Type{OneCmtDepotModel}) = (:Depot, :Central)
pk_init(::OneCmtDepotModel) = SLVector(Depot=0.0,Central=0.0)

struct OneCmtTwoDepotModel <: ExplicitModel end
(m::OneCmtTwoDepotModel)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::OneCmtTwoDepotModel, p)
    a = p.Ka1
    b = p.Ka2
    e = p.CL/p.V

    frac1 = (e-a)/a
    invfrac1 = inv(frac1)

    frac2 = (e-b)/b
    invfrac2 = inv(frac2)

    Λ = @SVector([-a, -b, -e])

    v1 = -1 + e/a
    v2 = -1 + e/b
    𝕍 = @SMatrix([frac1 0     0;
                  0     frac2 0;
                  1     1     1])

    return Λ, 𝕍
end
varnames(::Type{OneCmtTwoDepotModel}) = (:Depot1, :Depot2, :Central)
pk_init(::OneCmtTwoDepotModel) = SLVector(Depot1=0.0,Depot2=0.0,Central=0.0)

function _Λ(::TwoCmtPeriModel, a, b, c)
  A = a + b + c
  S = sqrt(A^2-4*a*c)
  Λ = @SVector([-(A+S)/2, -(A-S)/2])
end
# b is from actual cmt to peri, c is back
struct TwoCmtPeriModel <: ExplicitModel end
_V(::TwoCmtPeriModel, Λ, b, c) = @SMatrix([(Λ[1]+c)/b (Λ[2]+c)/b])
(m::TwoCmtPeriModel)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(m::TwoCmtPeriModel, p)
    a = p.CL/p.Vc
    b = p.Q/p.Vc
    c = p.Q/p.Vp

    Λ = _Λ(m, a, b, c)
    𝕍 = vcat(_V(m, Λ, b, c), @SMatrix([1 1]))

    return Λ, 𝕍
end
varnames(::Type{TwoCmtPeriModel}) = (:Central, :Peripheral)
pk_init(::TwoCmtPeriModel) = SLVector(Central=0.0, Peripheral=0.0)

struct TwoCmtDepotPeriModel <: ExplicitModel end
(m::TwoCmtDepotPeriModel)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::TwoCmtDepotPeriModel, p)
  k = p.Ka
  a = p.CL/p.Vc
  b = p.Q/p.Vc
  c = p.Q/p.Vp

  A = a + b + c

  Λ, 𝕍 = eigen(TwoCmtPeriModel(), p)
  Λ = pushfirst(Λ, -k)

  𝕍 = vcat(@SMatrix([0 0;]), 𝕍) # pad with zeros
  v_depot = @SMatrix([((k-A)+a*c/k)/b; (c-k)/b; 1])
  𝕍 = hcat(v_depot, 𝕍)

  return Λ, 𝕍, inv(𝕍)
end
varnames(::Type{TwoCmtDepotPeriModel}) = (:Depot, :Central, :Peripheral)
pk_init(::TwoCmtDepotPeriModel) = SLVector(Depot=0.0, Central=0.0, Peripheral=0.0)


# use Vc and Vm
struct Metabolite011 <: ExplicitModel end # 011?
(m::Metabolite011)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::Metabolite011, p)
  a = p.CL1/p.V1
  b = p.Q1/p.V1
  c = p.Q1/p.Vp1
  d = p.T/p.V1
  e = p.CL2/p.V2
  f = p.Q2/p.V2
  h = p.Q2/p.Vp2

  a′ = a + d
  α′ = a′ + b
  ϵ = e + f

  m′ = TwoCmtPeriModel()
  Λ = vcat(_Λ(m′, a′, b, c),  _Λ(m′, e, f, h))

  v1_3 = (Λ[1] + h)/f
  v1_1 = ((Λ[1] + ϵ) * v1_3 - h)/d
  v1_2 = (Λ[1] + α′) * v1_1/c

  v2_3 = (Λ[2] + h)/f
  v2_1 = ((Λ[2] + ϵ) * v2_3 - h)/d
  v2_2 = (Λ[2] + α′) * v2_1/c


  v3_3 = (Λ[3] + h)/f
  v4_3 = (Λ[4] + h)/f

  𝕍 = @SMatrix([v1_1  v2_1  0   0  ;
                v1_2  v2_2  0   0  ;
                v1_3  v2_3  v3_3 v4_3;
                1     1    1   1])

  return Λ, 𝕍
end
varnames(::Type{Metabolite011}) = (:Central, :CPeripheral, :Metabolite, :MPeripheral)
pk_init(::Metabolite011) = SLVector(Central=0.0, CPeripheral=0.0, Metabolite=0.0, MPeripheral=0.0
)

# use Vc and Vm
_Λ(::Metabolite01, a, b, c, d) = _Λ(TwoCmtPeriModel(), a+d, b, c)
struct Metabolite01 <: ExplicitModel end # 011?
(m::Metabolite01)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(m::Metabolite01, p)
  a = p.CL1/p.V1
  b = p.Q11/p.V1
  c = p.Q11/p.Vp1
  d = p.Q12/p.V1
  e = p.CL2/p.V2

  α = a + b + c + d
  Λ = _Λ(m, a, b, c, d)


  v1_1 = (Λ[1] + ϵ)/d
  v1_2 = (Λ[1] + α - c)*(Λ[1] + e)/(c*d)
  v2_1 = (Λ[2] + ϵ)/d
  v2_2 = (Λ[2] + α - c)*(Λ[2] + e)/(c*d)

  𝕍 = @SMatrix([v1_1 v2_1 0;
                v1_2 v2_2 0;
                1    1    1])

  return Λ, 𝕍
end
varnames(::Type{Metabolite01}) = (:Central, :CPeripheral, :Metabolite)
pk_init(::Metabolite01) = SLVector(Central=0.0, CPeripheral=0.0, Metabolite=0.0)
