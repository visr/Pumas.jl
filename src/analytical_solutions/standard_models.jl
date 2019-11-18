export Central1, DepotCentral1, TwoDepotsCentral1,
       Central2, DepotCentral2,
       Metabolite12, Metabolite22

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

struct Central1 <: ExplicitModel end
(m::Central1)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::Central1, p)
  Ke = p.CL/p.V
  T = typeof(Ke)

  Λ = @SVector([-Ke])
  𝕍 = @SMatrix([T(1)])

  return Λ, 𝕍
end
varnames(::Type{Central1}) = (:Central,)
pk_init(::Central1) = SLVector(Central=0.0)

struct DepotCentral1 <: ExplicitModel end
(m::DepotCentral1)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::DepotCentral1, p)
    a = p.Ka
    e = p.CL/p.V

    Λ = @SVector([-a, -e])
    v = e/a - 1
    𝕍 = @SMatrix([v 0;
                  1 1])

    return Λ, 𝕍
end
varnames(::Type{DepotCentral1}) = (:Depot, :Central)
pk_init(::DepotCentral1) = SLVector(Depot=0.0,Central=0.0)

struct TwoDepotsCentral1 <: ExplicitModel end
(m::TwoDepotsCentral1)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::TwoDepotsCentral1, p)
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
varnames(::Type{TwoDepotsCentral1}) = (:Depot1, :Depot2, :Central)
pk_init(::TwoDepotsCentral1) = SLVector(Depot1=0.0,Depot2=0.0,Central=0.0)

# b is from actual cmt to peri, c is back
struct Central2 <: ExplicitModel end
_V(::Central2, Λ, b, c) = @SMatrix([(Λ[1]+c)/b (Λ[2]+c)/b])
function _Λ(::Central2, a, b, c)
  A = a + b + c
  S = sqrt(A^2-4*a*c)
  Λ = @SVector([-(A+S)/2, -(A-S)/2])
end
(m::Central2)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(m::Central2, p)
    a = p.CL/p.Vc
    b = p.Q/p.Vc
    c = p.Q/p.Vp

    Λ = _Λ(m, a, b, c)
    𝕍 = vcat(_V(m, Λ, b, c), @SMatrix([1 1]))

    return Λ, 𝕍
end
varnames(::Type{Central2}) = (:Central, :Peripheral)
pk_init(::Central2) = SLVector(Central=0.0, Peripheral=0.0)

struct DepotCentral2 <: ExplicitModel end
(m::DepotCentral2)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::DepotCentral2, p)
  k = p.Ka
  a = p.CL/p.Vc
  b = p.Q/p.Vc
  c = p.Q/p.Vp

  A = a + b + c

  Λ, 𝕍 = eigen(Central2(), p)
  Λ = pushfirst(Λ, -k)

  𝕍 = vcat(@SMatrix([0 0;]), 𝕍) # pad with zeros
  v_depot = @SMatrix([((k-A)+a*c/k)/b; (c-k)/b; 1])
  𝕍 = hcat(v_depot, 𝕍)

  return Λ, 𝕍, inv(𝕍)
end
varnames(::Type{DepotCentral2}) = (:Depot, :Central, :Peripheral)
pk_init(::DepotCentral2) = SLVector(Depot=0.0, Central=0.0, Peripheral=0.0)


# use Vc and Vm
struct Metabolite22 <: ExplicitModel end # 011?
(m::Metabolite22)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::Metabolite22, p)
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

  m′ = Central2()
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
varnames(::Type{Metabolite22}) = (:Central, :CPeripheral, :Metabolite, :MPeripheral)
pk_init(::Metabolite22) = SLVector(Central=0.0, CPeripheral=0.0, Metabolite=0.0, MPeripheral=0.0
)

# use Vc and Vm
struct Metabolite12 <: ExplicitModel end # 011?
_Λ(::Metabolite12, a, b, c, d) = _Λ(Central2(), a+d, b, c)
(m::Metabolite12)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(m::Metabolite12, p)
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
varnames(::Type{Metabolite12}) = (:Central, :CPeripheral, :Metabolite)
pk_init(::Metabolite12) = SLVector(Central=0.0, CPeripheral=0.0, Metabolite=0.0)
