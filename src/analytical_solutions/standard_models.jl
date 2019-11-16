export ImmediateAbsorptionModel, OneCompartmentModel, OneCompartmentParallelModel, TwoCompartmentModel, TwoCompartmentFirstAbsorpModel

abstract type ExplicitModel end
# Generic ExplicitModel solver. Uses an analytical eigen solution.
function _analytical_solve(m::M, t, t₀, amounts, doses, p, rates) where M<:ExplicitModel
  amt₀ = amounts + doses   # initial values for cmt's + new doses
  Λ, 𝕍, 𝕍⁻¹ = eigen(m, p)

  # We avoid the extra exp calls, but could have written:
  # Dh  = Diagonal(@SVector(exp.(λ * (_t - _t₀)))
  # Dp  = Diagonal(@SVector(expm1.(λ * (_t - _t₀))./λ))
  # Instead we write:
  Dp = Diagonal(expm1.(Λ * (t - t₀)) ./ Λ)
  Dh = Dp .* Λ + I
  amtₜ = 𝕍*(Dp*(𝕍⁻¹*rates) + Dh*(𝕍⁻¹*amt₀))

  return SLVector(NamedTuple{varnames(M)}(amtₜ))
end

struct ImmediateAbsorptionModel <: ExplicitModel end
(m::ImmediateAbsorptionModel)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::ImmediateAbsorptionModel, p)
  Ke = p.CL/p.V
  T = typeof(Ke)

  Λ = @SVector([-Ke])
  𝕍 = @SMatrix([T(1)])
  𝕍⁻¹ = 𝕍

  return Λ, 𝕍, 𝕍⁻¹
end
varnames(::Type{ImmediateAbsorptionModel}) = (:Central,)
pk_init(::ImmediateAbsorptionModel) = SLVector(Central=0.0)

struct OneCompartmentModel <: ExplicitModel end
(m::OneCompartmentModel)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::OneCompartmentModel, p)
    a = p.Ka
    e = p.CL/p.V

    Λ = @SVector([-a, -e])
    v = -1 + e/a
    𝕍 = @SMatrix([v 0;
                  1 1])

    iv = inv(v)
    𝕍⁻¹ = @SMatrix([iv   0;
                    -iv  1])

    return Λ, 𝕍, 𝕍⁻¹
end
varnames(::Type{OneCompartmentModel}) = (:Depot, :Central)
pk_init(::OneCompartmentModel) = SLVector(Depot=0.0,Central=0.0)

struct OneCompartmentParallelModel <: ExplicitModel end
(m::OneCompartmentParallelModel)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::OneCompartmentParallelModel, p)
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

    iv1 = inv(v1)
    iv2 = inv(v2)
    𝕍⁻¹ = @SMatrix([iv1 0 0; 0 iv2 0; -iv1 -iv2 1])
    return Λ, 𝕍, 𝕍⁻¹
end
varnames(::Type{OneCompartmentParallelModel}) = (:Depot1, :Depot2, :Central)
pk_init(::OneCompartmentParallelModel) = SLVector(Depot1=0.0,Depot2=0.0,Central=0.0)


struct TwoCompartmentModel <: ExplicitModel end
(m::TwoCompartmentModel)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::TwoCompartmentModel, p)
    e = p.CL/p.Vc
    q = p.Q/p.Vc
    r = p.Q/p.Vp

    K = e + q + r
    S = sqrt(K^2-4*e*r)
    Λ = @SVector([(-K-S)/2, (-K+S)/2])

    KV = e + q - r
    SV = sqrt(e^2 + 2*e*(q-r) + (q+r)^2)
    𝕍 = @SMatrix([-(KV + SV)/(2*q)  (- KV + SV)/(2*q);
                  1                  1])

    iSV = inv(SV)
    SVi = sqrt((e+q)^2 + 2*(-e+q)*r+r^2)
    𝕍⁻¹ = @SMatrix([-iSV*q  iSV*(-KV + SVi)/2;
                     iSV*q    iSV*( KV + SVi)/2])
    return Λ, 𝕍, 𝕍⁻¹
end
varnames(::Type{TwoCompartmentModel}) = (:Central, :Peripheral)
pk_init(::TwoCompartmentModel) = SLVector(Central=0.0, Peripheral=0.0)

struct TwoCompartmentFirstAbsorpModel <: ExplicitModel end
(m::TwoCompartmentFirstAbsorpModel)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::TwoCompartmentFirstAbsorpModel, p)
  k = p.Ka
  e = p.CL/p.Vc
  q = p.Q/p.Vc
  r = p.Q/p.Vp

  K = e + q + r
  S = sqrt(K^2-4*e*r)
  Λ = @SVector([-k, (-K-S)/2, (-K+S)/2])

  KV = e + q - r
  SV = sqrt(e^2 + 2*e*(q-r) + (q+r)^2)
  𝕍 = @SMatrix([(k^2-k*K+e*r)/(k*q)  0                 0;
                  (-k+r)/q               -(KV + SV)/(2*q)   (-KV + SV)/(2*q);
                  1                       1                1])

  v1i = inv(𝕍[1,1])
  iSV = inv(SV)
  SVi = sqrt((e+q)^2 + 2*(-e+q)*r+r^2)
  𝕍⁻¹ = @SMatrix([v1i                     0       0;
                  -iSV*(2*k-K+SVi)*v1i/2   -iSV*q  iSV*(-KV + SVi)/2;
                  -iSV*(-2*k+K+SVi)*v1i/2   iSV*q  iSV*( KV + SVi)/2])

  return Λ, 𝕍, 𝕍⁻¹
end
varnames(::Type{TwoCompartmentFirstAbsorpModel}) = (:Depot, :Central, :Peripheral)
pk_init(::TwoCompartmentFirstAbsorpModel) = SLVector(Depot=0.0, Central=0.0, Peripheral=0.0)


# use Vc and Vm
struct Metabolite11 <: ExplicitModel end # 011?
(m::Metabolite11)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::Metabolite11, p)
  a = p.CL1/p.V1
  b = p.Q11/p.V1
  c = p.Q11/p.Vp1
  d = p.Q12/p.V1
  e = p.CL2/p.V2
  f = p.Q21/p.V2
  h = p.Q21/p.Vp2

  G = a + b
  K̅ = a + b + c
  ϵ = e + f
  tf = 2*f
  K̲ = e + f + h
  S = sqrt(K̅^2 - 4*a*c)
  W = sqrt(K̲^2 - 4*e*h)
  Λ = @SVector([(-K̅ - S)/2, (-K̅ + S)/2,  (-K̲ - W)/2,  (-K̲ + W)/2])

  R = sqrt(G^2 + (c - 2*(a + b))*c)

  v11 = (-Λ[1]*2 - 2*ϵ)*(-Λ[1]*2 - 2*h)/(4*d*f)-h/d
  v21 = -(-a-b-Λ[1])*(-Λ[1]-ϵ)*(-Λ[1]*2-2*h)/(2*c*d*f)-(2*(a+b+Λ[1]))*h/(2*c*d)
  v31 = (Λ[1] + h)/f

  v12 = (-Λ[2]*2 - 2*ϵ)*(-Λ[2]*2 - 2*h)/(4*d*f)-h/d
  v22 = -(-a-b-Λ[2])*(-Λ[2]-ϵ)*(-Λ[2]*2-2*h)/(2*c*d*f)-(2*(a+b+Λ[2]))*h/(2*c*d)
  v32 = (Λ[2] + h)/f

  v33 = -(ϵ - h - W)/tf
  v34 = -(ϵ - h + W)/tf

  𝕍 = @SMatrix([v11  v12  0   0  ;
                v21  v22  0   0  ;
                v31  v32  v33 v34;
                1    1    1   1])

  𝕍⁻¹ = inv(𝕍)

  return Λ, 𝕍, 𝕍⁻¹
end
varnames(::Type{Metabolite11}) = (:Central, :CPeripheral, :Metabolite, :MPeripheral)
pk_init(::Metabolite11) = SLVector(Central=0.0, CPeripheral=0.0, Metabolite=0.0, MPeripheral=0.0
)
