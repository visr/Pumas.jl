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
    e = p.CL/p.V₁
    q = p.Q/p.V₁
    r = p.Q/p.V₂

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
                     iSV    iSV*( KV + SVi)/2])
    return Λ, 𝕍, 𝕍⁻¹
end
varnames(::Type{TwoCompartmentModel}) = (:Central, :Peri)
pk_init(::TwoCompartmentModel) = SLVector(Central=0.0, Peri=0.0)

struct TwoCompartmentFirstAbsorpModel <: ExplicitModel end
(m::TwoCompartmentFirstAbsorpModel)(args...) = _analytical_solve(m, args...)
@inline function LinearAlgebra.eigen(::TwoCompartmentFirstAbsorpModel, p)
    k = p.Ka
    e = p.CL/p.V₁
    q = p.Q/p.V₁
    r = p.Q/p.V₂

    eqr = e + q + r
    d = sqrt(eqr^2-4*e*r)
    Λ = @SVector([-k, (-eqr-d)/2, (-eqr+d)/2])
    S = sqrt((e+q)^2+2*(-e+q)*r+r^2)
    𝕍 = @SMatrix([(-k*(e-k+q)+(e-k)*r)/(k*q)  0            0;
                  (-k+r)/q                    -(e+q-r-S)/(2*q) (-e-q+r+S)/(2*q);
                  1                           1            1])
    𝕍⁻¹ = inv(𝕍)
    return Λ, 𝕍, 𝕍⁻¹
end
varnames(::Type{TwoCompartmentFirstAbsorpModel}) = (:Depot, :Central, :Peri)
pk_init(::TwoCompartmentFirstAbsorpModel) = SLVector(Depot=0.0, Central=0.0, Peri=0.0)
