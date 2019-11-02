using Test
using Pumas
using LinearAlgebra, DiffEqSensitivity

@testset "GSA Tests" begin
choose_covariates() = (isPM = rand([1, 0]),
                       Wt = rand(55:80))

function generate_population(events,nsubs=4)
  pop = Population(map(i -> Subject(id=i,evs=events,cvs=choose_covariates()),1:nsubs))
  return pop
end

ev = DosageRegimen(100, cmt = 2)
ev2 = generate_population(ev)

m_diffeq = @model begin
  @param   begin
    θ1 ∈ RealDomain(lower=0.1,  upper=3)
    θ2 ∈ RealDomain(lower=0.5,  upper=10)
    θ3 ∈ RealDomain(lower=10,  upper=30)	
  end

  @pre begin
    Ka = θ1
    CL = θ2
    V  = θ3
  end

  @covariates isPM Wt

  @dynamics begin
    Depot'   = -Ka*Depot
    Central' =  Ka*Depot - (CL/V)*Central
  end

  @derived begin
    cp = @. 1000*(Central / V)
    nca := @nca cp
    auc =  NCA.auc(nca)
    thalf =  NCA.thalf(nca)
    cmax = NCA.cmax(nca)
  end
end


p = (  θ1 = 1.5,  #Ka
       θ2  =  1.1,  #CL
       θ3  =   20.0  #V
           ,
    )

sobol = gsa(m_diffeq,
            ev2,
            p,
            DiffEqSensitivity.Sobol(N=10000),
            [:auc], (θ1 = 0.1, θ2 = 0.5, θ3 = 10))

@test sobol.first_order[1][1] ≈ 0.0 atol = 1e-2
@test sobol.first_order[1][2] ≈ 1.035837606317531 atol = 4e-1
@test sobol.first_order[1][3] ≈ -5.6704008995506364e-5 atol = 1e-2
@test sobol.total_order[1][1] ≈ 0.0 atol = 1e-2
@test sobol.total_order[1][2] ≈ 0.9591774724587756 atol = 4e-1
@test sobol.total_order[1][3] ≈ 4.113887839053021e-7 atol = 1e-2
sobol_subj = sobol = gsa(m_diffeq,
            ev2[1],
            p,
            DiffEqSensitivity.Sobol(),
            [:auc], (θ1 = 0.1, θ2 = 0.5, θ3 = 10))
@test sprint((io, t) -> show(io, MIME"text/plain"(), t), sobol_subj) ==
"""
First Order Indices

Derived Variable: auc

Parameter first order indices 

θ1          0.0

θ2          $(sobol_subj.first_order[1][2])

θ3          $(sobol_subj.first_order[1][3])

Total Order Indices

Derived Variable: auc

Parameter total order indices 

θ1          0.0

θ2          $(sobol_subj.total_order[1][2])

θ3          $(sobol_subj.total_order[1][3])

"""

end
