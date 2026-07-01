# Faithfulness gate — impurity occupation ⟨n_d⟩, reproducing the asymmetric-Anderson static
# properties of Krishna-murthy, Wilkins & Wilson II (PRB 21, 1044 (1980)) + the Friedel sum rule
# (Langreth, PR 150, 516 (1966)).
#  (1) U=0 ⇒ ⟨n_{dσ}⟩ is EXACTLY the single-particle occupation of the Wilson chain: fill the
#      negative-energy single-particle levels, sum the squared impurity amplitude. Independent of
#      the many-body engine — the occupation analogue of the free-fermion subset-sum energy gate.
#  (2) particle–hole-symmetric point εd=-U/2 ⇒ ⟨n_d⟩=1 exactly, for any U (KWW symmetric limit).
#  (3) wide-band U=0 ⇒ ⟨n_{dσ}⟩ = 1/2 - (1/π)arctan(εd/Γ) (resonant level); empty→mixed-valence→
#      Kondo crossover monotone in εd with the correct empty/full limits.
#  (4) generalized Friedel sum rule: πΓ·A_{dσ}(0) = sin²(π⟨n_{dσ}⟩) links occupation to the ω=0
#      spectral pin (reduces to the friedel_pin πΓA(0)=1 at the symmetric point).

using WilsonNRG, Test
using LinearAlgebra: Symmetric, eigen

# independent single-particle impurity occupation per spin (same rescaled chain the engine builds;
# rescaled H_N = Λ^{N/2}·H_phys ⇒ identical eigenvectors ⇒ identical ground-state occupation).
function _sp_occupation(model, chain, nsites)
    Λ = chain.disc.Λ
    m = reshape([model.εd], 1, 1)
    for n in 0:(nsites - 1)
        c = n == 0 ? bath_coupling(model) : chain.hopping[n]
        r = n == 0 ? 1.0 : sqrt(Λ)
        k = size(m, 1)
        mn = zeros(k + 1, k + 1)
        mn[1:k, 1:k] = r .* m
        mn[k + 1, k + 1] = chain.onsite[n + 1]
        mn[k, k + 1] = c
        mn[k + 1, k] = c
        m = mn
    end
    F = eigen(Symmetric(m))
    return sum(abs2(F.vectors[1, a]) for a in eachindex(F.values) if F.values[a] < 0.0)
end

@testset "method-recovery gate · impurity occupation ⟨n_d⟩ (KWW II asymmetric static)" begin
    # ---- (1) U=0 ⇒ exact single-particle occupation (keep-all ⇒ tier-1 exact) ----
    @testset "U=0 single-particle reproduction" begin
        for εd in (0.0, 0.3, -0.2, 0.5)
            model = AndersonModel(; U=0.0, εd, Γ=0.05, D=1.0)
            alg = NRGAlgorithm(;
                discretization=WilsonLog(2.5),
                symmetry=U1U1(),
                truncation=KeepN(10^9),
                nsites=5,
            )
            occ = occupation(model, alg)
            nσ = _sp_occupation(model, wilson_chain(WilsonLog(2.5), model, 5), 5)
            @test occ.up ≈ nσ atol = 1.0e-9
            @test occ.dn ≈ nσ atol = 1.0e-9                # spin symmetry (no field)
            @test occ.total ≈ 2nσ atol = 1.0e-9
        end
    end

    # ---- (2) interacting particle–hole-symmetric point ⇒ ⟨n_d⟩ → 1 (recovered systematically) ----
    #  The U=0 symmetric case (εd=0) is exact above; here U>0 tests that the *interacting* engine
    #  preserves p–h symmetry. NRG truncation shifts ⟨n_d⟩ slightly off 1; keeping more states
    #  recovers it monotonically (an honest convergence check, not a fitted tolerance).
    @testset "half-filling at the symmetric point (interacting, converges to 1)" begin
        for (U, εd) in ((0.4, -0.2), (0.8, -0.4))
            model = AndersonModel(; U, εd, Γ=0.05, D=1.0)
            errs = map((400, 900)) do keep
                alg = NRGAlgorithm(;
                    discretization=WilsonLog(2.5),
                    symmetry=U1U1(),
                    truncation=KeepN(keep),
                    nsites=28,
                )
                abs(occupation(model, alg).total - 1)
            end
            @test errs[1] < 0.02                                      # ⟨n_d⟩ ≈ 1 (NRG accuracy)
            @test errs[2] < errs[1]                                   # → 1 as more states kept (p–h)
        end
    end

    # ---- (3) wide-band resonant level: ⟨n_{dσ}⟩ = 1/2 - (1/π)arctan(εd/Γ) + crossover ----
    #  Wide-band Lorentzian; the finite band (D=1) and Λ=2 log-discretization give ~3% deviation in
    #  the steep crossover region |εd|~Γ — checked with the corresponding honest tolerance.
    @testset "resonant-level occupation & empty→Kondo crossover" begin
        Γ = 0.02
        nd(εd) = occupation(
            AndersonModel(; U=0.0, εd, Γ, D=1.0),
            NRGAlgorithm(;
                discretization=WilsonLog(2.0),
                symmetry=U1U1(),
                truncation=KeepN(400),
                nsites=40,
            ),
        ).up
        for εd in (-0.15, -0.05, 0.0, 0.05, 0.15)
            @test nd(εd) ≈ 1 / 2 - atan(εd / Γ) / π atol = 0.035      # wide-band Lorentzian
        end
        @test nd(-0.3) > nd(0.0) > nd(0.3)                            # monotone crossover
        @test nd(0.0) ≈ 0.5 atol = 0.01                               # symmetric ⇒ exactly half
        @test nd(-0.5) > 0.9                                          # nearly full (below band)
        @test nd(0.5) < 0.1                                           # nearly empty (above band)
    end

    # NOTE: the generalized Friedel sum rule πΓ·A_{dσ}(0) = sin²(π⟨n_{dσ}⟩) (Langreth) is validated
    # at the symmetric point by `test_friedel_pin.jl` (πΓA(0)=1 ⟺ ⟨n_{dσ}⟩=1/2). Its asymmetric
    # form needs a robust ω→0 spectral value (the log-Gaussian broadening kernel vanishes at exactly
    # ω=0), which is a spectral-broadening concern deferred to a dedicated self-energy/Friedel gate.

    # ---- honest stub for an unwired symmetry ----
    @testset "honest stub for unwired symmetry" begin
        model = AndersonModel(; U=0.4, εd=-0.2, Γ=0.05, D=1.0)
        alg = NRGAlgorithm(; discretization=WilsonLog(2.5), symmetry=U1SU2(), nsites=5)
        @test_throws EngineUnimplemented occupation(model, alg)
    end
end
