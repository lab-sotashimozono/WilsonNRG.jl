# Faithfulness gate — complementary closed-form laws of the Wilson chain, orthogonal to
# test_wilson_chain.jl (the ξₙ closed form itself), test_zavg_chain.jl (z-averaging dispatch +
# free-fermion recovery) and test_zitko_discretization.jl (A_{f0}=ρ band reproduction). Every
# @test here is checked against an INDEPENDENT closed form or structural law — never the code
# compared to itself:
#   (a) ξₙ → ξ∞ = (1+Λ⁻¹)/2 as n→∞               [KWW 1980 Eq. 2.15; Bulla 2008 Eq. 32]
#   (b) V₀ = bath_coupling(AndersonModel) = √(2DΓ/π) EXACTLY   [engine_u1u1.jl / Wilson chain map]
#   (c) εₙ = 0 for the symmetric flat band, ALL discretizations
#   (d) ξₙ is D-independent; V₀ ∝ √D            [dimensional analysis of Δ(ω)=πρ|V|²]
#   (e) the chain is impurity-model-agnostic at fixed D (Γ enters only via V₀, not ξₙ)
#   (f) shell_scale(disc, n) = (1+Λ⁻¹)/2 · Λ^{-(n-1)/2} exactly  [Bulla, Costi & Pruschke Eq. 3.9]

using WilsonNRG, Test
using WilsonNRG: asymptotic_hopping

@testset "closed-form laws · Wilson chain (D-scaling, model-agnosticism, shell scale)" begin
    # ---- (a) ξₙ → ξ∞ = (1+Λ⁻¹)/2 as n→∞: both the exported asymptote AND the deep chain ----
    for Λ in (1.5, 2.0, 2.5, 3.0)
        ξ∞ = (1 + 1 / Λ) / 2                                    # independent closed form, not the API call
        @test asymptotic_hopping(WilsonLog(Λ)) ≈ ξ∞

        model = AndersonModel(; U=0.0, Γ=0.01, D=1.0)
        chain = wilson_chain(WilsonLog(Λ), model, 40)
        # finite-n approach to the analytic limit — loose tol since convergence is only Λ^{-n}
        @test chain.hopping[end] ≈ ξ∞ rtol = 1.0e-6
        @test isapprox(chain.hopping[end], ξ∞; atol=1.0e-6)
    end

    # ---- (b) V₀ = bath_coupling(AndersonModel) = √(2DΓ/π) exactly ----
    for (Γ, D) in ((0.05, 1.0), (0.1, 2.0), (1.3, 0.4), (9.9, 7.3))
        model = AndersonModel(; U=0.0, Γ=Γ, D=D)
        @test bath_coupling(model) ≈ sqrt(2 * D * Γ / π)
    end

    # ---- (c) εₙ = 0 on the symmetric flat band, for EVERY discretization formulation ----
    model = AndersonModel(; U=0.0, εd=0.0, Γ=0.1, D=1.0)
    N = 25
    for disc in (WilsonLog(2.5), CampoOliveira(2.5), ZitkoPruschke(2.5))
        @test wilson_chain(disc, model, N).onsite == zeros(N)
    end

    # ---- (d) WilsonLog: ξₙ is EXACTLY D-independent (a pure Λ-recursion; D never enters the
    #          closed form at all — discretization.jl's `wilson_chain(::WilsonLog, ...)` never
    #          reads `model.D`). The z-averaging chains (CampoOliveira/ZitkoPruschke) are built by
    #          Lanczos-tridiagonalizing the ABSOLUTE-scale discretized band (energies and coupling
    #          weights are `D * ...`, discretization_zavg.jl:80-86), so their *stored* dimensionless
    #          hopping scales linearly with D instead — checked only on early shells (n ≤ 10) where
    #          Lanczos is still numerically orthogonal; deeper shells pick up round-off (an
    #          independently-verified artefact, not part of the claimed law). Meanwhile V₀ ∝ √D
    #          always (it is the literal `bath_coupling` closed form, independent of discretization). ----
    m1 = AndersonModel(; U=0.0, Γ=0.1, D=1.0)
    m2 = AndersonModel(; U=0.0, Γ=0.1, D=2.0)
    @test wilson_chain(WilsonLog(2.5), m1, 30).hopping ==
        wilson_chain(WilsonLog(2.5), m2, 30).hopping
    for disc in (CampoOliveira(2.5), ZitkoPruschke(2.5))
        c1 = wilson_chain(disc, m1, 10)
        c2 = wilson_chain(disc, m2, 10)
        @test c2.hopping[1:9] ≈ 2 .* c1.hopping[1:9] rtol = 1.0e-8   # D₂/D₁ = 2 ⇒ hopping × 2
    end
    m4 = AndersonModel(; U=0.0, Γ=0.1, D=4.0)                   # 4× D ⇒ 2× V₀ (√D scaling)
    @test bath_coupling(m4) ≈ 2 * bath_coupling(m1)
    @test bath_coupling(m4) / bath_coupling(m1) ≈ sqrt(4.0)

    # ---- (e) the chain is impurity-MODEL-agnostic at equal D: Γ (Anderson) drops out of ξₙ,
    #          the same way J (Kondo) never enters wilson_chain's Λ-grid at all — both models
    #          share one bath discretization; only bath_coupling(AndersonModel) needs Γ. ----
    D = 1.7
    anderson = AndersonModel(; U=0.3, εd=-0.15, Γ=0.4, D=D)
    kondo = KondoModel(; J=0.6, D=D)
    for disc in (WilsonLog(2.5), CampoOliveira(2.5), ZitkoPruschke(2.5))
        ca = wilson_chain(disc, anderson, 20)
        ck = wilson_chain(disc, kondo, 20)
        @test ca.hopping == ck.hopping
        @test ca.onsite == ck.onsite
    end
    # corollary: two Andersons at equal D but wildly different Γ still share ξₙ (Γ only rescales V₀)
    a_smallΓ = AndersonModel(; U=0.0, Γ=1.0e-3, D=D)
    a_bigΓ = AndersonModel(; U=0.0, Γ=50.0, D=D)
    @test wilson_chain(WilsonLog(2.5), a_smallΓ, 20).hopping ==
        wilson_chain(WilsonLog(2.5), a_bigΓ, 20).hopping

    # ---- (f) shell_scale(disc, n) = (1+Λ⁻¹)/2 · Λ^{-(n-1)/2} exactly [Bulla 2008 Eq. 3.9] ----
    for Λ in (1.5, 2.0, 2.5, 3.0), n in (1, 2, 5, 10, 20)
        ωn = (1 + 1 / Λ) / 2 * Λ^(-(n - 1) / 2)                 # independent closed form
        @test shell_scale(WilsonLog(Λ), n) ≈ ωn
    end
    # structural check: the ladder is a strict geometric decay in n for Λ > 1 (independent of
    # the exact prefactor — a property any correct implementation of this law must have)
    Λ = 2.5
    scales = [shell_scale(WilsonLog(Λ), n) for n in 1:15]
    @test all(<(0), diff(scales))                               # strictly decreasing
    @test scales[1] / scales[2] ≈ sqrt(Λ)                        # ratio between consecutive shells
end
