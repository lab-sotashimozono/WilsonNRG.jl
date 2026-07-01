# Faithfulness gate — density-matrix NRG (DM-NRG) spectral function A(ω), Hofstetter PRL 85, 1508
# (2000); Bulla–Costi–Pruschke RMP 80, 395 (2008), Eqs. 86–88. DM-NRG uses the OFF-DIAGONAL reduced
# density matrix of the ground state (the coherences the ground-state-projector CFS drops).
#  (1) EXACT sum rule Σ pole weights = ⟨{d_σ,d†_σ}⟩ = 1 (the anticommutator identity — the general-ρ
#      Lehmann weights guarantee it regardless of the coherences); ∫A dω ≈ 1 after broadening.
#  (2) particle–hole symmetry A(ω)=A(−ω) at the symmetric point.
#  (3) resonance at the hybridization scale; non-negativity.
#  (4) DM-NRG refines CFS: same complete basis, so they agree closely, differing only by the
#      retained coherences (the off-diagonal reduced DM is what fixes e.g. the magnetic-field case).
#  (5) sum rule + p–h survive interactions (U>0); honest EngineUnimplemented stub for U1SU2.

using WilsonNRG, Test
using WilsonNRG: _cfs_collect, _dmnrg_reduced_dms, _dmnrg_poles

@testset "method-recovery gate · DM-NRG spectral function A(ω) [Hofstetter]" begin
    Γ = 0.1
    m = AndersonModel(; U=0.0, εd=0.0, Γ, D=1.0)
    alg = NRGAlgorithm(;
        discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(300), nsites=26
    )
    trapz(x, y) = sum((y[i] + y[i + 1]) / 2 * (x[i + 1] - x[i]) for i in 1:(length(x) - 1))

    # ---- (1) EXACT sum rule: Σ pole weights = 1 (anticommutator; independent of coherences) ----
    shells = _cfs_collect(m, alg)
    ρ = _dmnrg_reduced_dms(shells)
    poles = _dmnrg_poles(shells, ρ, 1)
    @test isapprox(sum(w for (_, w) in poles), 1.0; atol=1.0e-4)     # ⟨{d,d†}⟩ = 1, per spin
    @test isapprox(sum(w for (ω, w) in poles if ω > 0), 0.5; atol=1.0e-3)  # p–h: add = ½
    @test isapprox(sum(w for (ω, w) in poles if ω < 0), 0.5; atol=1.0e-3)  # p–h: remove = ½

    res = spectral(DMNRG(), m, alg)
    ω, A = res.ω, res.A
    @test all(≥(-1.0e-10), A)                                        # non-negative
    @test isapprox(trapz(ω, A), 1.0; atol=0.03)                      # broadened ∫A ≈ 1

    # ---- (2) particle–hole symmetry A(ω) = A(−ω) ----
    npos = length(ω) ÷ 2
    @test maximum(abs, A[(npos + 1):end] .- reverse(A[1:npos])) < 1.0e-3 * maximum(A)

    # ---- (3) resonance at ~Γ, decaying into the wings ----
    A_at(x) = A[argmin(abs.(ω .- x))]
    @test A_at(Γ) > 1.0
    @test A_at(Γ) > 5 * A_at(1.0)

    # ---- (4) DM-NRG refines CFS (same complete basis) — agree closely, differ by the coherences ----
    A_cfs = spectral(CFS(), m, alg; ω=ω).A
    @test maximum(abs, A .- A_cfs) < 0.1 * maximum(A_cfs)            # close (coherence-level refinement)
    @test abs(trapz(ω, A) - 1) ≤ abs(trapz(ω, A_cfs) - 1) + 0.02     # sum rule at least as tight

    # ---- (5) interactions: sum rule + p–h survive; honest stub for U1SU2 ----
    @testset "U>0 + honest stub" begin
        U = 0.5
        mU = AndersonModel(; U, εd=(-U / 2), Γ, D=1.0)
        pU = _dmnrg_poles(
            _cfs_collect(mU, alg), _dmnrg_reduced_dms(_cfs_collect(mU, alg)), 1
        )
        @test isapprox(sum(w for (_, w) in pU), 1.0; atol=1.0e-3)    # sum rule survives U>0
        rU = spectral(DMNRG(), mU, alg)
        npU = length(rU.ω) ÷ 2
        @test maximum(abs, rU.A[(npU + 1):end] .- reverse(rU.A[1:npU])) <
            1.0e-3 * maximum(rU.A)
        alg_su2 = NRGAlgorithm(; discretization=WilsonLog(2.5), symmetry=U1SU2(), nsites=5)
        @test_throws EngineUnimplemented green_function(DMNRG(), mU, alg_su2)
    end
end
