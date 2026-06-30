# Faithfulness gate — full-density-matrix (FDM) spectral function A(ω) at finite T.
# FDM weights the complete basis of discarded states by the full thermal density matrix, so the
# spectral sum rule ∫A dω = 1 holds at ANY temperature (completeness) — far tighter than BHP
# patching. The gate checks: (1) the finite-T sum rule is tight AND tighter than BHP; (2) T=0
# reduces EXACTLY to CFS (delegation); (3) the ω≈0 region is finite (two-regime broadening tames
# the log-Gaussian 1/|ω| divergence on the dense quasi-elastic poles); (4) resonance at ~Γ;
# (5) the self-energy axis dispatches on FDM/CFS via Dyson (clean dispatch across spectral methods).
# Ref: Weichselbaum & von Delft, PRL 99, 076402 (2007).

using WilsonNRG, Test

@testset "method-recovery gate · FDM spectral function A(ω) [finite T]" begin
    Γ = 0.1
    m = AndersonModel(; U=0.0, εd=0.0, Γ, D=1.0)
    alg = NRGAlgorithm(;
        discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(120), nsites=18
    )
    trapz(x, y) = sum((y[i] + y[i + 1]) / 2 * (x[i + 1] - x[i]) for i in 1:(length(x) - 1))

    # ---- (1) finite-T sum rule ∫A dω ≈ 1 (completeness), and tighter than BHP ----
    T = 0.02
    res = spectral(FDM(), m, alg; T)
    ω, A = res.ω, res.A
    @test all(≥(-1.0e-12), A)                                   # non-negative (up to rounding)
    ∫A = trapz(ω, A)
    @test isapprox(∫A, 1.0; atol=0.03)                          # finite-T sum rule (got ≈1.003)
    A_bhp = spectral(BHP(), m, alg; ω=ω).A
    @test abs(∫A - 1) < abs(trapz(ω, A_bhp) - 1)                # tighter than BHP patching

    # ---- (2) T=0 reduces EXACTLY to CFS (FDM delegates to the GS projector) ----
    @test maximum(abs, spectral(FDM(), m, alg; T=0.0).A .- spectral(CFS(), m, alg).A) < 1.0e-12

    # ---- (3) ω≈0 finite: two-regime broadening tames the 1/|ω| divergence ----
    @test isfinite(A[argmin(abs.(ω))]) && maximum(A) < 10.0     # no near-zero spike

    # ---- (4) resonance at the hybridization scale ~Γ ----
    A_at(x) = A[argmin(abs.(ω .- x))]
    @test A_at(Γ) > 5 * A_at(1.0)                               # a resonance, not flat

    # ---- (5) clean dispatch: the self-energy axis works for FDM and CFS via Dyson ----
    seF = self_energy(FDM(), m, alg; via=Dyson(), T, ω=ω)
    @test length(seF.Σ) == length(ω) && eltype(seF.Σ) <: Complex
    @test length(self_energy(CFS(), m, alg; via=Dyson(), ω=ω).Σ) == length(ω)
end
