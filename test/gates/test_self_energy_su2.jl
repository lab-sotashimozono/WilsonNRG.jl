# Faithfulness gate — the DYSON self-energy Σ(ω) = ω − εd − Δ(ω) − 1/G is a deterministic
# functional of the Green's function G, and green_function(CFS) reproduces the U1U1 G on the
# U1SU2 (spin-SU(2)) engine to machine precision (test_cfs_su2.jl), so the U1SU2 Dyson self-energy
# must reproduce the U1U1 one — a cross-symmetry identity, not self-consistency.
#
# NB the Fermi-liquid pin ReΣ(0) = U/2 is carried by the self-energy TRICK (Σ = U·F/G, exact in
# F/G), which needs the BHP F-correlator and stays U1U1-only; plain Dyson is the broadening-limited
# comparison method (ReΣ(0) is noisy). So we assert the symmetry-independence of Σ, NOT the pinned
# value — and check that the trick still refuses non-U1U1 / non-BHP honestly.

using WilsonNRG, Test

@testset "faithfulness gate · Dyson self-energy on the U1SU2 engine" begin
    m = AndersonModel(; U=0.5, εd=-0.25, Γ=0.05, D=1.0)

    # ---- keep-all: G matches to machine precision ⇒ Σ matches ----
    @testset "keep-all Σ_U1SU2 == Σ_U1U1 · nsites=$nsites" for nsites in (3, 4)
        alg1 = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(10^9), nsites
        )
        algs = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1SU2(), truncation=KeepN(10^9), nsites
        )
        r1 = self_energy(CFS(), m, alg1; via=Dyson())
        r2 = self_energy(CFS(), m, algs; via=Dyson(), ω=r1.ω)
        @test r2.ω == r1.ω
        @test maximum(abs, real.(r2.Σ .- r1.Σ)) < 1.0e-9
        @test maximum(abs, imag.(r2.Σ .- r1.Σ)) < 1.0e-9
    end

    # ---- truncated: still tracks the U1U1 Dyson Σ near ω=0 (same CFS-G construction) ----
    @testset "truncated Σ tracks U1U1 near ω=0" begin
        alg1 = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(120), nsites=20
        )
        algs = NRGAlgorithm(;
            discretization=WilsonLog(2.5),
            symmetry=U1SU2(),
            truncation=KeepN(120),
            nsites=20,
        )
        r1 = self_energy(CFS(), m, alg1; via=Dyson())
        r2 = self_energy(CFS(), m, algs; via=Dyson(), ω=r1.ω)
        near0 = findall(x -> abs(x) < 0.1, r1.ω)
        @test maximum(abs, real.(r2.Σ[near0] .- r1.Σ[near0])) < 0.05
    end

    # ---- honest refusals: the trick needs BHP+U1U1; Dyson needs a supported symmetry ----
    @testset "honest refusals" begin
        algs = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1SU2(), truncation=KeepN(64), nsites=4
        )
        alg22 = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=SU2SU2(), truncation=KeepN(64), nsites=4
        )
        @test_throws EngineUnimplemented self_energy(BHP(), m, algs)                        # default trick → needs U1U1
        @test_throws EngineUnimplemented self_energy(CFS(), m, algs; via=SelfEnergyTrick()) # trick needs BHP
        @test_throws EngineUnimplemented self_energy(CFS(), m, alg22; via=Dyson())          # SU2SU2 not wired
    end
end
