# Faithfulness gate — Žitko-Pruschke (PRB 79, 085106 (2009)) improved discretization reproduces
# the conduction band. The z-averaged local DOS at the first Wilson site, A_{f0}(ω), must equal
# the band DOS ρ(ω)=1/(2D) (Žitko Eq. 31). The Žitko-Pruschke representative energy (Eq. 35/36) is
# constructed to make A_{f0}=ρ EXACTLY; the conventional (Eq. 33, arithmetic mean) and
# Campo-Oliveira (Eq. 32, log mean) recipes leave band-edge artefacts. This reproduces Fig. 6/7.

using WilsonNRG, Test

@testset "method-recovery gate · Žitko-Pruschke discretization reproduces the band A_{f0}=ρ" begin
    D = 1.0
    ρ = 1 / (2D)
    m = AndersonModel(; U=0.0, εd=0.0, Γ=0.1, D)
    disc = ZitkoPruschke(2.5)

    # ---- Žitko-Pruschke: A_{f0}(ω) = ρ exactly across the band (the paper's construction) ----
    zk = band_dos(disc, m; scheme=:zitko)
    @test !isempty(zk.A)
    @test all(0 .< zk.ω .≤ D)                                  # samples lie within the band
    @test maximum(abs, zk.A .- ρ) < 1.0e-6                     # band reproduced (got ≈1e-9)

    # ---- conventional & Campo-Oliveira leave band-edge artefacts (Fig. 6/7) ----
    co = band_dos(disc, m; scheme=:campo_oliveira)
    cv = band_dos(disc, m; scheme=:conventional)
    @test maximum(abs, co.A .- ρ) > 0.1                        # Campo-Oliveira artefacts
    @test maximum(abs, cv.A .- ρ) > 0.1                        # conventional artefacts (worse)
    @test maximum(abs, zk.A .- ρ) < maximum(abs, co.A .- ρ)    # Žitko strictly better than CO
    @test maximum(abs, co.A .- ρ) < maximum(abs, cv.A .- ρ)    # CO better than conventional

    # ---- D-scaling: A_{f0} = ρ = 1/(2D) ----
    m2 = AndersonModel(; U=0.0, εd=0.0, Γ=0.1, D=2.0)
    @test maximum(abs, band_dos(disc, m2; scheme=:zitko).A .- 1 / (2 * 2.0)) < 1.0e-6
end
