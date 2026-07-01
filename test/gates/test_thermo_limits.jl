# Faithfulness gate — impurity thermodynamics LIMITS, complementary to test_thermo.jl (which
# anchors the low-T local-moment plateau + Kondo screening) and test_magnetic.jl (which anchors
# the free-spin Brillouin curve + one FDT check against the hardcoded 1/4). Every @test here is
# checked against an INDEPENDENT closed-form statistical-mechanics or fluctuation–dissipation
# answer — never the code compared to itself. Refs: Krishna-murthy, Wilkins & Wilson, PRB 21,
# 1003/1044 (1980); Bulla, Costi & Pruschke, RMP 80, 395 (2008), §III.A.3.
#
# The textbook symmetric-Anderson crossover (Bulla–Costi–Pruschke §III.A.3):
#     free orbital  →   local moment   →  strong coupling (Kondo, screened singlet)
#     S_imp:  ln4 ≈1.386 →  ln2 ≈0.693  →   0
#     Tχ_imp:  1/8=0.125  →   1/4=0.25   →   0
# (a) high-T free orbital: all 4 impurity states (0, ↑, ↓, ↑↓) thermally degenerate ⇒ S_imp→ln4,
#     Tχ_imp→1/8. Needs U (and εd+U/2) ≪ T at the highest-T shell — U=0.5 (used for the low-T
#     plateau in test_thermo.jl) is NOT small enough there (T[1]≈0.75, so exp(-U/T[1])≈0.51 ≠ 1,
#     i.e. the double/empty states are still visibly Boltzmann-suppressed) — hence a much smaller
#     U here, at the SAME tiny Γ so the low-T local-moment plateau still forms cleanly.
# (b) local-moment plateau: DECOUPLED free spin (Γ→0) reproduces Tχ_imp=1/4, S_imp=ln2 EXACTLY
#     in the T→0 limit; reuses test_thermo.jl's EnergyCut recipe (fixed KeepN under-resolves the
#     impurity-doubled full run — see src/thermodynamics.jl).
# (c) Kondo screening: coupled large-U/Γ symmetric Anderson (test_thermo.jl's recipe) ⇒ S_imp,
#     Tχ_imp collapse WELL BELOW the local-moment values of (b) at the lowest reliable shells.
# (d) fluctuation–dissipation: the linear-response susceptibility χ_imp = dM_imp/dh|₀ from
#     `magnetization` reproduces χ_imp = Tχ_imp/T from `thermodynamics` — an independent
#     cross-CODE-PATH check (two different functions must agree), not a hardcoded 1/4.
# (e) free spin in a field: the decoupled-spin M_imp(T,h) matches the exact two-level formula
#     ½·tanh(h/2T) at several (h,T) points (not just the saturated low-T limit).

using WilsonNRG, Test

@testset "faithfulness gate · thermodynamic limits (free orbital, FDT)" begin
    # ---- (a) high-T free orbital: S_imp → ln4, Tχ_imp → 1/8 (all 4 states populated) ----
    @testset "free orbital: S_imp → ln4, Tχ_imp → 1/8 at the highest-T shell" begin
        # Small U (≪ T[1] ≈ 0.75 for Λ=2), still symmetric (εd=-U/2), tiny Γ so the SAME run's
        # low-T tail stays a clean local moment (cross-checked against test_thermo.jl's plateau).
        th = thermodynamics(
            AndersonModel(; U=0.04, εd=-0.02, Γ=0.0008, D=1.0),
            NRGAlgorithm(;
                discretization=WilsonLog(2.0),
                symmetry=U1U1(),
                truncation=EnergyCut(6.5),
                nsites=16,
            );
            betabar=1.0,
        )
        @test isapprox(th.S_imp[1], log(4); atol=0.05)          # ln4 = 1.3863 (loose: finite-U residual)
        @test isapprox(th.Tχ_imp[1], 0.125; atol=0.02)          # 1/8 (loose: finite-U residual)
        # sanity: still a genuine crossover (not stuck at free-orbital) — cools into the local moment
        @test th.S_imp[end] < 0.75 * th.S_imp[1]
        @test th.Tχ_imp[end] > 1.5 * th.Tχ_imp[1]
        @test all(isfinite, th.S_imp) && all(isfinite, th.Tχ_imp)
    end

    # ---- (b) local-moment plateau: decoupled free spin ⇒ Tχ_imp=1/4, S_imp=ln2 EXACTLY ----
    @testset "local moment: decoupled free spin → Tχ_imp=1/4, S_imp=ln2 (~1%)" begin
        th = thermodynamics(
            AndersonModel(; U=0.5, εd=-0.25, Γ=0.0008, D=1.0),   # V₀ tiny ⇒ T_K→0 (test_thermo.jl recipe)
            NRGAlgorithm(;
                discretization=WilsonLog(2.0),
                symmetry=U1U1(),
                truncation=EnergyCut(6.5),
                nsites=16,
            );
            betabar=1.0,
        )
        @test isapprox(th.Tχ_imp[end], 0.25; atol=0.01)         # 1/4, exact free-spin answer (~1%)
        @test isapprox(th.S_imp[end], log(2); atol=0.01)        # ln2, exact free-spin answer (~1%)
    end

    # ---- (c) Kondo screening: coupled large-U/Γ ⇒ S_imp, Tχ_imp ≪ the local-moment values ----
    @testset "Kondo screening: S_imp, Tχ_imp drop well below the local-moment plateau" begin
        moment = thermodynamics(
            AndersonModel(; U=0.5, εd=-0.25, Γ=0.0008, D=1.0),
            NRGAlgorithm(;
                discretization=WilsonLog(2.0),
                symmetry=U1U1(),
                truncation=EnergyCut(6.5),
                nsites=16,
            );
            betabar=1.0,
        )
        screened = thermodynamics(
            AndersonModel(; U=0.15, εd=-0.075, Γ=0.03, D=1.0),  # U/Γ=5 ⇒ T_K reachable (test_thermo.jl recipe)
            NRGAlgorithm(;
                discretization=WilsonLog(2.0),
                symmetry=U1U1(),
                truncation=EnergyCut(6.0),
                nsites=18,
            );
            betabar=1.0,
        )
        # independent cross-regime check: screened singlet ≪ local-moment plateau, not magic numbers
        @test screened.Tχ_imp[end] < 0.2 * moment.Tχ_imp[end]   # ≪ 1/4
        @test screened.S_imp[end] < 0.2 * moment.S_imp[end]     # ≪ ln2
        @test all(isfinite, screened.Tχ_imp) && all(isfinite, screened.S_imp)
    end

    # ---- (d) fluctuation–dissipation: dM_imp/dh|₀ (from magnetization) = χ_imp = Tχ_imp/T
    #          (from thermodynamics) — two independent code paths must agree, cross-checked
    #          directly against each other rather than against a hardcoded 1/4 ----
    @testset "FDT: linear-response χ_imp from magnetization matches Tχ_imp/T from thermodynamics" begin
        m = AndersonModel(; U=0.5, εd=-0.25, Γ=0.0008, D=1.0)     # local-moment regime (same as (b))
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.0),
            symmetry=U1U1(),
            truncation=EnergyCut(6.5),
            nsites=16,
        )
        th = thermodynamics(m, alg; betabar=1.0)
        h = 1.0e-4                                                # small ⇒ linear-response regime
        mag = magnetization(m, alg; h)
        @test mag.T == th.T                                       # same shell grid (sanity: aligned runs)
        lowT = findall(<(0.035), mag.T)                            # well into the local-moment plateau
        @test !isempty(lowT)
        for k in lowT
            χ_from_M = mag.M_imp[k] / h                            # linear response, dM/dh|₀
            χ_from_th = th.Tχ_imp[k] / th.T[k]                     # fluctuation susceptibility
            @test isapprox(χ_from_M, χ_from_th; rtol=0.05)         # ~1-5%: two independent code paths
        end
    end

    # ---- (e) free spin in a field: decoupled M_imp(T,h) = ½tanh(h/2T) exactly, several (h,T) ----
    @testset "free spin in a field: M_imp = ½tanh(h/2T) at several (h,T) points" begin
        m = AndersonModel(; U=0.5, εd=-0.25, Γ=0.0008, D=1.0)     # decoupled ⇒ exact free spin-½
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.0),
            symmetry=U1U1(),
            truncation=EnergyCut(6.5),
            nsites=16,
        )
        for h in (0.02, 0.05, 0.1)
            mag = magnetization(m, alg; h)
            lowT = findall(<(0.035), mag.T)                        # saturated/well-resolved regime
            @test !isempty(lowT)
            for k in lowT
                @test isapprox(mag.M_imp[k], 0.5 * tanh(h / (2 * mag.T[k])); atol=0.015)
            end
        end
    end
end
