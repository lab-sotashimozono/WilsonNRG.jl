# Faithfulness gate — extended SU(2) angular-momentum coefficients.
#
# Every expected value below is recomputed independently from the textbook Racah-formula
# definitions (Edmonds, "Angular Momentum in Quantum Mechanics"; cross-checked against the
# equivalent tables in Varshalovich–Moskalev–Khersonskii) — NOT read off from src/su2.jl and
# NOT self-consistency of the code with itself. The orthogonality/completeness/symmetry blocks
# assert exact algebraic identities (Racah orthogonality theorems, permutation symmetries of
# the 3j/6j symbols, the 3j↔CG relation) that hold for ANY correct implementation; they are laws,
# not tautologies, because a broken formula (wrong sign, dropped term, truncated Racah sum) will
# violate them even though it might still pass isolated spot values.
#
# Argument order (confirmed by reading src/su2.jl):
#   clebsch_gordan(j1, m1, j2, m2, J, M)     -- ⟨j1 m1; j2 m2 | J M⟩
#   wigner3j(j1, j2, j3, m1, m2, m3)         -- (j1 j2 j3; m1 m2 m3)
#   wigner6j(a, b, c, d, e, f)               -- {a b c; d e f}
# Half-integers are passed as Rational (e.g. 1//2). Sign convention verified against the
# trivially-known stretched state CG(1/2,1/2;1/2,1/2|1,1) = +1 and the standard (Condon–Shortley)
# singlet CG(1/2,1/2;1/2,-1/2|0,0) = +1/√2 (test/gates/test_su2_coeffs.jl already pins these).

using WilsonNRG, Test
using WilsonNRG: clebsch_gordan, wigner3j, wigner6j

@testset "method-recovery gate · extended angular-momentum coefficients" begin
    # =========================================================================================
    # (a) Clebsch–Gordan vs independently recomputed table values (≥ 12 entries)
    # =========================================================================================
    @testset "CG vs Edmonds/Varshalovich table values" begin
        # j1=j2=1/2 coupling (singlet/triplet)
        @test clebsch_gordan(1 // 2, 1 // 2, 1 // 2, -1 // 2, 0, 0) ≈ 1 / sqrt(2)
        @test clebsch_gordan(1 // 2, 1 // 2, 1 // 2, -1 // 2, 1, 0) ≈ 1 / sqrt(2)

        # j1=j2=1 coupling (2⊗1 = 2⊕1⊕0)
        @test clebsch_gordan(1, 0, 1, 0, 2, 0) ≈ sqrt(2 / 3)
        @test clebsch_gordan(1, 1, 1, -1, 2, 0) ≈ 1 / sqrt(6)
        @test clebsch_gordan(1, 1, 1, -1, 1, 0) ≈ 1 / sqrt(2)
        @test clebsch_gordan(1, 1, 1, -1, 0, 0) ≈ 1 / sqrt(3)

        # j1=1, j2=1/2 coupling (1⊗1/2 = 3/2⊕1/2)
        @test clebsch_gordan(1, 1, 1 // 2, -1 // 2, 3 // 2, 1 // 2) ≈ 1 / sqrt(3)
        @test clebsch_gordan(1, 1, 1 // 2, 1 // 2, 3 // 2, 3 // 2) ≈ 1.0
        @test clebsch_gordan(1, 0, 1 // 2, 1 // 2, 3 // 2, 1 // 2) ≈ sqrt(2 / 3)
        @test clebsch_gordan(1, 0, 1 // 2, 1 // 2, 1 // 2, 1 // 2) ≈ -1 / sqrt(3)

        # j1=3/2, j2=1/2 coupling (3/2⊗1/2 = 2⊕1)
        @test clebsch_gordan(3 // 2, 3 // 2, 1 // 2, -1 // 2, 2, 1) ≈ 1 / 2
        @test clebsch_gordan(3 // 2, 1 // 2, 1 // 2, 1 // 2, 2, 1) ≈ sqrt(3) / 2
        @test clebsch_gordan(3 // 2, 1 // 2, 1 // 2, -1 // 2, 1, 0) ≈ 1 / sqrt(2)

        # j1=2, j2=1 coupling (2⊗1 = 3⊕2⊕1)
        @test clebsch_gordan(2, 0, 1, 0, 3, 0) ≈ sqrt(3 / 5)
        @test clebsch_gordan(2, 1, 1, -1, 3, 0) ≈ 1 / sqrt(5)
    end

    # =========================================================================================
    # (b) CG orthogonality: Σ_{m1,m2} CG(j1,m1;j2,m2|J,M)·CG(j1,m1;j2,m2|J′,M′) = δ_JJ′ δ_MM′
    # =========================================================================================
    @testset "CG orthogonality (Racah orthogonality theorem, first kind)" begin
        function cg_orthogonality_sum(j1, j2, J, M, Jp, Mp)
            s = 0.0
            for m1 in (-j1):1:j1, m2 in (-j2):1:j2
                m1 + m2 == M || continue  # CG(...,J,M) vanishes unless m1+m2=M
                m1 + m2 == Mp || continue # and CG(...,J',M') vanishes unless m1+m2=M'
                s +=
                    clebsch_gordan(j1, m1, j2, m2, J, M) *
                    clebsch_gordan(j1, m1, j2, m2, Jp, Mp)
            end
            return s
        end

        # j1=1, j2=1/2: J,J' ∈ {1/2, 3/2}
        @test cg_orthogonality_sum(1, 1 // 2, 3 // 2, 1 // 2, 3 // 2, 1 // 2) ≈ 1.0
        @test cg_orthogonality_sum(1, 1 // 2, 1 // 2, 1 // 2, 1 // 2, 1 // 2) ≈ 1.0
        @test cg_orthogonality_sum(1, 1 // 2, 3 // 2, 1 // 2, 1 // 2, 1 // 2) ≈ 0.0 atol =
            1e-12
        @test cg_orthogonality_sum(1, 1 // 2, 3 // 2, -1 // 2, 1 // 2, -1 // 2) ≈ 0.0 atol =
            1e-12

        # j1=1, j2=1: J,J' ∈ {0,1,2}, M=M'=0 vs M≠M' cross term
        @test cg_orthogonality_sum(1, 1, 2, 0, 2, 0) ≈ 1.0
        @test cg_orthogonality_sum(1, 1, 1, 0, 1, 0) ≈ 1.0
        @test cg_orthogonality_sum(1, 1, 0, 0, 0, 0) ≈ 1.0
        @test cg_orthogonality_sum(1, 1, 2, 0, 1, 0) ≈ 0.0 atol = 1e-12
        @test cg_orthogonality_sum(1, 1, 2, 0, 0, 0) ≈ 0.0 atol = 1e-12
        @test cg_orthogonality_sum(1, 1, 1, 0, 0, 0) ≈ 0.0 atol = 1e-12
        @test cg_orthogonality_sum(1, 1, 2, 1, 2, 1) ≈ 1.0
        @test cg_orthogonality_sum(1, 1, 2, 1, 1, 1) ≈ 0.0 atol = 1e-12  # same M, different J

        # j1=3/2, j2=1/2: J,J' ∈ {1,2}
        @test cg_orthogonality_sum(3 // 2, 1 // 2, 2, 1, 2, 1) ≈ 1.0
        @test cg_orthogonality_sum(3 // 2, 1 // 2, 1, 0, 1, 0) ≈ 1.0
        @test cg_orthogonality_sum(3 // 2, 1 // 2, 2, 0, 1, 0) ≈ 0.0 atol = 1e-12
    end

    # =========================================================================================
    # (c) CG completeness: Σ_{J,M} CG(j1,m1;j2,m2|J,M)·CG(j1,m1′;j2,m2′|J,M) = δ_{m1m1′} δ_{m2m2′}
    # =========================================================================================
    @testset "CG completeness (Racah orthogonality theorem, second kind)" begin
        function cg_completeness_sum(j1, j2, m1, m2, m1p, m2p)
            s = 0.0
            for J in abs(j1 - j2):1:(j1 + j2)
                M = m1 + m2
                Mp = m1p + m2p
                M == Mp || continue  # both CG factors vanish unless M is shared
                abs(M) ≤ J || continue
                s +=
                    clebsch_gordan(j1, m1, j2, m2, J, M) *
                    clebsch_gordan(j1, m1p, j2, m2p, J, M)
            end
            return s
        end

        # j1=1, j2=1/2
        @test cg_completeness_sum(1, 1 // 2, 1, 1 // 2, 1, 1 // 2) ≈ 1.0
        @test cg_completeness_sum(1, 1 // 2, 1, 1 // 2, 0, -1 // 2) ≈ 0.0 atol = 1e-12
        @test cg_completeness_sum(1, 1 // 2, 0, 1 // 2, 0, -1 // 2) ≈ 0.0 atol = 1e-12
        @test cg_completeness_sum(1, 1 // 2, 0, -1 // 2, 0, -1 // 2) ≈ 1.0

        # j1=1, j2=1
        @test cg_completeness_sum(1, 1, 0, 0, 0, 0) ≈ 1.0
        @test cg_completeness_sum(1, 1, 1, -1, 1, -1) ≈ 1.0
        @test cg_completeness_sum(1, 1, 1, -1, 0, 0) ≈ 0.0 atol = 1e-12
        @test cg_completeness_sum(1, 1, 1, 0, -1, 1) ≈ 0.0 atol = 1e-12

        # j1=3/2, j2=1/2
        @test cg_completeness_sum(3 // 2, 1 // 2, 1 // 2, 1 // 2, 1 // 2, 1 // 2) ≈ 1.0
        @test cg_completeness_sum(3 // 2, 1 // 2, 1 // 2, 1 // 2, -1 // 2, -1 // 2) ≈ 0.0 atol =
            1e-12
    end

    # =========================================================================================
    # (d) wigner3j vs independently recomputed table values
    # =========================================================================================
    @testset "3j vs Edmonds/Varshalovich table values" begin
        @test wigner3j(1, 1, 0, 0, 0, 0) ≈ -1 / sqrt(3)
        @test wigner3j(1 // 2, 1 // 2, 0, 1 // 2, -1 // 2, 0) ≈ 1 / sqrt(2)
        @test wigner3j(1 // 2, 1 // 2, 1, 1 // 2, -1 // 2, 0) ≈ 1 / sqrt(6)
        @test wigner3j(1, 1, 2, 0, 0, 0) ≈ sqrt(2 / 15)
        @test wigner3j(1, 1, 2, 1, -1, 0) ≈ sqrt(1 / 30)
        @test wigner3j(1, 1, 2, 1, 1, -2) ≈ 1 / sqrt(5)
        @test wigner3j(2, 2, 2, 0, 0, 0) ≈ -sqrt(2 / 35)
        @test wigner3j(1, 1, 1, 0, 0, 0) == 0.0   # j1+j2+j3 odd ⇒ parity zero

        # a large-spin value away from the tiny-spin corner (regression against a truncated
        # Racah-sum range, same spirit as the existing wigner3j(16,16,0,...) check)
        @test wigner3j(6, 6, 6, 0, 0, 0) ≈ -0.09305950021129074 atol = 1e-10
    end

    # =========================================================================================
    # (e) 3j permutation symmetry (Edmonds §3.3 / Racah): even permutations of the three
    #     columns leave the 3j symbol unchanged; odd permutations multiply by (-1)^(j1+j2+j3);
    #     reversing the sign of all three m's multiplies by (-1)^(j1+j2+j3).
    # =========================================================================================
    @testset "3j permutation and sign symmetries" begin
        cases = [
            (1 // 2, 1 // 2, 1, 1 // 2, -1 // 2, 0),
            (1, 1, 2, 1, -1, 0),
            (1, 1, 2, 0, 0, 0),
            (3 // 2, 1, 1 // 2, 1 // 2, -1, 1 // 2),
            (2, 2, 2, 1, -1, 0),
        ]
        for (j1, j2, j3, m1, m2, m3) in cases
            base = wigner3j(j1, j2, j3, m1, m2, m3)
            phase = (-1)^Int(j1 + j2 + j3)

            # even permutation (cyclic: 123 → 231)
            @test wigner3j(j2, j3, j1, m2, m3, m1) ≈ base atol = 1e-12
            @test wigner3j(j3, j1, j2, m3, m1, m2) ≈ base atol = 1e-12

            # odd permutation (transposition: swap columns 1 and 2)
            @test wigner3j(j2, j1, j3, m2, m1, m3) ≈ phase * base atol = 1e-12
            # transposition: swap columns 2 and 3
            @test wigner3j(j1, j3, j2, m1, m3, m2) ≈ phase * base atol = 1e-12

            # m → -m (all three magnetic quantum numbers negated)
            @test wigner3j(j1, j2, j3, -m1, -m2, -m3) ≈ phase * base atol = 1e-12
        end
    end

    # =========================================================================================
    # (f) wigner6j vs independently recomputed table values, incl. a large-spin regression
    # =========================================================================================
    @testset "6j vs Edmonds/Varshalovich table values" begin
        @test wigner6j(1 // 2, 1 // 2, 1, 1 // 2, 1 // 2, 0) ≈ 1 / 2
        @test wigner6j(1 // 2, 1 // 2, 0, 1 // 2, 1 // 2, 1) ≈ 1 / 2
        @test wigner6j(1, 1, 1, 1, 1, 1) ≈ 1 / 6
        @test wigner6j(1, 1, 0, 1, 1, 1) ≈ -1 / 3
        @test wigner6j(1, 1, 2, 1, 1, 1) ≈ 1 / 6
        @test wigner6j(1, 1, 2, 1, 1, 2) ≈ 1 / 30

        # large-spin regression (independent from the existing {8 8 8;8 8 8} check in
        # test_su2_coeffs.jl — a different spin combination, still deep in the Racah-sum range)
        @test wigner6j(6, 6, 6, 6, 6, 6) ≈ 0.028432094221567904 atol = 1e-10
    end

    # =========================================================================================
    # (g) 6j symmetry: invariant under any column permutation, and under swapping the upper
    #     and lower arguments in any two columns simultaneously (Edmonds §6.2/Regge symmetry
    #     subset used here is the classical tetrahedral symmetry group of order 24).
    # =========================================================================================
    @testset "6j column-permutation and paired upper/lower symmetries" begin
        cases = [
            (1 // 2, 1 // 2, 1, 1 // 2, 1 // 2, 1),
            (1, 1, 1, 1, 1, 1),
            (1, 1, 2, 1, 1, 1),
            (3 // 2, 3 // 2, 1, 3 // 2, 3 // 2, 2),
            (2, 2, 2, 2, 2, 2),
        ]
        for (a, b, c, d, e, f) in cases
            base = wigner6j(a, b, c, d, e, f)

            # column permutations: {a b c; d e f} is invariant under permuting the three columns
            # (a,d), (b,e), (c,f) as whole units
            @test wigner6j(b, a, c, e, d, f) ≈ base atol = 1e-12   # swap columns 1,2
            @test wigner6j(a, c, b, d, f, e) ≈ base atol = 1e-12   # swap columns 2,3
            @test wigner6j(c, b, a, f, e, d) ≈ base atol = 1e-12   # swap columns 1,3
            @test wigner6j(b, c, a, e, f, d) ≈ base atol = 1e-12   # cyclic permutation

            # simultaneously exchange upper and lower arguments in any two columns
            @test wigner6j(d, e, c, a, b, f) ≈ base atol = 1e-12   # exchange columns 1,2
            @test wigner6j(a, e, f, d, b, c) ≈ base atol = 1e-12   # exchange columns 2,3
            @test wigner6j(d, b, f, a, e, c) ≈ base atol = 1e-12   # exchange columns 1,3
        end
    end

    # =========================================================================================
    # (h) 3j ↔ CG relation:
    #     CG(j1,m1;j2,m2|J,M) = (-1)^(j1-j2+M)·√(2J+1)·3j(j1,j2,J,m1,m2,-M)
    # This is the defining relation in src/su2.jl itself, but it is asserted here as an
    # ALGEBRAIC LAW that must hold for every (j1,m1,j2,m2,J,M), independent of and in addition
    # to the direct table-value checks above — a broken 3j sign or CG prefactor would violate
    # this identity even for cases the direct spot-checks above happen not to cover.
    # =========================================================================================
    @testset "3j ↔ CG relation" begin
        cases = [
            (1 // 2, 1 // 2, 1 // 2, -1 // 2, 0, 0),
            (1 // 2, 1 // 2, 1 // 2, 1 // 2, 1, 1),
            (1, 0, 1, 0, 2, 0),
            (1, 1, 1, -1, 2, 0),
            (1, 1, 1, -1, 1, 0),
            (1, 1, 1, -1, 0, 0),
            (1, 1, 1 // 2, -1 // 2, 3 // 2, 1 // 2),
            (3 // 2, 3 // 2, 1 // 2, -1 // 2, 2, 1),
            (3 // 2, 1 // 2, 1 // 2, -1 // 2, 1, 0),
            (2, 0, 1, 0, 3, 0),
            (2, 1, 1, -1, 3, 0),
            (2, -1, 1, 1, 1, 0),
        ]
        for (j1, m1, j2, m2, J, M) in cases
            lhs = clebsch_gordan(j1, m1, j2, m2, J, M)
            rhs = (-1)^Int(j1 - j2 + M) * sqrt(2J + 1) * wigner3j(j1, j2, J, m1, m2, -M)
            @test lhs ≈ rhs atol = 1e-12
        end
    end
end
