# Faithfulness gate — SU(2) angular-momentum coefficients (foundation for U1SU2).
# These are independently verifiable against textbook values (Edmonds), so they are checked
# directly here; the non-abelian engine's 6j recoupling will instead be pinned against the
# U1U1 spectrum (cross-method agreement) once it lands.

using WilsonNRG, Test
using WilsonNRG: clebsch_gordan, wigner3j, wigner6j

@testset "method-recovery gate · SU(2) CG/6j coefficients" begin
    # ---- Clebsch–Gordan vs known values ----
    @test clebsch_gordan(1 // 2, 1 // 2, 1 // 2, -1 // 2, 0, 0) ≈ 1 / sqrt(2)   # singlet
    @test clebsch_gordan(1 // 2, 1 // 2, 1 // 2, 1 // 2, 1, 1) ≈ 1.0            # stretched
    @test clebsch_gordan(1, 0, 1 // 2, 1 // 2, 1 // 2, 1 // 2) ≈ -1 / sqrt(3)
    @test clebsch_gordan(1 // 2, 1 // 2, 1, 0, 3 // 2, 1 // 2) ≈ sqrt(2 / 3)

    # orthonormality: Σ_{m1} ⟨j1 m1; j2 M-m1|J M⟩² = 1 for each (J,M) in the decomposition
    s = sum(clebsch_gordan(1, m, 1 // 2, 1 // 2 - m, 3 // 2, 1 // 2)^2 for m in (-1, 0, 1))
    @test s ≈ 1.0

    # ---- 6j vs known values ----
    @test wigner6j(1 // 2, 1 // 2, 1, 1 // 2, 1 // 2, 1) ≈ 1 / 6
    @test wigner6j(1, 1, 1, 1, 1, 1) ≈ 1 / 6
    @test wigner6j(1 // 2, 1 // 2, 0, 1 // 2, 1 // 2, 1) ≈ 1 / 2
    @test wigner6j(1, 1, 0, 1, 1, 1) ≈ -1 / 3                                   # {1 1 0;1 1 1}

    # large spins exercise the full Racah summation range — regression for the
    # analytic t-bounds (a fixed 0:30 cap silently truncated these terms).
    @test wigner6j(8, 8, 8, 8, 8, 8) ≈ -0.012652080723153538 atol = 1e-10       # t ∈ 24:32
    @test wigner3j(16, 16, 0, -16, 16, 0) ≈ 1 / sqrt(33) atol = 1e-10           # single term at t=32

    # ---- 3j directly (independent of the clebsch_gordan wrapper) ----
    @test wigner3j(1 // 2, 1 // 2, 0, 1 // 2, -1 // 2, 0) ≈ 1 / sqrt(2)
    @test wigner3j(1 // 2, 1 // 2, 1, 1 // 2, -1 // 2, 0) ≈ 1 / sqrt(6)
    @test wigner3j(1, 1, 1, 0, 0, 0) == 0.0                                      # parity zero (j-sum odd)
    @test wigner3j(1 // 2, 1 // 2, 1, 1 // 2, -1 // 2, 0) ≈
          wigner3j(1 // 2, 1, 1 // 2, -1 // 2, 0, 1 // 2)                        # cyclic invariance

    # triangle violations vanish
    @test clebsch_gordan(1 // 2, 1 // 2, 1 // 2, 1 // 2, 0, 0) == 0.0           # |1/2-1/2|..=0 only
    @test wigner6j(1 // 2, 1 // 2, 5, 1 // 2, 1 // 2, 1) == 0.0

    # ---- multiplet weighting: 2S+1 ----
    @test WilsonNRG.multiplicity(U1SU2(), (1, 1 // 2)) == 2
    @test WilsonNRG.multiplicity(U1SU2(), (0, 0 // 1)) == 1
    @test WilsonNRG.multiplicity(U1SU2(), (1, 3 // 2)) == 4
    @test WilsonNRG.multiplicity(U1SU2(), (2, 1 // 1)) == 3                      # integer S=1

    # ---- reduced-ME (Wigner-Eckart) layer: f† reduced MEs reproduce the full single-site c† ----
    @testset "reduced f† reproduces full c† (Wigner-Eckart)" begin
        R = WilsonNRG._ELECTRON_REDUCED_FDAG
        # ⟨S′Sz′|c†_σ|S Sz⟩ = CG(S Sz; ½ σ | S′ Sz′)·R[(Q,S)]; check all four nonzero c† elements
        cases = [  # (S, Sz, σ, S′, Sz′, Q_src, full c† element)
            (0 // 1, 0 // 1, 1 // 2, 1 // 2, 1 // 2, 0, 1.0),    # c†↑|0⟩=|↑⟩
            (0 // 1, 0 // 1, -1 // 2, 1 // 2, -1 // 2, 0, 1.0),  # c†↓|0⟩=|↓⟩
            (1 // 2, -1 // 2, 1 // 2, 0 // 1, 0 // 1, 1, 1.0),   # c†↑|↓⟩=|↑↓⟩
            (1 // 2, 1 // 2, -1 // 2, 0 // 1, 0 // 1, 1, -1.0),  # c†↓|↑⟩=−|↑↓⟩
        ]
        for (S, Sz, σ, S′, Sz′, Q, full) in cases
            @test clebsch_gordan(S, Sz, 1 // 2, σ, S′, Sz′) * R[(Q, S)] ≈ full atol = 1e-12
        end
    end
end
