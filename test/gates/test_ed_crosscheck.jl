# ED cross-check gate — the "trust this reference" anchor. For a SMALL Wilson chain the full many-body
# Hamiltonian (`wilson_chain_hamiltonian`, built INDEPENDENTLY of the engine via Jordan–Wigner) is
# EXACTLY diagonalized. The interacting impurity spectral function then has EXACT integrated weight
# across the Fermi level — `∫_{ω<0}A = ⟨n_↑⟩`, `∫A = 1` (Lehmann sum rules) — which the NRG CFS assembly
# must reproduce. Two subtleties, both handled: the ground state is a SPIN DOUBLET, so `⟨n_↑⟩` is traced
# over the GS multiplet (exactly what NRG's ground reduced density matrix does); and the spectral weight
# must be integrated on a WIDE grid (the default clips the band tails). Broadening-robust (integrated
# weights), no energy rescaling. Mirrors ComplexTimeSIAM's own `ed_reference.jl`. Issue #58.

using WilsonNRG, Test
using LinearAlgebra: eigen, Symmetric, dot

# exact ground-multiplet-averaged impurity up-occupation ⟨n_↑⟩ of the full (small) chain
function _ed_nup(m, alg)
    L = alg.nsites + 1
    F = eigen(Symmetric(wilson_chain_hamiltonian(m, alg)))
    dup = WilsonNRG._jw(WilsonNRG._CDU, 0, L)              # impurity d†_↑ (leftmost orbital)
    nop = dup * adjoint(dup)                               # d†_↑ d_↑ = n_↑
    E0 = F.values[1]
    gs = findall(e -> e - E0 < 1.0e-6, F.values)           # the degenerate ground multiplet
    return sum(dot(F.vectors[:, d], nop * F.vectors[:, d]) for d in gs) / length(gs)
end

# NRG CFS spectral function: (weight below E_F, total weight) on a wide grid
function _nrg_below(m, alg)
    ωg = collect(-3.0:0.005:3.0)
    A = (-1 / π) .* imag.(green_function(CFS(), m, alg; ω=ωg).G)
    below = sum(A[k] * (ωg[k + 1] - ωg[k]) for k in 1:(length(ωg) - 1) if ωg[k] < 0)
    tot = sum(A[k] * (ωg[k + 1] - ωg[k]) for k in 1:(length(ωg) - 1))
    return below, tot
end

@testset "ED cross-check · NRG CFS spectral weight = exact diagonalization (issue #58)" begin
    U = 0.5
    Γ = 0.1
    D = 1.0
    ns = 4                                                  # dim 4^5 = 1024, exactly diagonalizable
    keepall = NRGAlgorithm(;
        discretization=WilsonLog(2.0), symmetry=U1U1(), truncation=KeepN(1024), nsites=ns
    )

    # keep-all ⇒ the CFS complete-basis assembly must be the EXACT chain spectrum:
    #   ∫A = 1 (sum rule)   and   ∫_{ω<0}A = ⟨n_↑⟩ (the exact below-Fermi weight).
    @testset "∫A=1 and ∫_{ω<0}A=⟨n_↑⟩ · εd=$εd" for εd in (-0.25, -0.35, -0.15)
        m = AndersonModel(; U, εd, Γ, D)
        nup = _ed_nup(m, keepall)
        below, tot = _nrg_below(m, keepall)
        @test isapprox(tot, 1.0; atol=0.02)                # complete-basis sum rule
        @test isapprox(below, nup; atol=0.02)              # NRG reproduces the EXACT occupation ⟨n_↑⟩
    end

    # the symmetric point is exactly half-filled per spin (validates the GS-multiplet trace + p–h symmetry)
    @test isapprox(
        _ed_nup(AndersonModel(; U, εd=(-U / 2), Γ, D), keepall), 0.5; atol=1.0e-6
    )
end
