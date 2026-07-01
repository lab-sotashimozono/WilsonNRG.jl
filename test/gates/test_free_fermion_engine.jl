# Faithfulness gate — free-fermion (U=0) reproduction of the U1U1 engine, swept over a GRID.
# Complements test_nrg_engine_u1u1.jl, which checks the exact subset-sum law at a SINGLE
# (Λ, nsites, εd) spot. The free-fermion many-body spectrum = subset-sums of the single-particle
# Wilson chain is a CONVENTION-INDEPENDENT operator identity, so it must hold at EVERY
# discretization Λ, chain length, and bare level εd — not one point. On top of that we add the
# particle–hole-symmetric-chain ZERO-MODE structural law: a real symmetric tridiagonal with zero
# diagonal has eigenvalues in ± pairs plus exactly one zero eigenvalue iff its dimension is odd, so
# the free-fermion ground state is 4-fold degenerate for an odd orbital count and unique for an
# even one — an independent linear-algebra prediction the engine's block degeneracy must match.
#
# Reference = independent diagonalization of the single-particle Wilson-chain matrix (the same √Λ
# recursion the engine uses internally), NOT the engine's own spectrum — so this is a genuine
# cross-check against an external answer, not tautological self-consistency.

using WilsonNRG, Test
using LinearAlgebra: Symmetric, eigvals

# independent single-particle Wilson-chain spectrum (mirrors the engine's √Λ recursion). Unique
# helper names so this file coexists with test_nrg_engine_u1u1.jl's `_single_particle` when both
# are included into the same shard session.
function _ffe_levels(model, chain, nsites)
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
    return eigvals(Symmetric(m))
end
function _ffe_subset_sums(levels)
    s = [0.0]
    for λ in levels
        s = vcat(s, s .+ λ)
    end
    return sort(s)
end

@testset "faithfulness gate · free-fermion engine reproduction (U1U1, grid)" begin
    # ---- (A) exact subset-sum law across a (Λ, nsites, εd) grid --------------------------------
    # Every combination is an independent free-fermion diagonalization; the engine's full keep-all
    # many-body spectrum must equal the ↑⊕↓ subset-sums to machine precision (relative to ground,
    # since the engine ground-subtracts each iteration).
    @testset "U=0 subset-sums · Λ=$Λ nsites=$nsites εd=$εd" for Λ in (2.0, 3.0),
        nsites in (1, 3, 4),
        εd in (0.0, 0.15)

        model = AndersonModel(; U=0.0, εd, Γ=0.05, D=1.0)
        alg = NRGAlgorithm(;
            discretization=WilsonLog(Λ), symmetry=U1U1(), truncation=KeepN(10^9), nsites
        )
        got = sort(nrg_solve(model, alg).energies[end])
        sp = _ffe_levels(model, wilson_chain(WilsonLog(Λ), model, nsites), nsites)
        ref = _ffe_subset_sums(vcat(sp, sp))                    # spin ↑ and ↓ channels
        @test length(got) == length(ref) == 4^(nsites + 1)
        @test maximum(abs, (got .- got[1]) .- (ref .- ref[1])) < 1e-9
    end

    # ---- (A') one larger chain (4^7 = 16384 states, still keep-all ⇒ exact) --------------------
    @testset "U=0 subset-sums · large chain nsites=6" begin
        model = AndersonModel(; U=0.0, εd=0.0, Γ=0.05, D=1.0)
        alg = NRGAlgorithm(;
            discretization=WilsonLog(2.5), symmetry=U1U1(), truncation=KeepN(10^9), nsites=6
        )
        got = sort(nrg_solve(model, alg).energies[end])
        sp = _ffe_levels(model, wilson_chain(WilsonLog(2.5), model, 6), 6)
        ref = _ffe_subset_sums(vcat(sp, sp))
        @test length(got) == length(ref) == 16384
        @test maximum(abs, (got .- got[1]) .- (ref .- ref[1])) < 1e-9
    end

    # ---- (B) particle–hole-symmetric chain: zero-mode ⇒ ground degeneracy ----------------------
    # εd=0 with a flat p-h-symmetric band ⇒ the single-particle tridiagonal has an all-ZERO diagonal;
    # its eigenvalues come in ±pairs with exactly one zero eigenvalue iff the dimension N=nsites+1 is
    # ODD. A zero single-particle level is a spinful Fermi-level mode → 4 degenerate fillings
    # (0,↑,↓,↑↓) at zero energy → a 4-fold many-body ground state for odd N, unique for even N. The
    # engine's ground-block degeneracy must reproduce this, independent of any spectral convention.
    @testset "zero-mode ground degeneracy · nsites=$nsites" for nsites in (2, 3, 4, 5)
        model = AndersonModel(; U=0.0, εd=0.0, Γ=0.05, D=1.0)
        chain = wilson_chain(WilsonLog(2.5), model, nsites)
        @test all(x -> abs(x) < 1e-12, chain.onsite)            # symmetric band ⇒ zero onsite energies
        sp = _ffe_levels(model, chain, nsites)
        nzero = count(x -> abs(x) < 1e-9, sp)
        @test nzero == (nsites + 1) % 2                         # one zero mode iff N=nsites+1 is odd
        got = sort(
            nrg_solve(
                model,
                NRGAlgorithm(;
                    discretization=WilsonLog(2.5),
                    symmetry=U1U1(),
                    truncation=KeepN(10 ^ 9),
                    nsites,
                ),
            ).energies[end],
        )
        deg = count(x -> abs(x - got[1]) < 1e-9, got)
        @test deg == (isodd(nsites + 1) ? 4 : 1)               # 4^(#zero modes), #zero ∈ {0,1}
    end
end
