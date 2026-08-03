ENV["GKSwstype"] = "100"

using WilsonNRG
using Test, Aqua
using TestShards

# Every `test_*.jl` under `test/`, in a deterministic order, each one its own shardable unit.
# `@shard` shadows `include` inside the block, so a unit is whatever this loop includes — a new
# file, or a whole new directory, is picked up BY BEING ON DISK. That is what the old
# `test/ci/universe.jl` completeness guard existed to enforce by hand, and it could only ever
# ERROR, because the wiring lived in a second list that could disagree with the tree.
#
# Two rules when adding to this, and they are the only two:
#
#   1. SHARED FIXTURES GO ABOVE THIS BLOCK. A helper included inside becomes a unit of its own,
#      lands on ONE shard, and every test file on the other shards that needed it fails.
#   2. ANYTHING THAT IS NOT A `test_*.jl` FILE MUST BE NAMED, as Aqua is below. The glob does not
#      error on what it does not match; it silently stops running it.
#
# A bare `Pkg.test()` with nothing set in the environment runs all of it, in this order. Run one
# shard locally with `TESTSHARDS_ID=s3 TESTSHARDS_N=16 julia --project -e 'using Pkg; Pkg.test()'`.
TestShards.@shard begin
    for (dir, _, files) in sort!(collect(walkdir(@__DIR__)); by=first)
        for f in sort(files)
            startswith(f, "test_") && endswith(f, ".jl") || continue
            include(joinpath(dir, f))
        end
    end
    # Whole-package QA. It is not a file, so `@unit` gives it a key of its own and it becomes an
    # ordinary shardable unit — it used to be pinned to whichever shard carried the `aqua` flag.
    TestShards.@unit "aqua" begin
        Aqua.test_all(WilsonNRG)
    end
end
