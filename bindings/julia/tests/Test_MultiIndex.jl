using Test
using .MParT

function test_sum()
    idx = MParT.MultiIndex(30,2)
    @test sum(idx) == 60
    idx[1] = 3
    @test sum(idx) == 61
end

function test_max()
    idx = MParT.MultiIndex(30,2)
    @test maximum(idx) == 2
    idx[1] = 100
    @test maximum(idx) == 100
end

function test_count_nonzero()
    idx = MParT.MultiIndex(30)
    @test count_nonzero(idx) == 0
    idx[4] = 1
    idx[27] = 3
    @test count_nonzero(idx) == 2
end

@testset verbose=true "MParT" begin
    @testset verbose=true "MultiIndex" begin
        test_sum()
        test_max()
        test_count_nonzero()
    end
end