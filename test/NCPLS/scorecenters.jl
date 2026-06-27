@testset "scorecenters from score matrix" begin
    scores = [
        1.0 10.0 100.0
        3.0 14.0 140.0
        9.0 18.0 180.0
        11.0 22.0 220.0
        13.0 26.0 260.0
    ]
    classes = ["A", "A", "B", "B", "B"]
    labels = ["s1", "s2", "s3", "s4", "s5"]

    sc = NCPLS.scorecenters(scores, classes; samplelabels = labels, comps = 1:2)

    @test sc isa NCPLS.ScoreCenters
    @test sc.classes == ["A", "B"]
    @test sc.comps == [1, 2]
    @test sc.center_method == :median
    @test NCPLS.sampleindices(sc, "A") == [1, 2]
    @test NCPLS.sampleindices(sc, "B") == [3, 4, 5]
    @test NCPLS.samplelabels(sc, "A") == ["s1", "s2"]
    @test NCPLS.scorecenter(sc, "A") == [2.0, 12.0]
    @test NCPLS.scorecenter(sc, "B") == [11.0, 22.0]
    @test_throws KeyError NCPLS.scorecenter(sc, "C")

    @test sprint(show, sc) ==
        "ScoreCenters(classes=2, components=[1, 2], center=:median)"
    @test sprint(show, MIME"text/plain"(), sc) ==
        "ScoreCenters\n" *
        "  classes: [\"A\", \"B\"]\n" *
        "  components: [1, 2]\n" *
        "  center: median"
end

@testset "scorecenters supports mean and selected components" begin
    scores = [
        1.0 10.0 100.0
        3.0 14.0 140.0
        9.0 18.0 180.0
        11.0 22.0 220.0
    ]
    classes = [:A, :A, :B, :B]

    sc = NCPLS.scorecenters(scores, classes; comps = [3, 1], center = :mean)

    @test sc.classes == [:A, :B]
    @test sc.comps == [3, 1]
    @test NCPLS.scorecenter(sc, :A) == [120.0, 2.0]
    @test NCPLS.scorecenter(sc, "B") == [200.0, 10.0]
end

@testset "scorecenters accepts projected scores and subset components" begin
    projected_scores = [
        1.0 100.0 10.0
        3.0 300.0 14.0
        9.0 900.0 18.0
        11.0 1100.0 22.0
    ]
    classes = ["A", "A", "B", "B"]
    labels = ["new1", "new2", "new3", "new4"]

    sc = NCPLS.scorecenters(projected_scores, classes;
        samplelabels = labels,
        comps = [1, 3],
    )

    @test sc.comps == [1, 3]
    @test NCPLS.samplelabels(sc, "A") == ["new1", "new2"]
    @test NCPLS.scorecenter(sc, "A") == [2.0, 12.0]
    @test NCPLS.scorecenter(sc, "B") == [10.0, 20.0]
end

@testset "scorecenters from NCPLSFit" begin
    model = NCPLS.NCPLSModel(ncomponents = 2, multilinear = false)
    mf = NCPLS.NCPLSFit(
        model,
        reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1),
        [1.0 2.0; 3.0 4.0],
        [1.0 0.0; 0.0 1.0; 3.0 2.0; 5.0 4.0],
        [2.0 1.0; 0.0 1.0],
        reshape([5.0, 6.0], 1, 2),
        [1.0 0.0; 0.0 1.0],
        nothing,
        reshape([1.0, 2.0], 1, 2),
        reshape(collect(1.0:4.0), 2, 1, 2),
        [0.5, 0.7],
        reshape([1.0, 2.0, 3.0, 4.0], :, 1),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        [1.0, 2.0],
        [2.0, 4.0],
        [10.0];
        samplelabels = ["s1", "s2", "s3", "s4"],
        sampleclasses = ["A", "A", "B", "B"],
    )

    sc = NCPLS.scorecenters(mf; comps = 2, center = :mean)

    @test sc.classes == ["A", "B"]
    @test sc.comps == [2]
    @test NCPLS.sampleindices(sc, "A") == [1, 2]
    @test NCPLS.samplelabels(sc, "B") == ["s3", "s4"]
    @test NCPLS.scorecenter(sc, "A") == [0.5]
    @test NCPLS.scorecenter(sc, "B") == [3.0]
end

@testset "scorecenters validates inputs" begin
    scores = [1.0 2.0; 3.0 4.0]

    @test_throws ArgumentError NCPLS.scorecenters(scores, ["A"])
    @test_throws ArgumentError NCPLS.scorecenters(scores, ["A", "B"]; samplelabels = ["s1"])
    @test_throws ArgumentError NCPLS.scorecenters(scores, ["A", "B"]; comps = Int[])
    @test_throws ArgumentError NCPLS.scorecenters(scores, ["A", "B"]; comps = 3)
    @test_throws ArgumentError NCPLS.scorecenters(scores, ["A", "B"]; center = :mode)
    @test_throws ArgumentError NCPLS.scorecenter_values(scores, :mode)

    model = NCPLS.NCPLSModel(ncomponents = 2, multilinear = false)
    mf = NCPLS.NCPLSFit(
        model,
        reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1),
        [1.0 2.0; 3.0 4.0],
        [1.0 0.0; 0.0 1.0],
        [2.0 1.0; 0.0 1.0],
        reshape([5.0, 6.0], 1, 2),
        [1.0 0.0; 0.0 1.0],
        nothing,
        reshape([1.0, 2.0], 1, 2),
        reshape(collect(1.0:4.0), 2, 1, 2),
        [0.5, 0.7],
        reshape([1.0, 2.0], :, 1),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        [1.0, 2.0],
        [2.0, 4.0],
        [10.0],
    )

    @test_throws ArgumentError NCPLS.scorecenters(mf)
end

@testset "scorerepresentatives from score matrix" begin
    scores = [
        1.0 10.0
        2.0 11.0
        4.0 14.0
        8.0 20.0
        10.0 22.0
        13.0 24.0
    ]
    classes = ["A", "A", "A", "B", "B", "B"]
    labels = ["s1", "s2", "s3", "s4", "s5", "s6"]

    sr = NCPLS.scorerepresentatives(scores, classes;
        samplelabels = labels,
        comps = 1:2,
        n = 2,
    )

    @test sr isa NCPLS.ScoreRepresentatives
    @test sr.classes == ["A", "B"]
    @test sr.comps == [1, 2]
    @test sr.center_method == :median
    @test NCPLS.sampleindices(sr, "A") == [2, 1]
    @test NCPLS.samplelabels(sr, "A") == ["s2", "s1"]
    @test NCPLS.representativescores(sr, "A") == [2.0 11.0; 1.0 10.0]
    @test NCPLS.representativedistances(sr, "A") ≈ [0.0, sqrt(2.0)]
    @test NCPLS.scorecenters(sr) === sr.centers
    @test NCPLS.scorecenter(sr, "B") == [10.0, 22.0]
    @test NCPLS.sampleindices(sr, "B") == [5, 4]
    @test_throws KeyError NCPLS.sampleindices(sr, "C")

    @test sprint(show, sr) ==
        "ScoreRepresentatives(classes=2, representatives=2, components=[1, 2], center=:median)"
    @test sprint(show, MIME"text/plain"(), sr) ==
        "ScoreRepresentatives\n" *
        "  classes: [\"A\", \"B\"]\n" *
        "  representatives per class: [2, 2]\n" *
        "  components: [1, 2]\n" *
        "  center: median"

    sr_empty = NCPLS.ScoreRepresentatives(
        String[],
        Vector{Int}[],
        Vector{String}[],
        Matrix{Float64}[],
        Vector{Float64}[],
        nothing,
        [1],
        :mean,
    )
    @test sprint(show, sr_empty) ==
        "ScoreRepresentatives(classes=0, representatives=0, components=[1], center=:mean)"
end

@testset "scorerepresentatives accepts projected scores and subset components" begin
    projected_scores = [
        1.0 100.0 10.0
        2.0 200.0 11.0
        4.0 400.0 14.0
        8.0 800.0 20.0
        10.0 1000.0 22.0
        13.0 1300.0 24.0
    ]
    classes = ["A", "A", "A", "B", "B", "B"]
    labels = ["new1", "new2", "new3", "new4", "new5", "new6"]

    sr = NCPLS.scorerepresentatives(projected_scores, classes;
        samplelabels = labels,
        comps = [1, 3],
        n = 1,
    )

    @test sr.comps == [1, 3]
    @test NCPLS.sampleindices(sr, "A") == [2]
    @test NCPLS.samplelabels(sr, "A") == ["new2"]
    @test NCPLS.representativescores(sr, "B") == reshape([10.0, 22.0], 1, 2)
    @test NCPLS.representativedistances(sr, "B") == [0.0]
end

@testset "scorerepresentatives from NCPLSFit" begin
    model = NCPLS.NCPLSModel(ncomponents = 2, multilinear = false)
    mf = NCPLS.NCPLSFit(
        model,
        reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1),
        [1.0 2.0; 3.0 4.0],
        [1.0 0.0; 2.0 0.0; 4.0 0.0; 9.0 1.0; 11.0 1.0],
        [2.0 1.0; 0.0 1.0],
        reshape([5.0, 6.0], 1, 2),
        [1.0 0.0; 0.0 1.0],
        nothing,
        reshape([1.0, 2.0], 1, 2),
        reshape(collect(1.0:4.0), 2, 1, 2),
        [0.5, 0.7],
        reshape([1.0, 2.0, 3.0, 4.0, 5.0], :, 1),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        [1.0, 2.0],
        [2.0, 4.0],
        [10.0];
        samplelabels = ["s1", "s2", "s3", "s4", "s5"],
        sampleclasses = ["A", "A", "A", "B", "B"],
    )

    sr = NCPLS.scorerepresentatives(mf; comps = 1, center = :mean)

    @test sr.classes == ["A", "B"]
    @test sr.comps == [1]
    @test NCPLS.sampleindices(sr, "A") == [2]
    @test NCPLS.samplelabels(sr, "B") == ["s4"]
    @test NCPLS.representativescores(sr, "A") == reshape([2.0], 1, 1)
    @test NCPLS.representativedistances(sr, "B") == [1.0]
end

@testset "scorerepresentatives validates inputs" begin
    scores = [1.0 2.0; 3.0 4.0]

    @test_throws ArgumentError NCPLS.scorerepresentatives(scores, ["A", "B"]; n = 0)
    @test_throws ArgumentError NCPLS.scorerepresentatives(scores, ["A"]; n = 1)
    @test_throws ArgumentError NCPLS.scorerepresentatives(scores, ["A", "B"]; center = :mode)

    model = NCPLS.NCPLSModel(ncomponents = 2, multilinear = false)
    mf = NCPLS.NCPLSFit(
        model,
        reshape([1.0, 2.0, 3.0, 4.0], 2, 2, 1),
        [1.0 2.0; 3.0 4.0],
        [1.0 0.0; 0.0 1.0],
        [2.0 1.0; 0.0 1.0],
        reshape([5.0, 6.0], 1, 2),
        [1.0 0.0; 0.0 1.0],
        nothing,
        reshape([1.0, 2.0], 1, 2),
        reshape(collect(1.0:4.0), 2, 1, 2),
        [0.5, 0.7],
        reshape([1.0, 2.0], :, 1),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        [1.0, 2.0],
        [2.0, 4.0],
        [10.0],
    )

    @test_throws ArgumentError NCPLS.scorerepresentatives(mf)
end
