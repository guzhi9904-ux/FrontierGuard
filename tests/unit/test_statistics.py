from frontierguard.evaluation.statistics import PairedObservation, paired_problem_bootstrap


def test_bootstrap_groups_repeated_seeds_by_problem():
    observations = [
        PairedObservation("p1", 0, 1),
        PairedObservation("p1", 1, 1),
        PairedObservation("p2", 0, 0),
        PairedObservation("p2", 0, 1),
    ]
    result = paired_problem_bootstrap(observations, samples=1000, seed=2)
    assert result.problems == 2
    assert result.estimate == 0.5
    assert result.lower <= result.estimate <= result.upper
