# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Akbar Arif
import numpy as np
import pytest

from ie.speculative import (
    accept_probability,
    expected_tokens,
    residual_distribution,
    simulate_tokens_per_pass,
    speculative_step,
    speedup,
    verify_block,
)

P = np.array([0.45, 0.22, 0.15, 0.08, 0.10])
Q = np.array([0.10, 0.08, 0.12, 0.30, 0.40])


def _empirical(draw, size, n):
    counts = np.zeros(size)
    for _ in range(n):
        counts[draw()] += 1
    return counts / n


def test_accept_probability():
    assert accept_probability(0.72, 0.65) == 1.0  # target likes it at least as much
    assert accept_probability(0.08, 0.30) == pytest.approx(0.2667, abs=1e-4)


def test_residual_distribution_matches_the_worked_example():
    res, z = residual_distribution(P, Q)
    assert z == pytest.approx(0.52)
    np.testing.assert_allclose(res, [0.673, 0.269, 0.058, 0.0, 0.0], atol=5e-4)


def test_rejection_probability_equals_the_residual_normaliser():
    assert 1.0 - np.minimum(P, Q).sum() == pytest.approx(residual_distribution(P, Q)[1])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_rule_emits_exactly_the_target_distribution(seed):
    rng = np.random.default_rng(seed)
    p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
    emp = _empirical(lambda: speculative_step(p, q, rng)[0], 5, 40_000)
    assert np.max(np.abs(emp - p)) < 0.015


def test_a_terrible_draft_costs_acceptance_not_correctness():
    rng = np.random.default_rng(3)
    emp = _empirical(lambda: speculative_step(P, Q, rng)[0], 5, 40_000)
    assert np.max(np.abs(emp - P)) < 0.015
    accepted = np.mean([speculative_step(P, Q, rng)[1] for _ in range(20_000)])
    assert accepted == pytest.approx(np.minimum(P, Q).sum(), abs=0.02)  # 0.48


def test_first_token_of_a_verified_block_follows_the_target():
    rng = np.random.default_rng(4)
    k, v = 3, 4
    p_rows = rng.dirichlet(np.ones(v), size=k + 1)
    q_rows = rng.dirichlet(np.ones(v), size=k)

    def draw():
        drafts = [int(rng.choice(v, p=q_rows[i])) for i in range(k)]
        return verify_block(p_rows, q_rows, drafts, rng)[0]

    assert np.max(np.abs(_empirical(draw, v, 30_000) - p_rows[0])) < 0.015


def test_a_perfect_draft_is_always_fully_accepted_with_a_bonus_token():
    rng = np.random.default_rng(5)
    rows = np.tile(np.array([0.5, 0.3, 0.2]), (5, 1))
    for _ in range(50):
        drafts = [int(rng.choice(3, p=rows[0])) for _ in range(4)]
        assert len(verify_block(rows, rows[:4], drafts, rng)) == 5


def test_a_rejection_ends_the_block_early():
    rng = np.random.default_rng(6)
    p_rows = np.tile(np.array([1.0, 0.0]), (4, 1))
    q_rows = np.tile(np.array([0.0, 1.0]), (3, 1))  # draft always proposes token 1, target never accepts it
    out = verify_block(p_rows, q_rows, [1, 1, 1], rng)
    assert out == [0]


def test_expected_tokens_formula():
    assert expected_tokens(0.7, 4) == pytest.approx(2.7731, abs=1e-4)
    assert expected_tokens(1.0, 6) == 7
    assert expected_tokens(0.0, 6) == 1


def test_simulation_agrees_with_the_formula():
    rng = np.random.default_rng(7)
    assert simulate_tokens_per_pass(0.7, 4, 60_000, rng) == pytest.approx(expected_tokens(0.7, 4), abs=0.03)


def test_free_drafts_give_the_expected_token_count_as_speedup():
    assert speedup(0.7, 4, 0.0) == pytest.approx(expected_tokens(0.7, 4))


def test_an_expensive_or_poor_draft_can_be_slower_than_plain_decoding():
    assert speedup(0.3, 8, 0.3) < 1.0
