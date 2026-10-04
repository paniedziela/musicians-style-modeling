import numpy as np
import pytest

from musicians_style.e3.objective import grouped_distance
from musicians_style.e3.profile import style_vector
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent
from musicians_style.style_metrics import (assert_disjoint, cosine, event_profiles, fit_measures,
                                           rank_credit, score_measures)


def piece(pitch=60, shift=0):
    return InternalRepr(480, tuple(NoteEvent(t + shift, 0, p + pitch, 80, d) for t, p, d in
                                    [(0, 0, 120), (0, 7, 240), (240, 2, 480), (600, 5, 1440)]))


@pytest.mark.parametrize("field", ["sample_id", "group_id", "sha256"])
def test_fit_rejects_forbidden_overlap_before_data_access(field):
    train = dict(sample_id="a", group_id="ga", sha256="ha", composer="A")
    forbidden = dict(sample_id="b", group_id="gb", sha256="hb", composer="B")
    forbidden[field] = train[field]
    with pytest.raises(ValueError, match=field):
        fit_measures([train], [forbidden], {}, {}, {})


def test_identity_provenance_required():
    with pytest.raises(ValueError, match="missing"):
        assert_disjoint([dict(sample_id="a", group_id="", sha256="x", composer="A")], [])


def test_event_profiles_are_separate_and_time_pitch_is_transposition_invariant():
    a, diagnostics = event_profiles(piece())
    b, _ = event_profiles(piece(72))
    assert a["onset_duration"].shape == (288,)
    assert a["time_pitch"].shape == (984,)
    assert a["onset_duration"].sum() == pytest.approx(1)
    assert a["time_pitch"].sum() == pytest.approx(1)
    assert np.array_equal(a["time_pitch"], b["time_pitch"])
    assert diagnostics["duration_ge_2"] == 1
    assert cosine(a["onset_duration"], a["onset_duration"]) == pytest.approx(1)
    assert cosine(np.zeros(288), a["onset_duration"]) is None


def test_pair_edges_overflow_and_simultaneous_orientation():
    p = InternalRepr(480, (NoteEvent(0, 0, 60, 80, 0), NoteEvent(0, 0, 67, 80, 960),
                           NoteEvent(1919, 0, 80, 80, 960), NoteEvent(1920, 0, 101, 80, 1)))
    events, d = event_profiles(p)
    histogram = events["time_pitch"].reshape(24, 41)
    assert histogram[0, 20 - 7] > 0 and histogram[0, 20 + 7] > 0
    assert histogram[23, 20 + 20] > 0
    assert d["time_pitch_excluded_pitch_pairs"] > 0
    assert d["duration_nonpositive"] == 1
    assert events["onset_duration"].reshape(24, 12)[0, 11] > 0


def test_empty_and_irregular_meter_are_explicit():
    events, d = event_profiles(InternalRepr(480, meta=(MetaEvent(0, "time_signature", {"numerator": 3, "denominator": 4}),)))
    assert d["non_4_4"] and len(d["empty_profiles"]) == 2
    assert cosine(events["time_pitch"], events["time_pitch"]) is None


def test_ranking_ties_and_undefined_cases():
    assert rank_credit({"A": 1., "B": 1., "C": 0.}, "A") == (0.5, ["A", "B"])
    assert rank_credit({"A": None, "B": 1.}, "A") == (None, [])
    assert rank_credit({"A": 1., "B": 2.}, "A") == (0., ["B"])


def test_train_only_gaussian_floor_rms_exact_and_fixed_logistic_determinism():
    rng = np.random.default_rng(4)
    train = [dict(sample_id=f"{c}{i}", group_id=f"{c}{i}", sha256=f"h{c}{i}", composer=c)
             for c in ("A", "B", "C") for i in range(3)]
    forbidden = [dict(sample_id="test", group_id="test", sha256="test", composer="A")]
    reprs = {r["sample_id"]: piece(50 + i, shift=i * 17) for i, r in enumerate(train)}
    custom = {r["sample_id"]: rng.normal(size=93) for r in train}
    events = {s: event_profiles(p)[0] for s, p in reprs.items()}
    first, metadata = fit_measures(train, forbidden, reprs, custom, events)
    # Poison held-out arrays: none can affect fitted parameters or prototypes.
    reprs["test"], custom["test"], events["test"] = piece(120), np.full(93, 1e9), event_profiles(piece(120))[0]
    second, metadata2 = fit_measures(train, forbidden, reprs, custom, events)
    assert metadata == metadata2
    pooled = np.asarray([style_vector(reprs[r["sample_id"]])[0] for r in train]).std(axis=0)
    assert np.array_equal(pooled, metadata["gaussian_pooled_training_std"])
    x = style_vector(piece())[0]
    scores = score_measures(first, x, custom["A0"], events["A0"])
    assert scores == score_measures(second, x, custom["A0"], events["A0"])
    for c in ("A", "B", "C"):
        profile = first["profiles"][c]
        assert scores["rms67"][c] == -np.mean(list(grouped_distance(x, profile).values()))
        expected_std = np.maximum.reduce([np.asarray(metadata["profiles"][c]["raw_std"]), .05 * pooled, np.full(67, 1e-6)])
        assert np.array_equal(expected_std, first["gaussian_std"][c])
        expected = np.exp(-.5 * ((x - profile.mean) / expected_std) ** 2)
        assert scores["gaussian67"][c] == pytest.approx(np.mean([expected[np.asarray(profile.feature_groups) == g].mean()
                                                               for g in ("pitch", "rhythm", "texture")]))
    corrupt_vectors = {r["sample_id"]: np.ones(67) for r in train}
    with pytest.raises(ValueError, match="differs from frozen E3"):
        fit_measures(train, forbidden, reprs, custom, events, vectors67=corrupt_vectors)
    assert first["logistic"][-1].C == 1
    assert first["logistic"][-1].max_iter == 5000


def test_duration_velocity_reassociation_changes_frozen_skyline_only():
    from collections import Counter
    from musicians_style.e1.composition_features import extract_composition_features
    tail = NoteEvent(480, 0, 60, 80, 480)
    a = InternalRepr(480, (NoteEvent(0, 0, 72, 49, 120), NoteEvent(0, 0, 72, 73, 960), tail))
    b = InternalRepr(480, (NoteEvent(0, 0, 72, 49, 960), NoteEvent(0, 0, 72, 73, 120), tail))
    # FIFO/serialization can preserve aggregate notes while associating durations with different velocities.
    without_velocity = lambda p: Counter((n.tick, n.channel, n.pitch, n.duration_ticks) for n in p.notes)
    assert without_velocity(a) == without_velocity(b)
    assert not np.array_equal(style_vector(a)[0], style_vector(b)[0])
    assert np.array_equal(extract_composition_features(a), extract_composition_features(b))
    assert all(np.array_equal(event_profiles(a)[0][n], event_profiles(b)[0][n]) for n in ('onset_duration', 'time_pitch'))
