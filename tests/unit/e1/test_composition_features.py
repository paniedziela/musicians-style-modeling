from __future__ import annotations

import numpy as np

from musicians_style.e1.composition_features import (
    FEATURE_GROUPS,
    FEATURE_SPECS,
    _variants,
    extract_composition_features,
    infer_musical_form,
)
from musicians_style.midi.types import InternalRepr, MetaEvent, NoteEvent


def _note(tick: int, pitch: int, duration: int, velocity: int = 80) -> NoteEvent:
    return NoteEvent(
        tick=tick,
        channel=0,
        pitch=pitch,
        velocity=velocity,
        duration_ticks=duration,
    )


def _score(*, tempo: int = 500_000, velocity: int = 80) -> InternalRepr:
    return InternalRepr(
        ticks_per_beat=480,
        notes=(
            _note(0, 60, 480, velocity),
            _note(0, 64, 960, velocity),
            _note(480, 62, 240, velocity),
            _note(960, 67, 480, velocity),
        ),
        meta=(
            MetaEvent(0, "tempo", {"tempo": tempo}),
            MetaEvent(0, "time_signature", {"numerator": 4, "denominator": 4}),
        ),
    )


def _named(vector: np.ndarray) -> dict[str, float]:
    return {
        specification["name"]: float(value)
        for specification, value in zip(FEATURE_SPECS, vector)
    }


def test_feature_contract_covers_every_value_and_all_predeclared_groups() -> None:
    vector = extract_composition_features(_score())
    assert vector.shape == (len(FEATURE_SPECS),)
    assert np.isfinite(vector).all()
    assert {item["group"] for item in FEATURE_SPECS} == set(FEATURE_GROUPS)
    assert len({item["name"] for item in FEATURE_SPECS}) == len(FEATURE_SPECS)
    assert all(item["unit"] for item in FEATURE_SPECS)


def test_pitch_melody_rhythm_texture_and_harmony_have_interpretable_values() -> None:
    values = _named(extract_composition_features(_score()))
    assert values["pitch_range"] == 7.0
    assert values["descending_interval_ratio"] == 0.5
    assert values["ascending_interval_ratio"] == 0.5
    assert values["downbeat_onset_ratio"] == 1.0 / 3.0
    assert values["metrical_pulse_onset_ratio"] == 1.0
    assert values["mean_polyphony"] == 1.5
    assert values["max_polyphony"] == 2.0
    assert values["chord_onset_ratio"] == 1.0 / 3.0
    assert values["chord_consonance_ratio"] == 1.0


def test_features_are_invariant_to_score_tempo_and_velocity() -> None:
    baseline = extract_composition_features(_score(tempo=500_000, velocity=30))
    changed = extract_composition_features(_score(tempo=250_000, velocity=120))
    np.testing.assert_allclose(changed, baseline)


def test_meter_relative_offbeat_and_syncopation_are_detected() -> None:
    score = InternalRepr(
        ticks_per_beat=480,
        notes=(_note(240, 60, 360), _note(720, 64, 240)),
        meta=(MetaEvent(0, "time_signature", {"numerator": 4, "denominator": 4}),),
    )
    values = _named(extract_composition_features(score))
    assert values["offbeat_onset_ratio"] == 1.0
    assert values["syncopated_note_ratio"] == 0.5


def test_variants_define_full_group_only_and_leave_one_group_out_ablations() -> None:
    variants = _variants()
    assert len(variants) == 1 + 2 * len(FEATURE_GROUPS)
    assert len(variants["composition_full"]["indices"]) == len(FEATURE_SPECS)
    for group in FEATURE_GROUPS:
        only = set(variants[f"only_{group}"]["indices"])
        without = set(variants[f"without_{group}"]["indices"])
        assert only
        assert only.isdisjoint(without)
        assert only | without == set(range(len(FEATURE_SPECS)))


def test_form_mapping_uses_predeclared_coarse_families() -> None:
    assert infer_musical_form("Prelude_bwv_846") == "prelude_fugue"
    assert infer_musical_form("Piano_Sonatas_17-1") == "sonata"
    assert infer_musical_form("Etudes_op_10_1") == "etude"
    assert infer_musical_form("Fantaisie_Impromptu") == "other"
