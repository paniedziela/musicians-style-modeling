"""Synthetic workflow tests; never load the local scientific corpus."""

import json
from types import SimpleNamespace

import joblib
import pytest

from musicians_style import research_ab as ab
from musicians_style.feature_backends.jsymbolic import numeric_schema
from musicians_style.midi.parser import MidiParser
from musicians_style.midi.printer import MidiPrettyPrinter
from musicians_style.midi.types import InternalRepr, NoteEvent
from musicians_style.provenance import sha256_file, write_json


def fixture(root):
    data = root / "datasets"
    derived = data / "derived/e1_asap"
    derived.mkdir(parents=True)
    sources = data / "asap-dataset-1.2"
    sources.mkdir()
    rows = []
    for ci, composer in enumerate(("Bach", "Beethoven", "Chopin")):
        for i in range(9):
            sid = f"{composer}-{i}"
            piece = InternalRepr(
                480,
                tuple(
                    n
                    for j in range(5)
                    for n in (
                        NoteEvent(j * 480, 0, 40 + ci * 8 + j, 60, 100 + i),
                        NoteEvent(j * 480, 0, 80 + ci, 80, 300 + i),
                    )
                ),
                (),
                0,
            )
            content = MidiPrettyPrinter().to_bytes(piece)
            import hashlib

            rows.append(
                dict(
                    sample_id=sid,
                    group_id=sid,
                    composer=composer,
                    sha256=hashlib.sha256(content).hexdigest(),
                    score_path=sid + ".mid",
                    validation_status="accepted",
                )
            )
            # No outer-test source files exist for r00/f00.
            if i % 3:
                (sources / (sid + ".mid")).write_bytes(content)
    folds = []
    for fold in range(3):
        test = [r["sample_id"] for r in rows if int(r["sample_id"][-1]) % 3 == fold]
        train = [r["sample_id"] for r in rows if r["sample_id"] not in test]
        left = [s for s in train if int(s[-1]) % 3 == (fold + 1) % 3]
        right = [s for s in train if s not in left]
        folds.append(
            dict(
                fold=fold,
                train=dict(sample_ids=train),
                test=dict(sample_ids=test),
                inner_folds=[
                    dict(
                        fold=0,
                        train=dict(sample_ids=left),
                        validation=dict(sample_ids=right),
                    ),
                    dict(
                        fold=1,
                        train=dict(sample_ids=right),
                        validation=dict(sample_ids=left),
                    ),
                ],
            )
        )
    manifest = dict(samples=rows)
    splits = dict(repetitions=[dict(repeat=0, folds=folds)])
    write_json(derived / "manifest.json", manifest)
    write_json(derived / "splits.json", splits)
    results = root / "experiments"
    results.mkdir()
    return manifest, splits, sources, results


class FakeRuntime:
    lock = dict(version="test")

    def __init__(self):
        self.paths = []

    def extract(self, path, output):
        self.paths.append(path.resolve())
        output.mkdir(parents=True)
        piece = MidiParser().parse(path)
        result = dict(
            status="success",
            schema=numeric_schema([dict(name="test", dimensions=3)]),
            values=[
                float(sum(n.pitch for n in piece.notes)),
                float(sum(n.duration_ticks for n in piece.notes)),
                1.0,
            ],
            diagnostics=dict(missing=[], nonfinite=[], excluded=[]),
        )
        write_json(output / "result.json", result)
        return result


def protocol(path, roots, runtime, stage="track_a", **kwargs):
    _, _, _, hashes = ab.inputs(roots)
    content = dict(
        stage=stage,
        approved_for_execution=True,
        input_metadata_hashes=hashes,
        backend_lock=runtime.lock,
        **kwargs,
    )
    write_json(path, content)
    return content


def test_selection_two_lexical_validation_works_no_score_or_length_dependence(tmp_path):
    manifest, splits, _, _ = fixture(tmp_path)
    train, validation, forbidden, cohort = ab.partition(
        manifest, splits, two_works=True
    )
    assert [r["sample_id"] for r in cohort] == [
        "Bach-2",
        "Bach-5",
        "Beethoven-2",
        "Beethoven-5",
        "Chopin-2",
        "Chopin-5",
    ]
    manifest["samples"].reverse()
    for r in manifest["samples"]:
        r.update(style_score=999, note_count=-1)
    assert [
        r["sample_id"] for r in ab.partition(manifest, splits, two_works=True)[3]
    ] == [r["sample_id"] for r in cohort]
    assert not {r["sample_id"] for r in train} & {r["sample_id"] for r in forbidden}


def test_review_gates_before_scientific_access(tmp_path, monkeypatch):
    _, _, _, results = fixture(tmp_path)
    runtime = FakeRuntime()
    roots = ab.resolve_asset_roots(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("original MIDI must not be touched")

    monkeypatch.setattr(ab, "verified_source", forbidden)
    with pytest.raises(ValueError, match="reviewed protocol"):
        ab.evaluate_a(tmp_path, results / "a", runtime, results / "previous", None)
    p = tmp_path / "review.json"
    protocol(p, roots, runtime)
    bad = json.loads(p.read_text())
    bad["approved_for_execution"] = False
    write_json(p, bad)
    with pytest.raises(ValueError, match="not approved"):
        ab.evaluate_a(tmp_path, results / "a", runtime, results / "previous", p)
    protocol(p, roots, runtime, stage="track_b")
    with pytest.raises(ValueError, match="predeclare"):
        ab.pilot_b(
            tmp_path, results / "b", runtime, results / "previous", tmp_path / "fit", p
        )
    assert not (results / "a").exists() and not (results / "b").exists()


def test_fresh_and_path_hash_guards(tmp_path):
    _, _, sources, results = fixture(tmp_path)
    with pytest.raises(ValueError, match="fresh"):
        ab.fresh(tmp_path, tmp_path / "datasets/new")
    (results / "old").mkdir()
    with pytest.raises(ValueError, match="nest"):
        ab.fresh(tmp_path, results / "old/new")
    missing = dict(sample_id="x", score_path="../escape.mid", sha256="bad")
    assert ab.custom_attempt(sources, missing)["status"] == "failure"


@pytest.mark.integration
def test_complete_synthetic_a_fit_before_score_and_b_budget_dispatch(
    tmp_path, monkeypatch
):
    manifest, splits, sources, results = fixture(tmp_path)
    runtime = FakeRuntime()
    roots = ab.resolve_asset_roots(tmp_path)
    train, validation, forbidden, cohort = ab.partition(
        manifest, splits, two_works=True
    )
    previous = results / "v205"
    previous.mkdir()
    rows = []
    for source in [cohort[i] for i in (0, 2, 4)]:
        for target in ("Bach", "Beethoven", "Chopin"):
            if target == source["composer"]:
                continue
            for name in ("rms67", "gaussian67", "logistic93"):
                task = f"{source['sample_id']}_{target}_{name}"
                directory = previous / "tasks" / task
                directory.mkdir(parents=True)
                (directory / "output.mid").write_bytes(
                    (sources / source["score_path"]).read_bytes()
                )
                rows.append(
                    dict(
                        task_id=task,
                        source_id=source["sample_id"],
                        source_sha256=source["sha256"],
                        target=target,
                        status="completed",
                        output_sha256=sha256_file(directory / "output.mid"),
                    )
                )
    write_json(previous / "per_output.json", rows)
    monkeypatch.setattr(ab, "collect_provenance", lambda _: dict(import_guard="test"))
    original_score = ab.score_external

    def score(*args, **kwargs):
        assert (results / "a/frozen_external.joblib").exists()
        return original_score(*args, **kwargs)

    monkeypatch.setattr(ab, "score_external", score)
    p = tmp_path / "review.json"
    protocol(p, roots, runtime)
    summary = ab.evaluate_a(tmp_path, results / "a", runtime, previous, p)
    assert summary["real_rows"] == 9 and summary["outputs"] == 18
    assert summary["output_movement"]["external"]["all"]["row_mean"] == 0
    assert all(
        path.exists() and ("tasks" in path.parts or int(path.stem[-1]) % 3)
        for path in runtime.paths
    )
    assert read_json(results / "a/fit_manifest.json")["persisted_before_scoring"]
    # Fit only synthetic inner-train to build a synthetic V2-05 development bundle.
    from musicians_style.e1.composition_features import extract_composition_features
    from musicians_style.style_metrics import event_profiles, fit_measures

    pieces = {
        r["sample_id"]: MidiParser().parse(sources / r["score_path"]) for r in train
    }
    fitted, metadata = fit_measures(
        train,
        forbidden,
        pieces,
        {s: extract_composition_features(p) for s, p in pieces.items()},
        {s: event_profiles(p)[0] for s, p in pieces.items()},
    )
    (previous / "fits").mkdir()
    joblib.dump(dict(style=fitted), previous / "fits/inner00.joblib")
    write_json(previous / "fits/inner00.json", metadata)
    external = results / "a/frozen_external.joblib"
    protocol(
        p,
        roots,
        runtime,
        stage="track_b",
        external_bundle_sha256=sha256_file(external),
        objective_bundle_sha256=sha256_file(previous / "fits/inner00.joblib"),
        continuation_thresholds={"synthetic": "only"},
        listening_design=None,
    )
    import musicians_style.accompaniment_search as pitch
    import musicians_style.objective_pilot as e3
    from musicians_style.e3.types import E3Genome

    requests = []

    def local(source, objective, **kwargs):
        assert kwargs["proposals"] == 512 and kwargs["changed_fraction"] == 0.20
        requests.append(("pitch", kwargs["seed"]))
        assert (results / "b/optimization_started.json").exists()
        return pitch.SearchResult(
            (),
            source.original,
            objective(source.original),
            0.0,
            (),
            dict(proposals=512),
        )

    def inherited(source, profile, objective, config, seed):
        assert (
            config.population_size == 32
            and config.generations == 16
            and config.stagnation_generations == 17
        )
        requests.append(("e3", seed))
        evaluation = objective.evaluate(E3Genome(), source)
        return SimpleNamespace(
            output=source,
            evaluation=evaluation,
            history=(),
            unique_candidates=1,
            cache_hits=544,
            genome=E3Genome(),
            stop_reason="test",
        ), {}

    monkeypatch.setattr(pitch, "local_search", local)
    monkeypatch.setattr(e3, "run_search", inherited)
    before = external.read_bytes()
    outcome = ab.pilot_b(tmp_path, results / "b", runtime, previous, external, p)
    assert outcome["runs"] == outcome["completed"] == 48
    assert len(requests) == 48 and set(seed for _, seed in requests) == {1729, 1730}
    assert external.read_bytes() == before
    assert len(read_json(results / "b/pilot_manifest.json")["cohort"]) == 6


def read_json(path):
    return json.loads(path.read_text())
