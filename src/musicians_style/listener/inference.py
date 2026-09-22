"""One local E3/E4 inference job at a time; paths never come from HTTP."""
from pathlib import Path
import threading
import uuid

from ..e3.algorithm import SearchConfig
from ..e3.inference import available_profiles, infer, validate_input, write_new


class InferenceJobs:
    def __init__(self, library, profiles: Path, workspace: Path, e4_checkpoint: Path):
        self.library = library
        self.profiles = profiles
        self.workspace = workspace.resolve()
        self.lock = threading.Lock()
        self.jobs = {}
        self.busy = False
        self.e4_checkpoint = e4_checkpoint

    def styles(self):
        return list(available_profiles(self.profiles))

    def methods(self):
        methods = []
        errors = {}
        try:
            methods.append({"id": "e3", "label": "E3 — algorytm genetyczny", "styles": self.styles()})
        except Exception as exc:
            errors["e3"] = str(exc)
        try:
            from ..e4.inference import checkpoint_info
            info = checkpoint_info(self.e4_checkpoint)
            methods.append({"id": "e4", "label": "E4.6 — eksperymentalny GAN (NO-GO)",
                            "styles": info["geometry"]["composers"], "experimental": True})
        except Exception as exc:
            errors["e4"] = str(exc)
        return {"methods": methods, "unavailable": errors}

    def status(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise FileNotFoundError("Nie znaleziono zadania inferencji.")
            return dict(self.jobs[job_id])

    def start(self, request):
        if set(request) - {"input_id", "target", "seed", "generations", "population_size", "method", "midi_policy"}:
            raise ValueError("Nieobsługiwane parametry inferencji.")
        target = request.get("target")
        method = request.get("method", "e3")
        if not isinstance(method, str) or method not in {"e3", "e4"}:
            raise ValueError("Nieznana metoda inferencji.")
        styles = self.styles() if method == "e3" else next((item["styles"] for item in self.methods()["methods"] if item["id"] == "e4"), [])
        if target not in styles:
            raise ValueError("Niedostępny profil docelowy.")
        policy = request.get("midi_policy", "preserve")
        values = {}
        for name, default, low, high in (("seed", 1729, 0, 2**32-1),
                                       ("generations", 60, 0, 200), ("population_size", 32, 2, 128)):
            value = request.get(name, default)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name}: wymagane {low}..{high}.")
            values[name] = value
        input_id = request.get("input_id")
        if not isinstance(input_id, str):
            raise ValueError("Najpierw załaduj MIDI do A.")
        source = self.library.resolve(input_id)
        if source.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Limit pliku: 10 MB.")
        data = source.read_bytes()
        validate_input(data, midi_policy=policy)
        with self.lock:
            if self.busy:
                raise ValueError("Inna transformacja trwa. Poczekaj na zakończenie.")
            job_id = uuid.uuid4().hex
            folder = self.workspace / job_id
            write_new(folder / "input.mid", data)
            self.busy = True
            self.jobs[job_id] = {"id": job_id, "state": "running", "method": method, "generation": None}
        def update(row):
            with self.lock:
                self.jobs[job_id].update({key: row[key] for key in ("generation", "segment", "segments") if key in row})
        def work():
            try:
                if method == "e3":
                    report = infer(folder / "input.mid", target, folder / "output.mid",
                                   profiles_dir=self.profiles, seed=values["seed"], midi_policy=policy,
                                   config=SearchConfig(generations=values["generations"],
                                                       population_size=values["population_size"]), progress=update)
                else:
                    from ..e4.inference import infer as infer_e4
                    report = infer_e4(folder / "input.mid", target, folder / "output.mid",
                                      checkpoint=self.e4_checkpoint, midi_policy=policy, progress=update)
                original_id, output_id = "upload:" + job_id + "-input", "upload:" + job_id + "-output"
                with self.library.lock:
                    self.library.uploads[original_id] = (folder / "input.mid", "Oryginał inferencji.mid")
                    self.library.uploads[output_id] = (folder / "output.mid", f"{report['method']}_{target}.mid")
                with self.lock:
                    self.jobs[job_id].update(state="completed", report=report,
                                             input_id=original_id, output_id=output_id)
            except Exception as exc:
                with self.lock:
                    self.jobs[job_id].update(state="failed", error=str(exc))
            finally:
                with self.lock:
                    self.busy = False
        threading.Thread(target=work, daemon=True).start()
        return {"id": job_id, "state": "running"}
