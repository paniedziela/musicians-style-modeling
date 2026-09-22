"""Small, offline MIDI listening server; no web framework required."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

MAX_BYTES = 10 * 1024 * 1024
MAX_SECONDS = 15 * 60
ASSETS = Path(__file__).with_name("static")


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class Library:
    def __init__(self, root: Path, soundfont: Path, cache: Path, *, profiles=None, inference_workspace=None, e4_checkpoint=None):
        self.root = root.resolve()
        self.soundfont = soundfont.resolve()
        self.cache = cache.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.uploads: dict[str, tuple[Path, str]] = {}
        from .inference import InferenceJobs
        self.inference = InferenceJobs(self, profiles or self.root / "inference_workspace/profiles",
                                       inference_workspace or self.cache / "inference",
                                       e4_checkpoint or self.root / "experiments/e4_asap_v3/best.pt")

    def resolve(self, file_id: str) -> Path:
        if file_id.startswith("upload:"):
            if file_id not in self.uploads:
                raise FileNotFoundError("Wgrany plik wygasł. Dodaj go ponownie.")
            return self.uploads[file_id][0]
        path = (self.root / file_id).resolve()
        if not path.is_relative_to(self.root) or path.suffix.lower() not in {".mid", ".midi"}:
            raise ValueError("Niedozwolona ścieżka MIDI.")
        if not path.is_file():
            raise FileNotFoundError("Nie znaleziono pliku MIDI.")
        return path

    def catalog(self) -> list[dict]:
        entries = []
        sources = {}
        source_labels = {}
        manifests = []
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                       {"node_modules", "venv", "build", "dist", "soundfonts"}
                       and (Path(directory) / d).resolve() != self.cache]
            folder = Path(directory)
            if "manifest.json" in files and folder.name == "inputs":
                manifests.append(folder / "manifest.json")
            for name in sorted(files):
                if Path(name).suffix.lower() not in {".mid", ".midi"}:
                    continue
                path = folder / name
                if not path.resolve().is_relative_to(self.root):
                    continue
                relative = path.relative_to(self.root)
                folder_name = relative.parent.as_posix()
                result = read_json(folder / "result.json") if name.lower() == "output.mid" else {}
                title = result.get("source_id") or path.stem
                target = result.get("target_composer", "")
                # E4 exports encode the work and target directly in the filename.
                if not target and "_to_" in title:
                    title, target = title.rsplit("_to_", 1)
                experiment = relative.parts[1] if relative.parts[0] == "experiments" and len(relative.parts) > 2 else relative.parts[0] if len(relative.parts) > 1 else "Pliki MIDI"
                entries.append({"id": relative.as_posix(), "name": title,
                                "path": relative.as_posix(), "folder": folder_name,
                                "experiment": experiment, "target": target,
                                "composer": result.get("source_composer", ""),
                                "kind": "Wynik" if target or result else "MIDI",
                                "source_id": result.get("source_id", title),
                                "size": path.stat().st_size})
        for manifest_path in manifests:
            manifest = read_json(manifest_path)
            dataset = Path(manifest.get("dataset_root", "datasets"))
            bases = [self.root, *manifest_path.parents]
            for sample in manifest.get("samples", []):
                if not isinstance(sample, dict) or not sample.get("score_path"):
                    continue
                if sample.get("sample_id") in sources:
                    continue
                for base in bases:
                    path = (base / dataset / sample["score_path"]).resolve()
                    if path.is_relative_to(self.root) and path.is_file():
                        sources[sample.get("sample_id")] = path.relative_to(self.root).as_posix()
                        source_labels[path.relative_to(self.root).as_posix()] = sample
                        break
        for entry in entries:
            entry["original"] = sources.get(entry["source_id"])
            if entry["id"] in source_labels:
                sample = source_labels[entry["id"]]
                entry.update(name=sample.get("sample_id", entry["name"]),
                             composer=sample.get("composer", ""), kind="Oryginał")
        return sorted(entries, key=lambda e: (e["experiment"], e["name"], e["target"]))

    def midi(self, file_id: str) -> tuple[object, Path]:
        import pretty_midi

        path = self.resolve(file_id)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Plik przekracza limit 10 MB.")
        try:
            midi = pretty_midi.PrettyMIDI(io.BytesIO(path.read_bytes()))
        except Exception as exc:
            raise ValueError("Nie można odczytać MIDI. Sprawdź format pliku.") from exc
        if midi.get_end_time() > MAX_SECONDS:
            raise ValueError("MIDI przekracza limit 15 minut.")
        if sum(len(i.notes) for i in midi.instruments) > 200_000:
            raise ValueError("MIDI ma zbyt wiele nut (limit 200 000).")
        if not any(i.notes for i in midi.instruments):
            raise ValueError("Plik MIDI nie zawiera nut do odsłuchu.")
        return midi, path

    def describe(self, file_id: str) -> dict:
        midi, path = self.midi(file_id)
        notes = sorted([[round(n.start, 4), round(n.end, 4), n.pitch, n.velocity, index]
                        for index, inst in enumerate(midi.instruments) for n in inst.notes])
        return {"id": file_id, "name": self.uploads[file_id][1] if file_id in self.uploads else path.name,
                "path": self.uploads[file_id][1] if file_id in self.uploads else file_id,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "duration": midi.get_end_time(), "notes": notes,
                "tracks": [{"name": i.name or ("Perkusja" if i.is_drum else f"Instrument {i.program + 1}"),
                            "notes": len(i.notes)} for i in midi.instruments],
                "audio": "/api/audio?id=" + quote(file_id, safe=""),
                "download": "/api/download?id=" + quote(file_id, safe="")}

    def upload(self, data: bytes, name: str) -> dict:
        if not data or len(data) > MAX_BYTES:
            raise ValueError("Wybierz plik MIDI do 10 MB.")
        key = hashlib.sha256(data).hexdigest()
        file_id = "upload:" + key
        path = self.cache / (key + ".mid")
        with self.lock:
            path.write_bytes(data)
            self.uploads[file_id] = (path, Path(name.replace("\\", "/")).name[:180] or "upload.mid")
            try:
                description = self.describe(file_id)
            except Exception:
                self.uploads.pop(file_id, None)
                path.unlink(missing_ok=True)
                raise
            return description

    def render(self, file_id: str) -> Path:
        import numpy as np

        if not self.soundfont.is_file():
            raise ValueError("Brak SoundFontu. Uruchom aplikację z --soundfont ścieżka/do/pliku.sf2.")
        midi, source = self.midi(file_id)
        stamp = self.soundfont.stat()
        key = hashlib.sha256(source.read_bytes() +
                             f"{self.soundfont}:{stamp.st_size}:{stamp.st_mtime_ns}:22050".encode()).hexdigest()
        target = self.cache / (key + ".wav")
        # FluidSynth rendering is serialized to bound RAM use and avoid duplicate work.
        with self.lock:
            if not target.exists():
                try:
                    samples = midi.fluidsynth(fs=22050, sf2_path=str(self.soundfont))
                except Exception as exc:
                    raise ValueError("Synteza nie działa. Sprawdź bibliotekę FluidSynth, pyFluidSynth i SoundFont.") from exc
                samples = np.nan_to_num(samples)
                peak = max(1.0, float(np.max(np.abs(samples))))
                pcm = (np.clip(samples / peak, -1, 1) * 32767).astype("<i2")
                with wave.open(str(target), "wb") as output:
                    output.setnchannels(1)
                    output.setsampwidth(2)
                    output.setframerate(22050)
                    output.writeframes(pcm.tobytes())
        return target


def make_handler(library: Library):
    class Handler(BaseHTTPRequestHandler):
        def send(self, body: bytes, content_type: str, status: int = 200, **headers):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in headers.items():
                self.send_header(key.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(body)

        def json(self, value, status=200):
            self.send(json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self):
            url = urlsplit(self.path)
            file_id = parse_qs(url.query).get("id", [""])[0]
            try:
                if url.path == "/api/inference/styles":
                    self.json({"styles": library.inference.styles(), "method": "E3"})
                elif url.path == "/api/inference/methods":
                    self.json(library.inference.methods())
                elif url.path == "/api/inference/status":
                    self.json(library.inference.status(file_id))
                elif url.path == "/api/library":
                    self.json({"files": library.catalog(), "root": str(library.root),
                               "soundfont": library.soundfont.is_file()})
                elif url.path == "/api/midi":
                    self.json(library.describe(file_id))
                elif url.path in {"/api/audio", "/api/download"}:
                    path = library.render(file_id) if url.path == "/api/audio" else library.resolve(file_id)
                    data = path.read_bytes()
                    content_type = "audio/wav" if url.path == "/api/audio" else "audio/midi"
                    # Browser media seeking requires byte range responses.
                    byte_range = self.headers.get("Range")
                    if byte_range:
                        import re
                        match = re.fullmatch(r"bytes=(\d*)-(\d*)", byte_range)
                        if not match or not any(match.groups()):
                            self.send(b"", content_type, 416, Content_Range=f"bytes */{len(data)}")
                            return
                        first, last = match.groups()
                        start = int(first) if first else max(0, len(data) - int(last))
                        end = min(int(last), len(data) - 1) if first and last else len(data) - 1
                        if start >= len(data) or start > end:
                            self.send(b"", content_type, 416, Content_Range=f"bytes */{len(data)}")
                            return
                        self.send(data[start:end + 1], content_type, 206,
                                  Content_Range=f"bytes {start}-{end}/{len(data)}", Accept_Ranges="bytes")
                    else:
                        self.send(data, content_type, Accept_Ranges="bytes")
                elif url.path in {"/", "/app.js", "/style.css", "/details.css"}:
                    name = "index.html" if url.path == "/" else url.path[1:]
                    mime = {"index.html": "text/html", "app.js": "text/javascript",
                            "style.css": "text/css", "details.css": "text/css"}[name]
                    self.send((ASSETS / name).read_bytes(), mime + "; charset=utf-8")
                else:
                    self.json({"error": "Nie znaleziono strony."}, 404)
            except FileNotFoundError as exc:
                self.json({"error": str(exc)}, 404)
            except ValueError as exc:
                self.json({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.json({"error": "Błąd odczytu lub syntezy pliku MIDI."}, 500)

        def do_POST(self):
            if urlsplit(self.path).path == "/api/inference":
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    self.json({"error": "Wymagany application/json."}, 415)
                    return
                origin = self.headers.get("Origin")
                if origin and origin != "http://" + self.headers.get("Host", ""):
                    self.json({"error": "Niedozwolone pochodzenie żądania."}, 403)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 4096:
                        raise ValueError("Niepoprawny rozmiar żądania.")
                    value = json.loads(self.rfile.read(length))
                    if not isinstance(value, dict):
                        raise ValueError("Wymagany obiekt JSON.")
                    self.json(library.inference.start(value), 202)
                except (ValueError, OSError) as exc:
                    self.json({"error": str(exc)}, 400)
                return
            # Uploads are accepted only from this application, not cross-origin forms.
            if urlsplit(self.path).path != "/api/upload":
                self.json({"error": "Nie znaleziono strony."}, 404)
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "audio/midi":
                self.json({"error": "Wymagany plik MIDI."}, 415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BYTES:
                    raise ValueError("Wybierz plik MIDI do 10 MB.")
                name = parse_qs(urlsplit(self.path).query).get("name", ["upload.mid"])[0]
                self.json(library.upload(self.rfile.read(length), name))
            except ValueError as exc:
                self.json({"error": str(exc)}, 400)
            except Exception:
                self.json({"error": "Nie można wgrać MIDI."}, 500)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Lokalny odsłuch MIDI i wyników eksperymentów")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Katalog biblioteki MIDI (rekurencyjnie)")
    parser.add_argument("--soundfont", type=Path, default=Path("soundfonts/FluidR3_GM.sf2"))
    parser.add_argument("--cache", type=Path, default=Path("experiments/.midi-listener"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--profiles-dir", type=Path, default=Path("inference_workspace/profiles"))
    parser.add_argument("--inference-workspace", type=Path, default=Path("inference_workspace/web"))
    parser.add_argument("--e4-checkpoint", type=Path, default=Path("experiments/e4_asap_v3/best.pt"))
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error("Katalog --root nie istnieje.")
    library = Library(args.root, args.soundfont, args.cache, profiles=args.profiles_dir,
                      inference_workspace=args.inference_workspace, e4_checkpoint=args.e4_checkpoint)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(library))
    print(f"Odsłuch MIDI: http://127.0.0.1:{server.server_port} (Ctrl+C kończy pracę)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
