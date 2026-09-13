"use strict";
const $ = id => document.getElementById(id);
const slots = { A: {}, B: {} };
let active = "A", files = [], uploads = [], loading = new Set();
let background = document.createElement("canvas");
const time = n => `${Math.floor((n || 0) / 60)}:${String(Math.floor((n || 0) % 60)).padStart(2, "0")}`;
function status(message, error = false) { $("status").textContent = message; $("status").classList.toggle("error", error); }
async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let message = "Nie udało się odczytać pliku.";
    try { message = (await response.json()).error || message; } catch (_) {}
    throw new Error(message);
  }
  return response;
}
function options(id, values, label) {
  const select = $(id), previous = select.value;
  select.replaceChildren(new Option(label, ""));
  [...new Set(values.filter(Boolean))].sort().forEach(v => select.add(new Option(v, v)));
  select.value = previous;
}
async function refresh() {
  $("refresh").disabled = true;
  try {
    const data = await (await request("/api/library")).json();
    files = data.files;
    $("root").textContent = data.root;
    options("experiment", [...files, ...uploads].map(f => f.experiment), "Wszystkie eksperymenty");
    options("target", files.map(f => f.target), "Wszystkie style docelowe");
    renderList();
    if (!data.soundfont) status("Brak SoundFontu. Ustaw --soundfont przy uruchamianiu aplikacji.", true);
  } catch (error) { status(error.message, true); }
  finally { $("refresh").disabled = false; }
}
function renderList() {
  const query = $("search").value.trim().toLocaleLowerCase();
  const matches = [...uploads, ...files].filter(f =>
    (!$("experiment").value || f.experiment === $("experiment").value) &&
    (!$("target").value || f.target === $("target").value) &&
    `${f.name} ${f.id} ${f.composer} ${f.target}`.toLocaleLowerCase().includes(query));
  $("count").textContent = `${matches.length} plików`;
  const fragment = document.createDocumentFragment();
  // Keep the library responsive even for a large dataset; filters search all entries.
  matches.slice(0, 250).forEach(file => {
    const row = document.createElement("div"); row.className = "file";
    const info = document.createElement("div"); info.className = "file-info";
    const name = document.createElement("strong"); name.textContent = file.name; name.title = file.id;
    const detail = document.createElement("small"); detail.textContent = `${file.experiment}${file.target ? ` → ${file.target}` : ` · ${file.kind || "MIDI"}`}`;
    info.append(name, detail); row.append(info);
    ["A", "B"].forEach(slot => {
      const button = document.createElement("button"); button.textContent = slot;
      button.title = `Załaduj ${file.name} do ${slot}`;
      button.setAttribute("aria-label", button.title);
      button.classList.toggle("selected", slots[slot].data?.id === file.id);
      button.disabled = loading.has(slot);
      button.onclick = () => load(slot, file);
      row.append(button);
    });
    fragment.append(row);
  });
  if (!matches.length || matches.length > 250) {
    const hint = document.createElement("p"); hint.className = "muted list-hint";
    hint.textContent = matches.length ? "Pokazano pierwsze 250 plików. Zawęź wyszukiwanie lub wybierz eksperyment." : "Brak plików pasujących do filtrów.";
    fragment.append(hint);
  }
  $("files").replaceChildren(fragment);
}
async function load(slot, file, description) {
  if (loading.has(slot)) return;
  loading.add(slot);
  const old = slots[slot];
  old.audio?.pause();
  if (old.audio) { old.audio.removeAttribute("src"); old.audio.load(); }
  if (old.url) URL.revokeObjectURL(old.url);
  slots[slot] = { file };
  $("name" + slot).textContent = file.name;
  $("meta" + slot).textContent = "Odczyt MIDI…";
  $("download" + slot).hidden = true;
  $("original" + slot).hidden = true;
  status(`Ładowanie pliku ${slot}…`); renderList(); updateControls(); drawBackground();
  try {
    const data = description || await (await request(`/api/midi?id=${encodeURIComponent(file.id)}`)).json();
    slots[slot].data = data;
    $("meta" + slot).textContent = `${time(data.duration)} · ${data.notes.length} nut · ${data.tracks.length} ścieżek`;
    $("download" + slot).href = data.download;
    $("download" + slot).download = data.name;
    $("download" + slot).hidden = false;
    if (file.original && file.original !== file.id) {
      const button = $("original" + slot); button.hidden = false;
      button.onclick = () => load(slot === "A" ? "B" : "A", files.find(f => f.id === file.original) || { id: file.original, name: `${file.name} · oryginał` });
    }
    drawBackground();
    status(`Synteza pliku ${slot}… Przy pierwszym odsłuchu może potrwać kilka sekund.`);
    const blob = await (await request(data.audio)).blob();
    const url = URL.createObjectURL(blob), audio = new Audio(url);
    slots[slot].url = url;
    audio.preload = "auto";
    audio.volume = Number($("volume").value);
    audio.playbackRate = Number($("speed").value);
    audio.preservesPitch = false;
    audio.loop = $("loop").checked;
    await new Promise((resolve, reject) => {
      audio.addEventListener("loadedmetadata", resolve, { once: true });
      audio.addEventListener("error", () => reject(new Error("Przeglądarka nie może odtworzyć audio.")), { once: true });
      audio.load();
    });
    Object.assign(slots[slot], { audio, url });
    audio.addEventListener("ended", updateControls);
    audio.addEventListener("play", updateControls);
    audio.addEventListener("pause", updateControls);
    status(`Plik ${slot} gotowy do odsłuchu.`);
  } catch (error) { status(error.message, true); $("meta" + slot).textContent = "Nie udało się przygotować odsłuchu."; }
  finally { loading.delete(slot); renderList(); updateControls(); }
}
async function selectSlot(slot) {
  if (slot === active) return;
  const previous = slots[active].audio, position = previous?.currentTime || 0;
  const playing = previous && !previous.paused;
  previous?.pause(); active = slot;
  const next = slots[active].audio;
  if (next) {
    next.currentTime = Math.min(position, Math.max(0, next.duration - 0.01));
    if (playing) { try { await next.play(); } catch (error) { status(error.message, true); } }
  }
  updateControls(); drawBackground();
}
function updateControls() {
  const item = slots[active], audio = item.audio;
  $("play").disabled = !audio;
  $("play").textContent = audio && !audio.paused ? "Ⅱ Pauza" : "▶ Odtwórz";
  $("seek").disabled = !audio;
  $("seek").max = audio?.duration || item.data?.duration || 1;
  $("duration").textContent = time(audio?.duration || item.data?.duration);
  $("wav").hidden = !audio;
  if (item.url) $("wav").href = item.url;
  ["A", "B"].forEach(s => {
    $("slot" + s).classList.toggle("active", s === active);
    document.querySelector(`[data-slot="${s}"]`).setAttribute("aria-pressed", String(s === active));
  });
  $("note-count").textContent = item.data ? `${item.data.notes.length} nut · plik ${active}` : "—";
}
async function togglePlay() {
  const audio = slots[active].audio;
  if (!audio) return;
  if (!audio.paused) audio.pause();
  else { try { await audio.play(); } catch (error) { status(error.message, true); } }
  updateControls();
}
function seek(value) {
  const audio = slots[active].audio;
  if (audio) audio.currentTime = Math.min(Math.max(0, value), audio.duration);
}
function drawBackground() {
  const canvas = $("roll"), box = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width = background.width = Math.round(box.width * dpr);
  canvas.height = background.height = Math.round(box.height * dpr);
  const ctx = background.getContext("2d"), w = background.width, h = background.height;
  ctx.fillStyle = "#111921"; ctx.fillRect(0, 0, w, h);
  const data = slots[active].data;
  $("roll-empty").hidden = !!data;
  if (!data) return;
  let low = 127, high = 0;
  data.notes.forEach(n => { low = Math.min(low, n[2]); high = Math.max(high, n[2]); });
  low -= 2; high += 2;
  const rows = high - low + 1, left = 34 * dpr, duration = Math.max(data.duration, 1);
  for (let pitch = low; pitch <= high; pitch++) {
    const y = (high - pitch) / rows * h;
    ctx.fillStyle = [1, 3, 6, 8, 10].includes(pitch % 12) ? "#16212b" : "#111921";
    ctx.fillRect(left, y, w - left, h / rows);
    if (pitch % 12 === 0) {
      ctx.fillStyle = "#728698"; ctx.font = `${10 * dpr}px sans-serif`;
      ctx.fillText(`C${Math.floor(pitch / 12) - 1}`, 4 * dpr, y + 9 * dpr);
    }
  }
  for (let step = 0; step <= 8; step++) {
    const x = left + step / 8 * (w - left);
    ctx.strokeStyle = "#24313d"; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  data.notes.forEach(n => {
    ctx.fillStyle = active === "A" ? `rgba(100,214,182,${0.35 + n[3] / 200})` : `rgba(175,160,255,${0.35 + n[3] / 200})`;
    ctx.fillRect(left + n[0] / duration * (w - left), (high - n[2]) / rows * h,
      Math.max(dpr, (n[1] - n[0]) / duration * (w - left)), Math.max(dpr, h / rows - dpr));
  });
}
function frame() {
  const audio = slots[active].audio, position = audio?.currentTime || 0;
  $("position").textContent = time(position);
  if (document.activeElement !== $("seek")) $("seek").value = position;
  const canvas = $("roll"), ctx = canvas.getContext("2d");
  ctx.drawImage(background, 0, 0);
  if (slots[active].data) {
    const left = 34 * (window.devicePixelRatio || 1);
    const x = left + position / Math.max(1, slots[active].data.duration) * (canvas.width - left);
    ctx.strokeStyle = "#f0f4f8"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  requestAnimationFrame(frame);
}
async function addFiles(selected) {
  for (const file of selected) {
    if (!/\.(mid|midi)$/i.test(file.name)) { status(`Plik ${file.name} nie jest MIDI.`, true); continue; }
    if (file.size > 10 * 1024 * 1024) { status("Limit rozmiaru pliku to 10 MB.", true); continue; }
    try {
      status(`Dodawanie ${file.name}…`);
      const data = await (await request(`/api/upload?name=${encodeURIComponent(file.name)}`, { method: "POST", headers: { "Content-Type": "audio/midi" }, body: file })).json();
      const entry = { id: data.id, name: file.name, experiment: "Własne pliki", target: "", composer: "" };
      uploads = [entry, ...uploads.filter(f => f.id !== entry.id)];
      options("experiment", [...files, ...uploads].map(f => f.experiment), "Wszystkie eksperymenty");
      $("experiment").value = "Własne pliki"; $("target").value = ""; $("search").value = "";
      await load(!slots.A.data ? "A" : !slots.B.data ? "B" : active, entry, data);
    } catch (error) { status(error.message, true); }
  }
  renderList(); $("upload").value = "";
}
$("refresh").onclick = refresh;
["search", "experiment", "target"].forEach(id => $(id).addEventListener("input", renderList));
document.querySelectorAll("[data-slot]").forEach(button => button.onclick = () => selectSlot(button.dataset.slot));
$("play").onclick = togglePlay;
$("restart").onclick = () => seek(0);
$("seek").oninput = event => seek(Number(event.target.value));
$("volume").oninput = () => Object.values(slots).forEach(s => { if (s.audio) s.audio.volume = Number($("volume").value); });
$("speed").onchange = () => Object.values(slots).forEach(s => { if (s.audio) s.audio.playbackRate = Number($("speed").value); });
$("loop").onchange = () => Object.values(slots).forEach(s => { if (s.audio) s.audio.loop = $("loop").checked; });
$("upload").onchange = event => addFiles(event.target.files);
["dragenter", "dragover"].forEach(type => document.addEventListener(type, event => { event.preventDefault(); $("drop").classList.add("dragging"); }));
document.addEventListener("dragleave", () => $("drop").classList.remove("dragging"));
document.addEventListener("drop", event => { event.preventDefault(); $("drop").classList.remove("dragging"); addFiles(event.dataTransfer.files); });
document.addEventListener("keydown", event => {
  if (event.target.closest("input,select,textarea")) return;
  if (event.code === "Space" && !event.target.closest("button,a")) { event.preventDefault(); togglePlay(); }
  if (["a", "b"].includes(event.key.toLowerCase())) selectSlot(event.key.toUpperCase());
  if (["ArrowLeft", "ArrowRight"].includes(event.key)) { event.preventDefault(); seek((slots[active].audio?.currentTime || 0) + (event.key === "ArrowRight" ? 5 : -5)); }
});
new ResizeObserver(drawBackground).observe($("roll"));
drawBackground(); updateControls(); frame(); refresh();
