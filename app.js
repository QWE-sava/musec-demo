// MUSE-C browser demo: mood -> chords (ONNX) -> piano (ONNX) -> Tone.js / MIDI
const PAD = 0, BOS = 1, EOS = 2, SEP = 3, BAR = 4;
const POS_B = 5, PIT_B = 21, DUR_B = 109, VEL_B = 141;
const CHQ = ['maj', 'min', '7', 'maj7', 'min7', 'sus'];
const CHB = 149, CHN = 221;
const MODE_MEL = 222, MODE_ACC = 223, MODE_FULL = 224;
const CCHB = 4, CCHN = 76, KEYB = 77, TAGB = 101, TAGN = 108;
const CH_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];
const MOODS = {
  '青春': { key: [0, 'maj'], prog: [[0, 'maj'], [7, 'maj'], [9, 'min'], [5, 'maj']] },
  '切ない': { key: [9, 'min'], prog: [[0, 'min'], [8, 'maj'], [3, 'maj'], [7, 'maj']] },
  '夜': { key: [9, 'min'], prog: [[0, 'min'], [5, 'min'], [8, 'maj'], [7, 'maj']] },
  '疾走感': { key: [2, 'min'], prog: [[0, 'min'], [8, 'maj'], [5, 'min'], [7, 'maj']] },
  '透明感': { key: [0, 'maj'], prog: [[0, 'maj7'], [4, 'min7'], [5, 'maj7'], [7, 'maj']] },
  '希望': { key: [7, 'maj'], prog: [[0, 'maj'], [9, 'min'], [5, 'maj'], [7, 'maj']] },
  '緊張感': { key: [4, 'min'], prog: [[0, 'min'], [0, 'min'], [1, 'maj'], [1, 'maj']] },
};
const MOOD_KEY = { '青春': [0, 'maj'], '切ない': [9, 'min'], '夜': [9, 'min'], '疾走感': [2, 'min'], '透明感': [0, 'maj'], '希望': [7, 'maj'], '緊張感': [4, 'min'] };
const MOOD_TAG = { '青春': 0, '切ない': 1, '夜': 2, '疾走感': 3, '透明感': 4, '希望': 5, '緊張感': 6 };

const logEl = document.getElementById('log');
const log = (s) => { logEl.textContent += '\n' + s; };
let chordSess = null, pianoSess = null, lastData = null;

function chordId(r, q) {
  if (r == null) return CHN;
  let qi = CHQ.indexOf(q); if (qi < 0) qi = 0;
  return CHB + r * 6 + qi;
}
function cChordId(r, q) {
  if (r == null) return CCHN;
  let qi = CHQ.indexOf(q); if (qi < 0) qi = 0;
  return CCHB + r * 6 + qi;
}
function keyId(r, q) { return KEYB + r * 2 + (q === 'maj' ? 0 : 1); }

function sample(logits, temp, topK) {
  // logits: Float32Array
  const scored = [];
  for (let i = 0; i < logits.length; i++) scored.push([logits[i] / temp, i]);
  scored.sort((a, b) => b[0] - a[0]);
  const top = scored.slice(0, topK);
  const mx = top[0][0];
  let sum = 0;
  const exps = top.map(([v]) => { const e = Math.exp(v - mx); sum += e; return e; });
  let r = Math.random() * sum;
  for (let i = 0; i < top.length; i++) { r -= exps[i]; if (r <= 0) return top[i][1]; }
  return top[top.length - 1][1];
}

async function runChord(sess, key, tag, nBars) {
  const ids = [BOS, TAGB + tag, keyId(key[0], key[1])];
  for (let s = 0; s < nBars + 4; s++) {
    const t = new ort.Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
    const out = await sess.run({ idx: t });
    const logits = out.logits.data;
    const last = logits.slice((ids.length - 1) * 109);
    const nxt = sample(last, 0.9, 12);
    if (nxt === EOS) break;
    ids.push(nxt);
    if (ids.filter(x => x >= CCHB && x < CCHB + 73).length >= nBars) break;
  }
  return ids;
}

async function runPiano(sess, chords, mode, nBars, onStep) {
  // past/cache mode: 12 layers, mask-as-input (parity validated)
  const L = 12, H = 8, DH = 32;
  const ids = [BOS, SEP, mode, ...chords.map(([r, q]) => chordId(r, q)), BAR];
  let K = [], V = [];
  for (let li = 0; li < L; li++) {
    K.push(new ort.Tensor('float32', new Float32Array(0), [1, H, 0, DH]));
    V.push(new ort.Tensor('float32', new Float32Array(0), [1, H, 0, DH]));
  }
  const zeros = (n) => new Float32Array(n);  // all-attend mask
  async function step(tok, pos, pastLen) {
    const feed = {
      idx: new ort.Tensor('int64', BigInt64Array.from([BigInt(tok)]), [1, 1]),
      pos: new ort.Tensor('int64', BigInt64Array.from([BigInt(pos)]), [1]),
      mask: new ort.Tensor('float32', zeros(1 * 1 * 1 * (pastLen + 1)), [1, 1, 1, pastLen + 1]),
    };
    for (let li = 0; li < L; li++) { feed['k' + li] = K[li]; feed['v' + li] = V[li]; }
    const names = ['logits'];
    for (let li = 0; li < L; li++) names.push('k' + li + 'n', 'v' + li + 'n');
    const out = await sess.run(feed);
    const nk = [], nv = [];
    for (let li = 0; li < L; li++) { nk.push(out['k' + li + 'n']); nv.push(out['v' + li + 'n']); }
    K = nk; V = nv;
    return out.logits;
  }
  // prefill prefix one by one
  let pastLen = 0, logits = null;
  for (const tok of ids.slice()) {
    logits = await step(tok, pastLen, pastLen);
    pastLen++;
  }
  let nbars = 1;
  for (let st = 0; st < nBars * 160; st++) {
    const data = logits.data;
    const last = Array.from(data.slice(0));
    const nxt = sample(last, 0.85, 25);
    ids.push(nxt);
    if (onStep) onStep(ids.length);
    if (nxt === EOS) break;
    if (nxt === BAR) { nbars++; if (nbars > nBars) break; }
    logits = await step(nxt, pastLen, pastLen);
    pastLen++;
  }
  return ids;
}


function decode(ids) {
  const bars = [];
  let cur = null, i = 0;
  const isSpec = (t) => t === BOS || t === SEP || t === EOS || t === PAD || t === MODE_MEL || t === MODE_ACC || t === MODE_FULL;
  while (i < ids.length) {
    const t = ids[i];
    if (t === BAR) {
      if (cur !== null) bars.push({ id: bars.length, notes: cur.sort((a, b) => a.pos16 - b.pos16 || a.pitch - b.pitch) });
      cur = []; i++;
    } else if (isSpec(t)) { i++; }
    else if (i + 3 < ids.length && ids[i] >= 5 && ids[i] < 21 && ids[i + 1] >= 21 && ids[i + 1] < 109 && ids[i + 2] >= 109 && ids[i + 2] < 141 && ids[i + 3] >= 141 && ids[i + 3] < 149) {
      if (cur === null) cur = [];
      cur.push({ pitch: ids[i + 1], pos16: ids[i] - 5, len16: ids[i + 2] - 109 + 1, vel: ids[i + 3] - 141 });
      i += 4;
    } else { i++; }
  }
  if (cur && cur.length) bars.push({ id: bars.length, notes: cur.sort((a, b) => a.pos16 - b.pos16 || a.pitch - b.pitch) });
  return { version: '0.1', tempo_qpm: 120, time_signature: '4/4', bars };
}

function midiFile(data) {
  // minimal SMF type-0 writer, 120bpm, piano
  const six = 0.125, tpq = 480;
  const ev = [];
  const pushVar = (arr, v) => { let b = [v & 127]; v >>= 7; while (v) { b.unshift((v & 127) | 128); v >>= 7; } b[b.length - 1] &= 127; arr.push(...b); };
  const tr = [];
  const tempo = Math.round(60000000 / 120);
  tr.push(0x00, 0xFF, 0x51, 0x03, (tempo >> 16) & 255, (tempo >> 8) & 255, tempo & 255);
  const notes = [];
  data.bars.forEach((bar, bi) => bar.notes.forEach(n => {
    const s = Math.round((bi * 16 + n.pos16) * six * tpq);
    const e = s + Math.round(Math.max(1, n.len16) * six * tpq);
    notes.push([s, 0x90, n.pitch, 40 + n.vel * 12]);
    notes.push([e, 0x80, n.pitch, 0]);
  }));
  notes.sort((a, b) => a[0] - b[0]);
  let last = 0;
  notes.forEach(([t, st, p, v]) => { const d = []; pushVar(d, t - last); last = t; tr.push(...d, st, p, v); });
  tr.push(0x00, 0xFF, 0x2F, 0x00);
  const head = [0x4D, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 0, 0, 1, (tpq >> 8) & 255, tpq & 255,
    0x4D, 0x54, 0x72, 0x6B, (tr.length >>> 24) & 255, (tr.length >>> 16) & 255, (tr.length >>> 8) & 255, tr.length & 255];
  return new Blob([new Uint8Array([...head, ...tr])], { type: 'audio/midi' });
}

async function ensureModels() {
  if (chordSess && pianoSess) return;
  if (window.libraryLoadError || typeof ort === 'undefined') {
    throw new Error(window.libraryLoadError || 'ONNX Runtime Webを読み込めませんでした。ページを再読み込みしてください。');
  }
  if (location.protocol === 'file:') {
    throw new Error('モデルを読み込むにはWebサーバー経由で開いてください（file:// では実行できません）。');
  }
  const ep = document.getElementById('ep').value;
  const providers = ep === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
  let loading = 'コードモデル (models/chord_model.onnx)';
  log(`${loading} 読込中 [${providers}]…`);
  try {
    chordSess = await ort.InferenceSession.create('./models/chord_model.onnx', {
      executionProviders: providers,
      // The exported chord graph stores its weights in this ONNX external-data sidecar.
      externalData: [{ path: 'chord_model.onnx.data', data: './models/chord_model.onnx.data' }],
    });
    // Dynamic INT8 is intended for WASM/CPU and cuts the piano model download
    // and weight memory substantially. Keep FP32 for WebGPU compatibility.
    const pianoModel = ep === 'wasm' ? './models/piano_past_int8.onnx' : './models/piano_past.onnx';
    loading = ep === 'wasm'
      ? 'ピアノモデル (INT8, 約10MB)'
      : 'ピアノモデル (FP32, 約40MB)';
    log(`${loading} 読込中…`);
    pianoSess = await ort.InferenceSession.create(pianoModel, { executionProviders: providers });
  } catch (e) {
    chordSess = null;
    pianoSess = null;
    const detail = [e?.name, e?.message, String(e)].filter((x, i, a) => x && a.indexOf(x) === i).join(' / ');
    throw new Error(`${loading} の読込に失敗しました。URLと接続を確認してください。詳細: ${detail || '詳細なし'}`);
  }
  log('models ok');
}

function moodChords(mood) {
  const m = MOODS[mood];
  const loop = [];
  for (let k = 0; k < 2; k++) for (const [r, q] of m.prog) loop.push([(m.key[0] + r) % 12, q]);
  return loop.slice(0, 8);
}

document.getElementById('mood').innerHTML = Object.keys(MOODS).map(m => `<option>${m}</option>`).join('');

document.getElementById('gen').onclick = async () => {
  const mood = document.getElementById('mood').value;
  const mode = document.getElementById('mode').value;
  if (mode === 'server') { await genServer(mood); return; }
  const bar = document.getElementById('bar');
  try {
    await ensureModels();
    log(`mood: ${mood} → chords…`);
    const key = MOOD_KEY[mood], tag = MOOD_TAG[mood];
    const cids = await runChord(chordSess, key, tag, 8);
    const chords = cids.filter(t => t >= CCHB && t < CCHB + 73).slice(0, 8)
      .map(t => t === CCHN ? [null, null] : [(t - CCHB) / 6 | 0, CHQ[(t - CCHB) % 6]]);
    while (chords.length < 8) chords.push([0, 'maj']);
    document.getElementById('chords').textContent =
      chords.map(([r, q]) => r == null ? 'N' : CH_NAMES[r] + q).join(' - ');
    log('piano…');
    const mel = await runPiano(pianoSess, chords, MODE_MEL, 8, (n) => { bar.value = Math.min(99, n / 14); });
    const acc = await runPiano(pianoSess, chords, MODE_ACC, 8, (n) => { bar.value = Math.min(99, n / 14); });
    const melD = decode(mel), accD = decode(acc);
    const full = { version: '0.1', tempo_qpm: 120, time_signature: '4/4', bars: [] };
    for (let i = 0; i < 8; i++) {
      const ns = [];
      if (i < melD.bars.length) melD.bars[i].notes.forEach(n => { if (n.pitch >= 28 && n.pitch <= 93) ns.push({ ...n, vel: Math.min(7, n.vel + 1) }); });
      if (i < accD.bars.length) accD.bars[i].notes.forEach(n => { if (n.pitch >= 28 && n.pitch <= 93) ns.push(n); });
      full.bars.push({ id: i, notes: ns.sort((a, b) => a.pos16 - b.pos16 || a.pitch - b.pitch) });
    }
    lastData = full;
    lastMidiB64 = null;
    showJson(full);
    bar.value = 100;
    log('done: ' + full.bars.map(b => b.notes.length).join(','));
    document.getElementById('play').disabled = false;
    document.getElementById('dl').disabled = false;
  } catch (e) { log('ERR: ' + e.message); }
};

document.getElementById('play').onclick = async () => {
  if (!lastData) return;
  await Tone.start();
  const synth = new Tone.PolySynth(Tone.Synth).toDestination();
  const six = 60 / 120 / 4;
  const now = Tone.now() + 0.1;
  lastData.bars.forEach((bar, bi) => bar.notes.forEach(n => {
    const t = now + (bi * 16 + n.pos16) * six;
    synth.triggerAttackRelease(Tone.Frequency(n.pitch, 'midi').toNote(), n.len16 * six, t, 0.3 + n.vel * 0.1);
  }));
};

async function genServer(mood) {
  const urlEl = document.getElementById('srv'), keyEl = document.getElementById('skey');
  const url = urlEl.value.replace(/\/$/, ''), key = keyEl.value;
  try { localStorage.setItem('musec_srv', url); } catch (e) {}
  try {
    const r = await fetch(url + '/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json', 'x-key': key },
      body: JSON.stringify({ mood, bars: 8 }),
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const j = await r.json();
    document.getElementById('chords').textContent = j.chords.join(' - ');
    lastData = j.json;
    showJson(j.json);
    document.getElementById('log').textContent +=
      `\nserver: ${j.time_s}s notes/bar: ${j.notes_per_bar}`;
    lastMidiB64 = j.midi_b64;
    document.getElementById('play').disabled = false;
    document.getElementById('dl').disabled = false;
  } catch (e) { document.getElementById('log').textContent += '\nERR: ' + e.message; }
}
try {
  const s = localStorage.getItem('musec_srv');
  if (s) document.getElementById('srv').value = s;
} catch (e) {}

let lastMidiB64 = null;
function showJson(data) {
  document.getElementById('jsonpre').textContent = JSON.stringify(data, null, 1);
  document.getElementById('dljson').disabled = false;
}
document.getElementById('dljson').onclick = () => {
  if (!lastData) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(lastData, null, 1)], { type: 'application/json' }));
  a.download = 'musec.json';
  a.click();
};
document.getElementById('dl').onclick = () => {
  if (!lastData) return;
  let blob;
  if (lastMidiB64) {
    const bin = atob(lastMidiB64);
    blob = new Blob([Uint8Array.from(bin, (c) => c.charCodeAt(0))], { type: 'audio/midi' });
    lastMidiB64 = null;
  } else {
    blob = midiFile(lastData);
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'musec.mid';
  a.click();
};
