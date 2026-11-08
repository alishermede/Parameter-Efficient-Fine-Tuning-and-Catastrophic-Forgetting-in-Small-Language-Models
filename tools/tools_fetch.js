// Run in a kaggle.com tab. SID = kernelSessionId, RUN/CKPT = adapter checkpoint.
// Returns compact JSON: per task, a hash of the item_id order, packed correct bits (and mmlu preds).
async function grab(SID, RUN, CKPT) {
  const xs = document.cookie.match(/XSRF-TOKEN=([^;]+)/);
  async function get(fn) {
    const r = await fetch('/api/i/datasets.DatasetService/GetKernelOutputDataViewExternal', {
      method: 'POST', credentials: 'include',
      headers: {'content-type': 'application/json', 'x-xsrf-token': xs ? decodeURIComponent(xs[1]) : ''},
      body: JSON.stringify({kernelSessionId: SID, fileName: fn})});
    if (r.status !== 200) return null;
    const j = await r.json();
    return j.dataView && j.dataView.dataRaw ? j.dataView.dataRaw.data : null;
  }
  const base = RUN.startsWith("repo/") ? RUN : `repo/results/eval/${RUN}/${CKPT}/`;
  const out = {run: RUN, ckpt: CKPT};
  const sum = await get(base + 'summary.json');
  out.summary = sum ? JSON.parse(sum) : null;
  if (out.summary) { delete out.summary.eval_config; }
  const pack = a => { const b = new Uint8Array(Math.ceil(a.length / 8)); a.forEach((v, i) => { if (v) b[i >> 3] |= 1 << (i & 7); }); return btoa(String.fromCharCode(...b)); };
  const hash = s => { let h = 5381; for (let i = 0; i < s.length; i++) h = ((h * 33) ^ s.charCodeAt(i)) >>> 0; return h; };
  for (const t of ['mmlu_primary', 'mmlu_overlap', 'gsm8k', 'medmcqa', 'casehold', 'mbpp']) {
    const raw = await get(base + t + '.jsonl');
    if (!raw) { out[t] = null; continue; }
    const rows = raw.trim().split('\n').map(l => JSON.parse(l));
    const o = {n: rows.length, h: hash(rows.map(r => r.item_id).join('|')), c: pack(rows.map(r => +r.correct))};
    if (t === 'gsm8k') o.cs = pack(rows.map(r => +r.correct_strict));
    if (t.startsWith('mmlu')) o.p = rows.map(r => r.pred).join('');
    out[t] = o;
  }
  return JSON.stringify(out);
}
