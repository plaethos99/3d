// Character Studio: the Daz cast (Genesis 9). Face and skin are made in Daz, the hair is grown in Blender; the sliders
// write the character's char.json ("face_shape", "blender_hair") and rebuild it (server: daz.mjs).
const el = (id) => document.getElementById(id);
const call = async (path, body) => {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
};

let info = null;   // /api/daz/cast
let open = null;   // the character on the page
let edits = {};    // slider values on the page
let polling = null;

const fmt = (v, unit) => `${(+v).toFixed(Math.abs(v) < 10 && !Number.isInteger(+v) ? 2 : 0).replace(/\.?0+$/, '') || 0}${unit ? ' ' + unit : ''}`;

const slider = (box, key, [label, min, max, step, unit], value, def, group) => {
  const wrap = document.createElement('label');
  wrap.className = 'slider';
  wrap.innerHTML = `<span>${label}<b></b></span><input type="range" min="${min}" max="${max}" step="${step}">`;
  const input = wrap.querySelector('input');
  const out = wrap.querySelector('b');
  const set = (v) => {
    input.value = v;
    out.textContent = fmt(v, unit);
    wrap.classList.toggle('changed', Math.abs(+v - def) > 1e-9);
  };
  set(value);
  input.addEventListener('input', () => {
    set(input.value);
    edits[group][key] = +input.value;
    note();
  });
  box.appendChild(wrap);
};

const note = () => {
  if (!open) return;
  const f = JSON.stringify(edits.face) !== JSON.stringify(faceValues(open));
  const h = JSON.stringify(edits.hair) !== JSON.stringify(hairValues(open));
  el('dazNote').textContent = f ? 'Face changed: rebuild the face to see it.' : h ? 'Hair changed: regrow the hair to see it.' : '';
};

const faceValues = (c) => Object.fromEntries(Object.keys(info.face_shape).map((k) => [k, c.face_shape[k] ?? (k === 'strength' ? 1 : 0)]));
const hairValues = (c) => ({
  ...Object.fromEntries(Object.keys(info.hair).map((k) => [k, c.blender_hair[k] ?? c.hair_defaults[k]])),
  colour: c.blender_hair.colour || null,
});

const pictures = (c) => {
  const box = el('dazPics');
  box.innerHTML = '';
  for (const [w, label] of [['portrait', 'Portrait'], ['face', 'Front'], ['face34', 'Three-quarter'], ['body', 'Body']]) {
    const f = document.createElement('figure');
    f.innerHTML = c.pictures[w] ? `<img src="${c.pictures[w]}" alt="${label}"><figcaption>${label}</figcaption>`
      : `<div class="card wait" style="aspect-ratio:4/5">Not rendered yet</div><figcaption>${label}</figcaption>`;
    if (c.pictures[w]) f.addEventListener('click', () => { el('viewerImg').src = c.pictures[w]; el('viewer').hidden = false; });
    box.appendChild(f);
  }
};

const page = (c) => {
  open = c;
  edits = { face: faceValues(c), hair: hairValues(c) };
  el('dazTitle').textContent = c.title;
  pictures(c);
  const face = el('dazFace');
  face.innerHTML = '';
  for (const [k, lim] of Object.entries(info.face_shape)) slider(face, k, lim, edits.face[k], k === 'strength' ? 1 : 0, 'face');
  const hair = el('dazHair');
  hair.innerHTML = '';
  for (const [k, lim] of Object.entries(info.hair)) slider(hair, k, lim, edits.hair[k], c.hair_defaults[k], 'hair');
  const portrait = c.portrait_hair || '#2a2a2a';
  el('dazColour').value = edits.hair.colour || portrait;
  el('dazColourVal').textContent = edits.hair.colour || `${portrait} (portrait)`;
  el('dazColourPortrait').checked = !edits.hair.colour;
  note();
};

el('dazColour').addEventListener('input', () => {
  edits.hair.colour = el('dazColour').value;
  el('dazColourVal').textContent = edits.hair.colour;
  el('dazColourPortrait').checked = false;
  note();
});
el('dazColourPortrait').addEventListener('change', () => {
  if (el('dazColourPortrait').checked) {
    edits.hair.colour = null;
    el('dazColour').value = open?.portrait_hair || '#2a2a2a';
    el('dazColourVal').textContent = `${el('dazColour').value} (portrait)`;
  }
  note();
});

const showPage = (name) => {
  // the studio's own views off (app.js), this one on
  for (const v of ['viewScenario', 'viewPick', 'viewFace', 'viewModel']) el(v).hidden = true;
  for (const b of ['applyBar', 'pickbar']) if (el(b)) el(b).hidden = true;
  document.querySelectorAll('#projects li').forEach((li) => li.classList.remove('on'));
  document.querySelectorAll('#dazCast li').forEach((li) => li.classList.toggle('on', li.dataset.name === name));
  el('viewDaz').hidden = false;
  page(info.cast.find((c) => c.name === name));
  job(info.job);
};

const list = () => {
  const ul = el('dazCast');
  ul.innerHTML = '';
  for (const c of info.cast) {
    const li = document.createElement('li');
    li.dataset.name = c.name;
    li.innerHTML = `<b>${c.title}</b><span class="hint">${c.sex}${info.job?.state === 'running' && info.job.name === c.name ? ' · rebuilding' : ''}</span>`;
    li.addEventListener('click', () => showPage(c.name));
    if (open && open.name === c.name && !el('viewDaz').hidden) li.classList.add('on');
    ul.appendChild(li);
  }
};

const load = async () => {
  info = await call('/api/daz/cast');
  list();
};

const job = (j) => {
  const running = j && j.state === 'running';
  el('dazFaceBtn').disabled = el('dazHairBtn').disabled = !!running;
  if (!j || !j.name || !open || j.name !== open.name) {
    el('dazJob').hidden = true;
    return;
  }
  el('dazJob').hidden = false;
  el('dazJobState').textContent = running ? `Rebuilding ${open.title} (from the ${j.from === 'blender' ? 'hair' : 'face'})…`
    : j.state === 'done' ? 'Rebuilt.' : 'The rebuild failed: see the last lines.';
  el('dazJobSteps').textContent = j.steps?.length ? `steps: ${[...new Set(j.steps)].join(' → ')}` : '';
  el('dazJobLog').textContent = (j.lines || []).join('\n');
};

const watch = () => {
  clearInterval(polling);
  polling = setInterval(async () => {
    const j = await call('/api/daz/job').catch(() => null);
    job(j);
    if (!j || j.state !== 'running') {
      clearInterval(polling);
      await load();
      if (open) {
        const c = info.cast.find((x) => x.name === open.name);
        if (c) { open = c; pictures(c); }
      }
    }
  }, 3000);
};

const rebuild = async (from) => {
  if (!open) return;
  const hair = { ...edits.hair };
  if (!hair.colour) delete hair.colour;
  try {
    const r = await call(`/api/daz/cast/${open.name}/apply`, { face_shape: edits.face, blender_hair: hair, from });
    open.face_shape = { ...edits.face };
    open.blender_hair = { ...edits.hair };
    note();
    job(r.job);
    watch();
  } catch (e) {
    el('dazNote').textContent = e.message;
  }
};
el('dazFaceBtn').addEventListener('click', () => rebuild('face'));
el('dazHairBtn').addEventListener('click', () => rebuild('blender'));
el('dazResetBtn').addEventListener('click', () => open && page(open));

load().then(() => { if (info.job?.state === 'running') watch(); }).catch((e) => console.error('Daz cast', e));
