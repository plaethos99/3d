/**
 * The studio's Daz cast: Genesis 9 characters made by pipeline/daz/make_daz_character.py (face and skin in Daz,
 * hair grown in Blender). Each character is a folder <cast>/<name>/daz/ with char.json (what it is made of: its
 * "face_shape" and "blender_hair" are what the studio edits), face_run/front.png (the portrait) and build.json.
 * The cast folder is DAZ_CAST, or ../game001/characters next to the repo.
 *
 *   GET  /api/daz/cast                    the characters: name, sex, face_shape, blender_hair, pictures, the job
 *   GET  /daz/<name>/<portrait|face|face34|body>   a picture (portrait; the latest Blender renders)
 *   POST /api/daz/cast/<name>/apply {face_shape, blender_hair, from}   save into char.json and rebuild: from
 *        "face" (the face shape changed: new Daz morph, skin, Daz, Blender: about 8 minutes) or "blender" (only the
 *        hair: about 3 minutes)
 *   GET  /api/daz/job                     the running or last job: state, steps done, the last lines
 * One rebuild at a time (Daz and Blender).
 */
import { spawn } from 'node:child_process';
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';

export const FACE_SHAPE = {  // the Face sliders: key -> [label, min, max, step, unit]
  strength: ['Face strength (the fit)', 0.5, 1.3, 0.05, '×'],
  jaw_width: ['Jaw width', -0.5, 0.3, 0.05, ''],
  chin_width: ['Chin width', -0.5, 0.5, 0.05, ''],
  chin_length: ['Chin length', -1, 1, 0.1, 'cm'],
  chin_forward: ['Chin forward', -1, 1, 0.1, 'cm'],
  mouth_width: ['Mouth width', -0.3, 0.3, 0.02, ''],
  lip_fullness: ['Lip fullness', -0.4, 0.6, 0.05, ''],
  nose_width: ['Nose width', -0.3, 0.3, 0.02, ''],
};
export const HAIR = {  // the Hair sliders (blender_hair.py options)
  length_cm: ['Length', 3, 45, 1, 'cm'],
  volume: ['Volume', 0, 1.5, 0.05, ''],
  messy: ['Messy', 0, 1.2, 0.05, ''],
  clump: ['Locks (clumping)', 0, 1, 0.05, ''],
  flick: ['Tips flicked up', 0, 0.8, 0.05, ''],
  wave_mm: ['Wave', 0, 8, 0.5, 'mm'],
  shade: ['Shade (darker ← → lighter)', 0.3, 1.5, 0.05, '×'],
  'cut_cm.front': ['Fringe ends (above the eyes)', -6, 6, 0.5, 'cm'],
  'cut_cm.side': ['Sides end (above the eyes)', -20, 4, 0.5, 'cm'],
  'cut_cm.back': ['Back ends (above the eyes)', -25, 0, 0.5, 'cm'],
};

// blender_hair.py's defaults (keep in step with DEFAULTS there): what the Hair sliders start at
export const HAIR_DEFAULTS = {
  male: { length_cm: 11, volume: 0.9, messy: 0.7, clump: 0.75, flick: 0.3, wave_mm: 2, shade: 0.85,
    'cut_cm.front': 0.6, 'cut_cm.side': -1.5, 'cut_cm.back': -7 },
  female: { length_cm: 32, volume: 0.5, messy: 0.25, clump: 0.6, flick: 0.1, wave_mm: 3, shade: 0.7,
    'cut_cm.front': 0, 'cut_cm.side': -14, 'cut_cm.back': -18 },
};

export function dazStudio(ROOT, PYTHON) {
  const CAST = process.env.DAZ_CAST || path.resolve(ROOT, '..', 'game001', 'characters');
  const MAKE = path.join(ROOT, 'pipeline', 'daz', 'make_daz_character.py');
  const RENDERS = path.join(ROOT, 'tests', 'blender_daz');
  let job = null;  // {name, from, state, started, finished, lines, steps}

  const dir = (name) => path.join(CAST, name, 'daz');
  const valid = (name) => /^[a-z0-9_-]{1,40}$/.test(name) && existsSync(path.join(dir(name), 'char.json'));
  const title = (name) => name.charAt(0).toUpperCase() + name.slice(1);
  const picture = (name, which) => which === 'portrait' ? path.join(dir(name), 'face_run', 'front.png')
    : path.join(RENDERS, `${name}_bl_${which}.png`);

  const cast = () => (existsSync(CAST) ? readdirSync(CAST) : []).filter(valid).map((name) => {
    const c = JSON.parse(readFileSync(path.join(dir(name), 'char.json'), 'utf8'));
    const pics = {};
    for (const w of ['portrait', 'face', 'face34', 'body']) {
      const f = picture(name, w);
      if (existsSync(f)) pics[w] = `/daz/${name}/${w}?t=${Math.round(statSync(f).mtimeMs)}`;
    }
    const shape = { ...(c.face_shape || {}) };
    if ('lower_face_narrow' in shape) { shape.jaw_width ??= -shape.lower_face_narrow; delete shape.lower_face_narrow; }
    const hj = path.join(dir(name), 'face_run', 'masks', 'hair.json');
    const rgb = existsSync(hj) ? JSON.parse(readFileSync(hj, 'utf8')).hair_rgb : null;
    const flat = {};
    for (const [k, v] of Object.entries(c.blender_hair || {})) {
      if (v && typeof v === 'object') for (const [s, x] of Object.entries(v)) flat[`${k}.${s}`] = x; else flat[k] = v;
    }
    return { name, title: title(name), sex: c.sex || 'female', face_shape: shape, blender_hair: flat,
      hair_defaults: HAIR_DEFAULTS[c.sex === 'male' ? 'male' : 'female'],
      portrait_hair: rgb && '#' + rgb.map((x) => Math.round(x).toString(16).padStart(2, '0')).join(''), pictures: pics };
  });

  const start = (name, from) => {
    const args = [MAKE, dir(name), title(name), '--from', from, '--rounds', '0'];
    job = { name, from, state: 'running', started: new Date().toISOString(), lines: [], steps: [] };
    const p = spawn(PYTHON, args, { cwd: ROOT, windowsHide: true });
    const take = (buf) => {
      for (const ln of buf.toString().split(/\r?\n/)) {
        if (!ln.trim()) continue;
        job.lines.push(ln);
        if (job.lines.length > 200) job.lines.shift();
        const m = ln.match(/^(flame|fb|face|skin|daz|blender)\b/);
        if (m) job.steps.push(m[1]);
      }
    };
    p.stdout.on('data', take);
    p.stderr.on('data', take);
    p.on('close', (code) => {
      job.finished = new Date().toISOString();
      job.state = code === 0 && job.lines.some((l) => l.includes('MAKE_DAZ_CHARACTER_OK')) ? 'done' : 'failed';
    });
  };

  /** Handles the Daz routes; returns true when it did. */
  return async function route(req, res, url, parts, { sendJson, sendFile, body }) {
    if (req.method === 'GET' && parts[0] === 'daz' && parts.length === 3) {
      const [, name, which] = parts;
      if (!valid(name) || !['portrait', 'face', 'face34', 'body'].includes(which)) return sendJson(res, 404, { error: 'no' }), true;
      const f = picture(name, which);
      if (!existsSync(f)) return sendJson(res, 404, { error: 'not rendered yet' }), true;
      return sendFile(res, f), true;
    }
    if (parts[0] !== 'api' || parts[1] !== 'daz') return false;
    if (req.method === 'GET' && parts[2] === 'cast') {
      return sendJson(res, 200, { cast: cast(), face_shape: FACE_SHAPE, hair: HAIR, job: job && { ...job, lines: job.lines.slice(-12) } }), true;
    }
    if (req.method === 'GET' && parts[2] === 'job') return sendJson(res, 200, job ? { ...job, lines: job.lines.slice(-30) } : {}), true;
    if (req.method === 'POST' && parts[2] === 'cast' && parts[4] === 'apply') {
      const name = parts[3];
      if (!valid(name)) return sendJson(res, 404, { error: 'no such character' }), true;
      if (job && job.state === 'running') return sendJson(res, 409, { error: `${title(job.name)} is still being rebuilt` }), true;
      const b = await body(req);
      const file = path.join(dir(name), 'char.json');
      const c = JSON.parse(readFileSync(file, 'utf8'));
      if (b.face_shape) {
        const s = {};
        for (const [k, v] of Object.entries(b.face_shape)) {
          const lim = FACE_SHAPE[k];
          const def = k === 'strength' ? 1 : 0;
          if (lim && Number.isFinite(+v) && +v !== def) s[k] = Math.min(lim[2], Math.max(lim[1], +v));
        }
        c.face_shape = s;
      }
      if (b.blender_hair) {
        const h = {};
        for (const [k, v] of Object.entries(b.blender_hair)) {
          if (k === 'colour' && /^#[0-9a-f]{6}$/i.test(v)) { h.colour = v; continue; }
          const lim = HAIR[k];
          if (!lim || !Number.isFinite(+v)) continue;
          const val = Math.min(lim[2], Math.max(lim[1], +v));
          if (k.includes('.')) { const [a, s] = k.split('.'); (h[a] ||= {})[s] = val; } else h[k] = val;
        }
        c.blender_hair = h;
      }
      writeFileSync(file, JSON.stringify(c, null, 1));
      start(name, b.from === 'blender' ? 'blender' : 'face');
      return sendJson(res, 200, { ok: true, job }), true;
    }
    return sendJson(res, 404, { error: 'not found' }), true;
  };
}
