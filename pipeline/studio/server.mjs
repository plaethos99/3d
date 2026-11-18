/**
 * Character studio: a local web app over the pipeline, so nobody has to see Blender or a terminal.
 *
 *   node pipeline/studio/server.mjs [port]        then open http://localhost:4280
 *
 * Phase 1: a scenario -> a portrait prompt -> sample pictures from Google Flow and Grok Imagine (the signed-in
 * windows of engine/imagegen, run side by side, one picture at a time each) -> pick one.
 * Projects live in pipeline/runs/studio/<id>/ (gitignored: generated faces stay private): project.json and the
 * pictures. Later phases add the face and hair, the model, and the age / outfit controls.
 *
 * API (JSON):
 *   GET  /api/services                  state of flow and grok (closed / signin / ready)
 *   POST /api/services/<svc>/open       open that service's window (to sign in)
 *   POST /api/prompt {scenario}         the portrait prompt for a scenario
 *   GET  /api/projects                  all projects, newest first
 *   POST /api/projects {scenario, prompt, count, services}   new project; its pictures start generating
 *   GET  /api/projects/<id>             one project (pictures with their state)
 *   POST /api/projects/<id>/more {count}  more pictures for it
 *   POST /api/projects/<id>/pick {file}   choose the picture to build from
 *   POST /api/projects/<id>/face {gender, hair, clip}   phase 2: map the face and hair (clip: flow or grok makes
 *                                       the turned views of the face; face_from_image.py --views stills, then
 *                                       pipeline/humgen/photo_character.py), about 15 minutes, as a background job
 *   POST /api/projects/<id>/hair {hair}  another hairstyle on the mapped character (about a minute)
 *   POST /api/projects/<id>/accept      the face and hair are good: on to the model
 *   POST /api/projects/<id>/model       phase 3: make the model the viewer shows (hair cards, baked textures, the
 *                                       face and body sliders as morphs: export_model.py), a background job
 *   POST /api/projects/<id>/apply {keys, height_cm, age, outfit, footwear, colors}   rebuild with those changes
 *                                       (edit_model.py: the rig and clothes refit), then export again
 *   GET  /api/outfits?gender=female     HG's outfits and shoes (with preview pictures)
 *   GET  /hg/<outfits|footwear>/<gender>/<category>/<name>.jpg   an outfit's or shoe's preview
 *   GET  /api/hair?gender=female        HG's hairstyles (with preview pictures)
 *   GET  /hair/<gender>/<category>/<name>.jpg   a hairstyle's preview
 *   GET  /runs/<id>/<path>              a project's file
 * Jobs run one at a time (Blender and the GPU); a project's job and its steps are in project.json ("job"), the
 * full output in job.log.
 */
import { spawn } from 'node:child_process';
import { createReadStream, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const RUNS = path.join(ROOT, 'pipeline', 'runs', 'studio');
const PUBLIC = path.join(HERE, 'public');
const IMAGEGEN = path.join(ROOT, 'engine', 'imagegen', 'imagegen.mjs');
const FACE_FROM_IMAGE = path.join(ROOT, 'pipeline', 'face', 'face_from_image.py');
const PHOTO_CHARACTER = path.join(ROOT, 'pipeline', 'humgen', 'photo_character.py');
const HAIR_DIR = path.join(ROOT, 'tools', 'Human', 'content', 'hair', 'head');
const HG_CONTENT = path.join(ROOT, 'tools', 'Human', 'content');
const BLENDER = path.join(ROOT, 'tools', 'blender-5.2.3-windows-x64', 'blender.exe');
const EXPORT_MODEL = path.join(ROOT, 'pipeline', 'humgen', 'export_model.py');
const EDIT_MODEL = path.join(ROOT, 'pipeline', 'humgen', 'edit_model.py');
const PYTHON = process.env.STUDIO_PYTHON || 'python';
const PORT = Number(process.argv[2] || 4280);
mkdirSync(RUNS, { recursive: true });

// ---------- projects ----------
const projectDir = (id) => path.join(RUNS, id);
const load = (id) => JSON.parse(readFileSync(path.join(projectDir(id), 'project.json'), 'utf8'));
const save = (p) => writeFileSync(path.join(projectDir(p.id), 'project.json'), JSON.stringify(p, null, 1));
const update = (id, fn) => {
  const p = load(id);
  fn(p);
  p.updated = new Date().toISOString();
  save(p);
  return p;
};
const validId = (id) => /^[a-z0-9-]{4,64}$/.test(id) && existsSync(path.join(projectDir(id), 'project.json'));

/** The portrait recipe: front-facing, neutral, evenly lit, plain background: what the face step needs. */
const portraitPrompt = (scenario) =>
  `Photorealistic head-and-shoulders portrait photograph of ${scenario.trim().replace(/[.\s]+$/, '')}. ` +
  'Facing the camera straight on, head level, neutral relaxed expression, mouth closed, eyes open looking into the ' +
  'lens. Soft even studio lighting from the front, sharp focus on the whole face, natural skin texture, ' +
  'hair fully visible, plain light grey background, no text, no watermark.';

const slug = (s) =>
  s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 32) || 'character';

// ---------- generation queues: one per service, one picture at a time ----------
const queues = { flow: [], grok: [] };
const busy = { flow: false, grok: false };

const runImagegen = (args) =>
  new Promise((resolve) => {
    const child = spawn(process.execPath, [IMAGEGEN, ...args], { cwd: ROOT });
    let out = '';
    let err = '';
    child.stdout.on('data', (d) => (out += d));
    child.stderr.on('data', (d) => (err += d));
    child.on('close', () => {
      const line = out.trim().split('\n').filter((l) => l.startsWith('{')).pop();
      try {
        resolve(JSON.parse(line));
      } catch {
        resolve({ ok: false, error: (err || out || 'no answer from imagegen').slice(-400) });
      }
    });
  });

// A face job's head-turn clip drives that service's window (Flow or Grok): pictures must not use it meanwhile
// (they navigate the page and break the clip). held[svc] pauses that service's picture queue during the clip.
const held = { flow: false, grok: false };
const pump = async (svc) => {
  if (busy[svc] || !queues[svc].length || held[svc]) return;
  busy[svc] = true;
  const job = queues[svc].shift();
  update(job.id, (p) => (p.images.find((i) => i.file === job.file).state = 'generating'));
  const started = Date.now();
  const res = await runImagegen(['image', svc, '3:4', path.join(projectDir(job.id), job.file), job.prompt]);
  update(job.id, (p) => {
    const img = p.images.find((i) => i.file === job.file);
    img.state = res.ok ? 'done' : 'failed';
    img.seconds = Math.round((Date.now() - started) / 1000);
    if (!res.ok) img.error = res.error;
  });
  busy[svc] = false;
  pump(svc);
};

const enqueue = (p, count, services) => {
  const svcs = services.filter((s) => s in queues);
  if (!svcs.length) throw new Error('Pick at least one service');
  for (let k = 0; k < count; k++) {
    const svc = svcs[k % svcs.length];
    const n = p.images.length + 1;
    const file = `sample_${String(n).padStart(2, '0')}_${svc}.png`;
    p.images.push({ file, service: svc, state: 'queued' });
    queues[svc].push({ id: p.id, file, prompt: p.prompt });
  }
  save(p);
  svcs.forEach(pump);
};

// ---------- long jobs (face mapping, hairstyle): one at a time ----------
const FACE_STEPS = [
  ['clip', 'Making a head-turn clip (Grok)'], ['scan', 'Tracking the face in every frame'],
  ['fit', 'Measuring the face in 3D'], ['shape', 'Shaping a head'], ['eyes', 'Placing the eyes'],
  ['frames', 'Choosing the best views'], ['masks', 'Finding skin and hair'], ['hairvol', 'Hair volume'],
  ['texture', 'Painting the skin from the frames'], ['check', 'Checking the head against the frames'],
  ['capture', 'Putting the face on the body'], ['bake', 'Baking the face skin'], ['skin', 'Matching the skin tone'],
  ['round1', 'Matching eyes, skin and hair (1/3)'], ['round2', 'Matching eyes, skin and hair (2/3)'],
  ['round3', 'Matching eyes, skin and hair (3/3)'], ['final', 'Building the character'],
];
const HAIR_STEPS = [['final', 'Building the character with the new hairstyle']];
const MODEL_STEPS = [['hair', 'Turning the hair into hair cards'], ['bake', 'Baking skin and clothes textures'],
  ['morphs', 'Adding the face and body sliders'], ['export', 'Saving the 3D model']];
const APPLY_STEPS = [['edit', 'Applying the changes (the clothes refit)'], ...MODEL_STEPS];
/** Which step a line of output starts. */
const stepOf = (line) => {
  let m = line.match(/^== (\w+)/);
  if (m) return m[1];
  m = line.match(/^(capture|bake|skin|final) \(/);
  if (m) return m[1];
  m = line.match(/^eyes\+skin round (\d)/);
  if (m) return `round${m[1]}`;
  if (/^SLIDERS /.test(line)) return 'hair';
  if (/^hair done/.test(line)) return 'bake';
  if (/^bake done/.test(line)) return 'morphs';
  if (/^morphs done/.test(line)) return 'export';
  if (/^EDIT_START/.test(line)) return 'edit';
  return null;
};
const jobQueue = [];
let jobBusy = false;

const runJob = async () => {
  if (jobBusy || !jobQueue.length) return;
  jobBusy = true;
  const { id, commands, clipService } = jobQueue.shift();
  const log = path.join(projectDir(id), 'job.log');
  writeFileSync(log, '');
  if (clipService) {  // hold the clip's service; wait for a picture it is making now
    held[clipService] = true;
    if (busy[clipService]) {
      update(id, (p) => (p.job.waiting = `Waiting for ${clipService === 'flow' ? 'Flow' : 'Grok'} to finish a picture`));
    }
    while (busy[clipService]) await new Promise((r) => setTimeout(r, 2000));
  }
  const releaseGrok = () => {
    if (!clipService || !held[clipService]) return;
    held[clipService] = false;
    pump(clipService);
  };
  update(id, (p) => {
    delete p.job.waiting;
    Object.assign(p.job, { state: 'running', started: new Date().toISOString() });
  });
  let failed = null;
  for (const cmd of commands) {
    const code = await new Promise((resolve) => {
      const child = spawn(cmd[0], cmd.slice(1), { cwd: ROOT, env: { ...process.env, PYTHONUNBUFFERED: '1' } });
      let buf = '';
      child.stdout.on('data', (d) => {
        const text = String(d);
        writeFileSync(log, text, { flag: 'a' });
        buf += text;
        const lines = buf.split(/\r?\n/);
        buf = lines.pop();
        for (const line of lines) {
          const step = stepOf(line);
          if (step && step !== 'clip') releaseGrok(); // the clip is made: Grok is free for pictures again
          if (step) {
            update(id, (p) => {
              p.job.step = step;
              p.job.stepStarted = new Date().toISOString();
            });
          }
        }
      });
      child.stderr.on('data', (d) => writeFileSync(log, String(d), { flag: 'a' }));
      child.on('close', resolve);
      child.on('error', (e) => {
        writeFileSync(log, `\n${e.message}\n`, { flag: 'a' });
        resolve(-1);
      });
    });
    if (code !== 0) {
      const tail = readFileSync(log, 'utf8').trim().split('\n').slice(-6).join('\n');
      failed = tail || `exit code ${code}`;
      break;
    }
  }
  update(id, (p) => {
    p.job.state = failed ? 'failed' : 'done';
    p.job.finished = new Date().toISOString();
    if (failed) p.job.error = failed;
    else if (p.job.kind === 'model' || p.job.kind === 'apply') {
      p.stage = 'model';
      p.modelVersion = (p.modelVersion || 0) + 1; // the viewer reloads the model
    } else {
      p.stage = 'face_ready';
      p.version = (p.version || 0) + 1; // the pictures changed: the page reloads them
    }
  });
  releaseGrok();
  jobBusy = false;
  runJob();
};

const startJob = (id, kind, steps, commands, clipService = null) => {
  const state = load(id).job?.state;
  if (state === 'queued' || state === 'running') throw new Error('This character is already being worked on');
  update(id, (p) => {
    p.job = { kind, state: 'queued', step: null, steps: steps.map(([key, label]) => ({ key, label })),
      queuedBehind: jobQueue.length + (jobBusy ? 1 : 0) };
  });
  jobQueue.push({ id, commands, clipService });
  runJob();
  return load(id);
};

const outfitOptions = (gender) => {
  const out = {};
  for (const kind of ['outfits', 'footwear']) {
    const base = path.join(HG_CONTENT, kind, gender);
    out[kind] = [];
    if (!existsSync(base)) continue;
    for (const cat of readdirSync(base)) {
      const dir = path.join(base, cat);
      if (!statSync(dir).isDirectory()) continue;
      for (const f of readdirSync(dir).filter((x) => x.endsWith('.blend'))) {
        const name = f.slice(0, -6);
        out[kind].push({ id: `${kind}\\${gender}\\${cat}\\${f}`, category: cat, name: name.replace(/^HG_/, '').replace(/_/g, ' '),
          preview: existsSync(path.join(dir, `${name}.jpg`))
            ? `/hg/${kind}/${gender}/${encodeURIComponent(cat)}/${encodeURIComponent(name)}.jpg` : null });
      }
    }
  }
  return out;
};
const faceName = (id) => id.split('-').pop(); // short and safe for HG's slider and skin names
const hairOptions = (gender) => {
  const base = path.join(HAIR_DIR, gender);
  if (!existsSync(base)) return [];
  const out = [];
  for (const cat of readdirSync(base)) {
    const dir = path.join(base, cat);
    if (!statSync(dir).isDirectory()) continue;
    for (const f of readdirSync(dir).filter((x) => x.endsWith('.json'))) {
      const name = f.slice(0, -5);
      const jpg = path.join(dir, `${name}.jpg`);
      out.push({ id: `${cat}\\${name}.json`, category: cat, name,
        preview: existsSync(jpg) ? `/hair/${gender}/${encodeURIComponent(cat)}/${encodeURIComponent(name)}.jpg` : null });
    }
  }
  return out;
};

// projects left mid-generation by a restart: mark their unfinished pictures failed so they can be retried
for (const id of readdirSync(RUNS)) {
  if (!existsSync(path.join(projectDir(id), 'project.json'))) continue;
  update(id, (p) => {
    p.images.forEach((i) => {
      if (i.state === 'queued' || i.state === 'generating') {
        i.state = 'failed';
        i.error = 'The studio was restarted';
      }
    });
    if (p.job && (p.job.state === 'queued' || p.job.state === 'running')) {
      p.job.state = 'failed';
      p.job.error = 'The studio was restarted while this was running';
    }
  });
}

// ---------- http ----------
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.mp4': 'video/mp4', '.json': 'application/json', '.svg': 'image/svg+xml' };

const sendJson = (res, code, body) => {
  res.writeHead(code, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
};
const sendFile = (res, file) => {
  if (!existsSync(file) || !statSync(file).isFile()) return sendJson(res, 404, { error: 'not found' });
  res.writeHead(200, { 'content-type': MIME[path.extname(file).toLowerCase()] || 'application/octet-stream',
    'cache-control': 'no-cache' });
  createReadStream(file).pipe(res);
};
const body = (req) =>
  new Promise((resolve) => {
    let s = '';
    req.on('data', (d) => (s += d));
    req.on('end', () => {
      try {
        resolve(s ? JSON.parse(s) : {});
      } catch {
        resolve({});
      }
    });
  });

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  const parts = url.pathname.split('/').filter(Boolean);
  try {
    if (req.method === 'GET' && (url.pathname === '/' || parts[0] === 'static')) {
      const rel = url.pathname === '/' ? 'index.html' : parts.slice(1).join('/');
      const file = path.resolve(PUBLIC, rel);
      if (!file.startsWith(PUBLIC)) return sendJson(res, 403, { error: 'no' });
      return sendFile(res, file);
    }
    if (req.method === 'GET' && parts[0] === 'hair' && parts.length === 4) {
      const file = path.resolve(HAIR_DIR, ...parts.slice(1).map(decodeURIComponent));
      if (!file.startsWith(HAIR_DIR) || !file.endsWith('.jpg')) return sendJson(res, 403, { error: 'no' });
      return sendFile(res, file);
    }
    if (req.method === 'GET' && parts[0] === 'hg' && parts.length === 5 && ['outfits', 'footwear'].includes(parts[1])) {
      const file = path.resolve(HG_CONTENT, ...parts.slice(1).map(decodeURIComponent));
      if (!file.startsWith(HG_CONTENT) || !file.endsWith('.jpg')) return sendJson(res, 403, { error: 'no' });
      return sendFile(res, file);
    }
    if (req.method === 'GET' && parts[0] === 'runs' && parts.length >= 3 && validId(parts[1])) {
      const file = path.resolve(projectDir(parts[1]), ...parts.slice(2).map(decodeURIComponent));
      if (!file.startsWith(projectDir(parts[1]))) return sendJson(res, 403, { error: 'no' });
      return sendFile(res, file);
    }
    if (parts[0] !== 'api') return sendJson(res, 404, { error: 'not found' });

    if (req.method === 'GET' && parts[1] === 'services') {
      const out = {};
      for (const svc of ['flow', 'grok']) out[svc] = (await runImagegen(['state', svc])).state || 'unknown';
      out.busy = { ...busy, queued: { flow: queues.flow.length, grok: queues.grok.length } };
      return sendJson(res, 200, out);
    }
    if (req.method === 'POST' && parts[1] === 'services' && parts[3] === 'open' && parts[2] in queues) {
      return sendJson(res, 200, await runImagegen(['open', parts[2]]));
    }
    if (req.method === 'GET' && parts[1] === 'outfits') {
      return sendJson(res, 200, outfitOptions(url.searchParams.get('gender') === 'male' ? 'male' : 'female'));
    }
    if (req.method === 'GET' && parts[1] === 'hair') {
      return sendJson(res, 200, hairOptions(url.searchParams.get('gender') === 'male' ? 'male' : 'female'));
    }
    if (req.method === 'POST' && parts[1] === 'prompt') {
      const { scenario = '' } = await body(req);
      if (!scenario.trim()) return sendJson(res, 400, { error: 'Describe the character first' });
      return sendJson(res, 200, { prompt: portraitPrompt(scenario) });
    }
    if (parts[1] === 'projects' && parts.length === 2) {
      if (req.method === 'GET') {
        const list = readdirSync(RUNS).filter(validId).map(load)
          .sort((a, b) => (b.created || '').localeCompare(a.created || ''));
        return sendJson(res, 200, list);
      }
      if (req.method === 'POST') {
        const b = await body(req);
        if (!b.scenario?.trim()) return sendJson(res, 400, { error: 'Describe the character first' });
        const id = `${slug(b.scenario)}-${Date.now().toString(36)}`;
        mkdirSync(projectDir(id), { recursive: true });
        const p = { id, scenario: b.scenario.trim(), prompt: (b.prompt || portraitPrompt(b.scenario)).trim(),
          created: new Date().toISOString(), stage: 'pick', images: [], picked: null };
        enqueue(p, Math.min(Math.max(Number(b.count) || 4, 1), 8), b.services || ['flow', 'grok']);
        return sendJson(res, 200, load(id));
      }
    }
    if (parts[1] === 'projects' && parts.length >= 3) {
      const id = parts[2];
      if (!validId(id)) return sendJson(res, 404, { error: 'No such project' });
      if (req.method === 'GET' && parts.length === 3) return sendJson(res, 200, load(id));
      if (req.method === 'POST' && parts[3] === 'more') {
        const b = await body(req);
        const p = load(id);
        if (b.prompt?.trim()) p.prompt = b.prompt.trim();
        enqueue(p, Math.min(Math.max(Number(b.count) || 4, 1), 8), b.services || ['flow', 'grok']);
        return sendJson(res, 200, load(id));
      }
      if (req.method === 'POST' && parts[3] === 'face') {
        const b = await body(req);
        const p = load(id);
        if (!p.picked) return sendJson(res, 400, { error: 'Pick a picture first' });
        const gender = b.gender === 'male' ? 'male' : 'female';
        const options = hairOptions(gender);
        const hair = options.some((h) => h.id === b.hair) ? b.hair
          : (gender === 'female' ? 'Long\\Medium Center Part.json' : options[0]?.id);
        const clip = b.clip === 'grok' ? 'grok' : 'flow';
        update(id, (q) => Object.assign(q, { gender, hair, clip }));
        const dir = projectDir(id);
        const photo = path.join(dir, p.picked);
        const by = clip === 'flow' ? 'Flow' : 'Grok';
        const steps = FACE_STEPS.map(([k, l]) => [k, k === 'clip' ? `Making turned views of the face (${by})` : l]);
        return sendJson(res, 200, startJob(id, 'face', steps, [
          // turned stills (sharper than a clip's frames), made by Flow or Grok from the picked picture
          [PYTHON, FACE_FROM_IMAGE, photo, path.join(dir, 'face_run'), '--views', 'stills', '--clip', clip],
          [PYTHON, PHOTO_CHARACTER, path.join(dir, 'face_run'), path.join(dir, 'character'), '--name', faceName(id),
            '--gender', gender, '--photo', photo, '--hair-style', hair],
        ], clip));
      }
      if (req.method === 'POST' && parts[3] === 'hair') {
        const { hair } = await body(req);
        const p = load(id);
        if (p.stage !== 'face_ready' && p.stage !== 'face_done') return sendJson(res, 400, { error: 'Map the face first' });
        if (!hairOptions(p.gender).some((h) => h.id === hair)) return sendJson(res, 400, { error: 'Unknown hairstyle' });
        update(id, (q) => (q.hair = hair));
        const dir = projectDir(id);
        return sendJson(res, 200, startJob(id, 'hair', HAIR_STEPS, [
          [PYTHON, PHOTO_CHARACTER, path.join(dir, 'face_run'), path.join(dir, 'character'), '--name', faceName(id),
            '--gender', p.gender, '--photo', path.join(dir, p.picked), '--hair-style', hair, '--from', 'final'],
        ]));
      }
      if (req.method === 'POST' && parts[3] === 'model') {
        const p = load(id);
        if (!['face_done', 'model'].includes(p.stage)) return sendJson(res, 400, { error: 'Accept the face and hair first' });
        const dir = projectDir(id);
        const blend = path.join(dir, 'model', 'human.blend');
        const src = existsSync(blend) ? blend : path.join(dir, 'character', 'final', 'human.blend');
        return sendJson(res, 200, startJob(id, 'model', MODEL_STEPS, [
          [BLENDER, '-b', src, '-P', EXPORT_MODEL, '--', path.join(dir, 'model'), '--tex', '1024'],
        ]));
      }
      if (req.method === 'POST' && parts[3] === 'apply') {
        const b = await body(req);
        const p = load(id);
        if (p.stage !== 'model') return sendJson(res, 400, { error: 'Make the model first' });
        const dir = projectDir(id);
        const blend = path.join(dir, 'model', 'human.blend');
        const src = existsSync(blend) ? blend : path.join(dir, 'character', 'final', 'human.blend');
        const changes = {};
        if (b.keys && Object.keys(b.keys).length) changes.keys = b.keys;
        for (const k of ['height_cm', 'age']) if (Number.isFinite(b[k])) changes[k] = b[k];
        const opts = outfitOptions(p.gender || 'female');
        if (b.outfit && opts.outfits.some((o) => o.id === b.outfit)) changes.outfit = b.outfit;
        if (b.footwear && opts.footwear.some((o) => o.id === b.footwear)) changes.footwear = b.footwear;
        if (b.colors && typeof b.colors === 'object') changes.colors = b.colors;
        writeFileSync(path.join(dir, 'model', 'changes.json'), JSON.stringify(changes, null, 1));
        return sendJson(res, 200, startJob(id, 'apply', APPLY_STEPS, [
          [PYTHON, '-c', 'print("EDIT_START")'],
          [BLENDER, '-b', src, '-P', EDIT_MODEL, '--', path.join(dir, 'model', 'changes.json'), blend],
          [BLENDER, '-b', blend, '-P', EXPORT_MODEL, '--', path.join(dir, 'model'), '--tex', '1024'],
        ]));
      }
      if (req.method === 'POST' && parts[3] === 'accept') {
        const p = load(id);
        if (p.stage !== 'face_ready') return sendJson(res, 400, { error: 'The face and hair are not ready yet' });
        return sendJson(res, 200, update(id, (q) => (q.stage = 'face_done')));
      }
      if (req.method === 'POST' && parts[3] === 'pick') {
        const { file } = await body(req);
        const p = load(id);
        if (!p.images.some((i) => i.file === file && i.state === 'done')) {
          return sendJson(res, 400, { error: 'That picture is not ready' });
        }
        return sendJson(res, 200, update(id, (q) => {
          q.picked = file;
          q.stage = 'picked';
        }));
      }
    }
    return sendJson(res, 404, { error: 'not found' });
  } catch (e) {
    return sendJson(res, 500, { error: String(e.message || e) });
  }
});

server.listen(PORT, '127.0.0.1', () => console.log(`STUDIO_OK http://localhost:${PORT}`));
