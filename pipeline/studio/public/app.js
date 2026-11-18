// Character Studio front end: scenario -> prompt -> sample pictures -> pick (phase 1).
const $ = (id) => document.getElementById(id);
const api = async (path, body) => {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
};

let current = null; // the open project
let chosen = null; // the picture clicked in the gallery
let poll = null;

const STEP_ORDER = ['scenario', 'pick', 'face', 'model', 'play'];
const setStep = (step) => {
  const at = STEP_ORDER.indexOf(step);
  document.querySelectorAll('#steps li').forEach((li) => {
    const i = STEP_ORDER.indexOf(li.dataset.step);
    li.classList.toggle('on', i === at);
    li.classList.toggle('done', i < at);
  });
};

const show = (v) => {
  $('viewScenario').hidden = v !== 'scenario';
  $('viewPick').hidden = v !== 'pick';
  $('viewFace').hidden = v !== 'face';
  $('viewModel').hidden = v !== 'model';
  $('applyBar').hidden = true;
  $('pickbar').hidden = v !== 'pick' || !(chosen || current?.picked);
  setStep(v === 'scenario' ? 'scenario' : v === 'pick' ? 'pick' : v === 'model' ? 'model'
    : current?.stage === 'face_done' ? 'model' : 'face');
};
let view = 'scenario';
const go = (v) => {
  view = v;
  show(v);
  if (v === 'pick') renderPick();
  if (v === 'face') renderFace();
  if (v === 'model') renderModel();
};

// ---------- services ----------
const refreshServices = async () => {
  try {
    const s = await api('/api/services');
    document.querySelectorAll('.svc').forEach((el) => {
      const st = s[el.dataset.svc];
      el.className = `svc ${st}`;
      el.title = { ready: 'Signed in and ready', signin: 'Needs you to sign in: click to open its window',
        closed: 'Window closed: click to open it' }[st] || st;
    });
  } catch { /* the server will say on the next try */ }
};
document.querySelectorAll('.svc').forEach((el) => el.addEventListener('click', async () => {
  if (el.classList.contains('ready')) return;
  await api(`/api/services/${el.dataset.svc}/open`, {});
  setTimeout(refreshServices, 3000);
}));

// ---------- projects list ----------
const refreshProjects = async () => {
  const list = await api('/api/projects');
  $('projects').innerHTML = '';
  for (const p of list) {
    const li = document.createElement('li');
    li.classList.toggle('on', p.id === current?.id);
    const thumb = p.picked || p.images.find((i) => i.state === 'done')?.file;
    li.innerHTML = thumb ? `<img src="/runs/${p.id}/${thumb}" alt="">` : '<span class="ph"></span>';
    const t = document.createElement('span');
    t.className = 't';
    t.textContent = p.scenario;
    li.append(t);
    li.addEventListener('click', () => openProject(p.id));
    $('projects').append(li);
  }
};

// ---------- scenario ----------
$('promptBtn').addEventListener('click', async () => {
  $('scenarioError').textContent = '';
  try {
    const { prompt } = await api('/api/prompt', { scenario: $('scenario').value });
    $('prompt').value = prompt;
    $('promptField').hidden = false;
    $('genRow').hidden = false;
  } catch (e) {
    $('scenarioError').textContent = e.message;
  }
});

const services = () => [$('useFlow').checked && 'flow', $('useGrok').checked && 'grok'].filter(Boolean);

$('genBtn').addEventListener('click', async () => {
  $('scenarioError').textContent = '';
  $('genBtn').disabled = true;
  try {
    const p = await api('/api/projects', { scenario: $('scenario').value, prompt: $('prompt').value,
      count: Number($('count').value), services: services() });
    await openProject(p.id);
  } catch (e) {
    $('scenarioError').textContent = e.message;
  } finally {
    $('genBtn').disabled = false;
  }
});

$('newBtn').addEventListener('click', () => {
  current = null;
  view = 'scenario';
  clearInterval(poll);
  $('scenario').value = '';
  $('prompt').value = '';
  $('promptField').hidden = true;
  $('genRow').hidden = true;
  show('scenario');
  refreshProjects();
});

// ---------- pick ----------
const renderPick = () => {
  const p = current;
  $('pickScenario').textContent = p.scenario;
  if (document.activeElement !== $('pickPrompt')) $('pickPrompt').value = p.prompt;
  const gallery = $('gallery');
  gallery.innerHTML = '';
  for (const img of p.images) {
    const card = document.createElement('div');
    if (img.state === 'done') {
      card.className = 'card' + ((chosen || p.picked) === img.file ? ' on' : '');
      card.innerHTML = `<img src="/runs/${p.id}/${img.file}" alt="Sample picture" loading="lazy">
        <span class="tag">${img.service}</span><button class="zoom" title="View large">⤢</button>`;
      card.addEventListener('click', (e) => {
        if (e.target.classList.contains('zoom')) {
          $('viewerImg').src = `/runs/${p.id}/${img.file}`;
          $('viewer').hidden = false;
          return;
        }
        chosen = img.file;
        renderPick();
      });
    } else if (img.state === 'failed') {
      card.className = 'card failed';
      card.textContent = `${img.service}: ${img.error || 'failed'}`;
    } else {
      card.className = 'card wait';
      card.innerHTML = `<div><div class="spinner"></div>${img.state === 'generating'
        ? `Making a picture with ${img.service}…` : `Waiting for ${img.service}…`}</div>`;
    }
    gallery.append(card);
  }
  const sel = chosen || p.picked;
  $('pickbar').hidden = !sel || view !== 'pick';
  if (sel) {
    $('pickThumb').src = `/runs/${p.id}/${sel}`;
    $('useBtn').textContent = p.picked === sel ? 'Next: face & hair →' : 'Use this picture';
    $('useBtn').disabled = false;
  }
  const pending = p.images.filter((i) => i.state === 'queued' || i.state === 'generating').length;
  $('pickTitle').textContent = pending ? `Making pictures (${pending} to go)` : 'Pick a picture';
  setStep('pick');
};

const openProject = async (id) => {
  clearInterval(poll);
  current = await api(`/api/projects/${id}`);
  chosen = null;
  hairPick = null;
  go(current.stage === 'model' || ['model', 'apply'].includes(current.job?.kind) ? 'model'
    : current.job || (current.stage || '').startsWith('face') ? 'face' : 'pick');
  refreshProjects();
  poll = setInterval(async () => {
    const was = JSON.stringify([current.images, current.job, current.stage, current.version, current.modelVersion]);
    current = await api(`/api/projects/${id}`);
    if (JSON.stringify([current.images, current.job, current.stage, current.version, current.modelVersion]) !== was) {
      if (view === 'pick') renderPick();
      if (view === 'face') renderFace();
      if (view === 'model') renderModel();
      refreshProjects();
    }
    if (view === 'face' && current.job?.state === 'running') renderTimes();
    if (view === 'model' && current.job?.state === 'running') {
      renderJobList(current.job, $('modelProgressHead'), $('modelProgressSteps'));
    }
  }, 2000);
};

$('moreBtn').addEventListener('click', async () => {
  $('pickError').textContent = '';
  try {
    current = await api(`/api/projects/${current.id}/more`, { count: 4, prompt: $('pickPrompt').value,
      services: services() });
    renderPick();
  } catch (e) {
    $('pickError').textContent = e.message;
  }
});

$('useBtn').addEventListener('click', async () => {
  $('pickError').textContent = '';
  try {
    const file = chosen || current.picked;
    if (file !== current.picked) current = await api(`/api/projects/${current.id}/pick`, { file });
    chosen = null;
    refreshProjects();
    go('face');
  } catch (e) {
    $('pickError').textContent = e.message;
  }
});

$('viewer').addEventListener('click', () => ($('viewer').hidden = true));

// ---------- face & hair ----------
let hairPick = null; // hairstyle clicked in a grid (not yet applied)
const hairCache = {};
const hairList = async (gender) => (hairCache[gender] ??= await api(`/api/hair?gender=${gender}`));
const guessGender = (text) => (/\b(man|boy|male|he|his|father|husband|son|king|gentleman)\b/i.test(text)
  && !/\b(woman|girl|female|she|her|mother|wife|daughter|queen|lady)\b/i.test(text) ? 'male' : 'female');

const renderHairs = async (box, gender, selected, applied) => {
  const list = await hairList(gender);
  box.innerHTML = '';
  for (const h of list) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'hair' + (h.id === selected ? ' on' : '') + (h.id === applied ? ' current' : '');
    b.title = `${h.category}: ${h.name}`;
    b.innerHTML = (h.preview ? `<img src="${h.preview}" alt="" loading="lazy">` : '<img alt="">') +
      `<span>${h.name}</span>`;
    b.addEventListener('click', () => {
      hairPick = h.id;
      renderFace();
    });
    box.append(b);
  }
};

const fmt = (sec) => (sec < 60 ? `${Math.round(sec)} s` : `${Math.floor(sec / 60)} min ${Math.round(sec % 60)} s`);

const renderTimes = () => {
  const j = current?.job;
  if (!j || j.state !== 'running') return;
  const now = Date.now();
  const li = document.querySelector('#progressSteps li.now .time');
  if (li && j.stepStarted) li.textContent = fmt((now - Date.parse(j.stepStarted)) / 1000);
  if (j.started && !j.waiting) {
    $('progressHead').textContent = `${j.kind === 'hair' ? 'Changing the hairstyle' : 'Mapping the face and hair'} · `
      + `${fmt((now - Date.parse(j.started)) / 1000)} so far`;
  }
};

const renderFace = async () => {
  const p = current;
  if (!p) return;
  $('faceScenario').textContent = p.scenario;
  const j = p.job;
  const running = j && (j.state === 'queued' || j.state === 'running');
  const ready = p.stage === 'face_ready' || p.stage === 'face_done';
  $('faceSetup').hidden = running || ready || j?.state === 'failed';
  $('faceProgress').hidden = !running;
  $('faceResult').hidden = running || !ready;
  $('faceFailed').hidden = !(j?.state === 'failed') || running;
  setStep(p.stage === 'face_done' ? 'model' : 'face');

  if (!$('faceSetup').hidden) {
    $('facePicked').src = `/runs/${p.id}/${p.picked}`;
    if (!$('gender').dataset.for || $('gender').dataset.for !== p.id) {
      $('gender').value = p.gender || guessGender(p.scenario);
      $('gender').dataset.for = p.id;
    }
    const sel = hairPick || p.hair || null;
    $('hairChosenName').textContent = sel ? `· ${sel.replace(/\.json$/, '').replace('\\', ' / ')}` : '· pick one (or keep the default)';
    await renderHairs($('hairsSetup'), $('gender').value, sel, null);
  }
  if (running) {
    const at = j.steps.findIndex((s) => s.key === j.step);
    $('progressSteps').innerHTML = j.steps.map((s, i) => {
      const cls = i < at ? 'done' : i === at ? 'now' : '';
      return `<li class="${cls}"><span class="dot">${i < at ? '✓' : ''}</span>${s.label}<span class="time"></span></li>`;
    }).join('');
    if (j.waiting) $('progressHead').textContent = `${j.waiting}…`;
    else if (j.state === 'queued') {
      $('progressHead').textContent = j.queuedBehind ? `Waiting for another character to finish (${j.queuedBehind} ahead)…` : 'Starting…';
    }
    renderTimes();
  }
  if (!$('faceResult').hidden) {
    $('compareImg').src = `/runs/${p.id}/character/compare.jpg?v=${p.version || 0}`;
    const sel = hairPick || p.hair;
    await renderHairs($('hairsResult'), p.gender, sel, p.hair);
    $('hairBtn').disabled = !hairPick || hairPick === p.hair;
    $('hairBtn').textContent = hairPick && hairPick !== p.hair
      ? `Apply “${hairPick.replace(/\.json$/, '').split('\\').pop()}”` : 'Apply hairstyle';
    $('acceptBtn').disabled = ['face_done', 'model'].includes(p.stage);
    $('acceptBtn').textContent = $('acceptBtn').disabled ? 'Accepted ✓' : 'Accept face & hair';
    $('acceptNote').textContent = '';
    $('toModelRow').hidden = !['face_done', 'model'].includes(p.stage);
  }
  if (!$('faceFailed').hidden) {
    $('faceErrorText').textContent = `It stopped: ${j.error || 'unknown error'}`;
  }
};

$('gender').addEventListener('change', () => {
  hairPick = null;
  renderFace();
});
$('backToPick').addEventListener('click', () => go('pick'));

const startFace = async () => {
  $('faceError').textContent = '';
  try {
    current = await api(`/api/projects/${current.id}/face`, { gender: $('gender').value, hair: hairPick || current.hair,
      clip: $('clipService').value });
    hairPick = null;
    renderFace();
  } catch (e) {
    $('faceError').textContent = e.message;
  }
};
$('faceBtn').addEventListener('click', startFace);
$('retryBtn').addEventListener('click', async () => {
  // run the same job again, with the settings it had (a failed hairstyle change retries that change)
  $('faceError').textContent = '';
  try {
    current = current.job?.kind === 'hair'
      ? await api(`/api/projects/${current.id}/hair`, { hair: current.hair })
      : await api(`/api/projects/${current.id}/face`, { gender: current.gender || guessGender(current.scenario),
        hair: current.hair, clip: current.clip || $('clipService').value });
    renderFace();
  } catch (e) {
    $('faceError').textContent = e.message;
  }
});
$('hairBtn').addEventListener('click', async () => {
  $('faceError').textContent = '';
  try {
    current = await api(`/api/projects/${current.id}/hair`, { hair: hairPick });
    hairPick = null;
    renderFace();
  } catch (e) {
    $('faceError').textContent = e.message;
  }
});
$('acceptBtn').addEventListener('click', async () => {
  $('faceError').textContent = '';
  try {
    current = await api(`/api/projects/${current.id}/accept`, {});
    renderFace();
  } catch (e) {
    $('faceError').textContent = e.message;
  }
});

// ---------- model: 3D viewer, face/body sliders (live morphs), age, height, costume ----------
const M = { info: null, version: -1, three: null, scene: null, meshes: [], clothes: [], pending: {},
  age: null, height: null, outfit: null, footwear: null, colors: {} };
const sanitize = (n) => n.replace(/\s/g, '_').replace(/[[\]\.:\/]/g, '');

const renderJobList = (j, head, list) => {
  const at = j.steps.findIndex((s) => s.key === j.step);
  list.innerHTML = j.steps.map((s, i) => {
    const cls = i < at ? 'done' : i === at ? 'now' : '';
    return `<li class="${cls}"><span class="dot">${i < at ? '✓' : ''}</span>${s.label}<span class="time"></span></li>`;
  }).join('');
  if (j.waiting) head.textContent = `${j.waiting}…`;
  else if (j.state === 'queued') head.textContent = 'Starting…';
  else if (j.started) {
    const sec = (Date.now() - Date.parse(j.started)) / 1000;
    head.textContent = `${j.kind === 'apply' ? 'Applying your changes' : 'Making the 3D model'} · ${fmt(sec)} so far`;
  }
};

let viewerReady = null; // one viewer, even when several renders ask for it at once
const ensureViewer = () => (viewerReady ??= makeViewer());
const makeViewer = async () => {
  const THREE = await import('three');
  const { OrbitControls } = await import('three/addons/controls/OrbitControls.js');
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const stage = $('stage');
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping ?? THREE.ACESFilmicToneMapping;
  stage.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x2a2a28);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444440, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(1.5, 2.5, 3);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0xffffff, 1.0);
  rim.position.set(-2, 2, -2.5);
  scene.add(rim);
  const camera = new THREE.PerspectiveCamera(30, 1, 0.05, 50);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.minDistance = 0.2; // zooming into the target point breaks the controls (the camera goes NaN)
  controls.maxDistance = 8;
  const resize = () => {
    const w = stage.clientWidth;
    const h = stage.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(stage);
  resize();
  const loop = () => {
    if (!Number.isFinite(camera.position.x) || !Number.isFinite(camera.position.y)) {
      frame(document.querySelector('.tab.on')?.dataset.tab === 'face' ? 'face' : 'body'); // recover a broken view
    }
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  };
  loop();
  M.three = { THREE, GLTFLoader, scene, camera, controls, renderer, root: null };
};

let loading = null; // one load at a time; a load for the same version is shared
const loadModel = () => {
  const v = current?.modelVersion || 0;
  if (loading && loading.v === v) return loading.promise;
  const promise = doLoadModel().finally(() => { if (loading?.promise === promise) loading = null; });
  loading = { v, promise };
  return promise;
};
const doLoadModel = async () => {
  const p = current;
  await ensureViewer();
  const { THREE, GLTFLoader, scene, camera, controls } = M.three;
  const v = p.modelVersion || 0;
  M.info = await (await fetch(`/runs/${p.id}/model/model.json?v=${v}`)).json();
  const gltf = await new GLTFLoader().loadAsync(`/runs/${p.id}/model/model.glb?v=${v}`);
  if (M.three.root) scene.remove(M.three.root);
  M.three.root = gltf.scene;
  scene.add(gltf.scene);
  M.meshes = [];
  const clothesNames = new Set(M.info.garments.map((g) => sanitize(g.object)));
  M.clothes = [];
  gltf.scene.traverse((o) => {
    if (o.isMesh && o.morphTargetDictionary && Object.keys(o.morphTargetDictionary).some((k) => k.startsWith('m:'))) {
      M.meshes.push(o);
      hideUnderClothes(o);
    }
    if (o.isMesh || o.isObject3D) {
      let n = o;
      while (n && !clothesNames.has(n.name)) n = n.parent;
      if (n && o.isMesh) M.clothes.push(o);
    }
    if (o.isMesh && o.material && /Hair_Cards|Hair_Cap/.test(o.material.name)) {
      // hair cards (export_model.py): strands cut out by alpha, both sides drawn. HG's cards carry a colour layer
      // with no alpha: used as vertex colours it cuts every strand away, so it's ignored. Soft shine.
      o.material.vertexColors = false;
      o.material.side = THREE.DoubleSide;
      o.material.alphaToCoverage = true;
      o.material.roughness = 0.75;
      if (o.material.normalScale) o.material.normalScale.set(0.4, 0.4);
      o.material.needsUpdate = true;
    } else if (o.isMesh && o.material && /lash/i.test(`${o.name} ${o.material.name}`)) {
      o.visible = false; // the lash cap bakes empty; the face texture has the lash line
    }
  });
  M.box = new THREE.Box3().setFromObject(gltf.scene);
  frame(document.querySelector('.tab.on')?.dataset.tab === 'face' ? 'face' : 'body');
  M.version = v;
  M.pending = {};
  M.age = M.info.age;
  M.height = M.info.height_cm;
  M.outfit = null;
  M.footwear = null;
  M.colors = {};
  M.hair = null;
  M.hairLight = M.info.hair?.lightness ?? null;
  M.hairRed = M.info.hair?.redness ?? null;
  M.hairLen = M.info.hair?.length_cm ?? 0;
  M.hairWave = M.info.hair?.wave ?? 0;
  M.indian = null;
  buildPanels();
  applyMorphs();
  updateApplyBar();
};

/** The average colour of a texture's opaque pixels (a hair texture's strands), dark brown if there is none. */
const strandColour = (tex) => {
  const fallback = M.three.THREE ? new M.three.THREE.Color(0x2b1d16) : null;
  const img = tex?.image;
  if (!img) return fallback;
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const x = c.getContext('2d');
  x.drawImage(img, 0, 0, 64, 64);
  const d = x.getImageData(0, 0, 64, 64).data;
  let r = 0, g = 0, b = 0, n = 0;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] > 100) { r += d[i]; g += d[i + 1]; b += d[i + 2]; n++; }
  }
  if (n < 20) return fallback;
  return new M.three.THREE.Color().setRGB(r / n / 255, g / n / 255, b / n / 255, M.three.THREE.SRGBColorSpace);
};

/** Frame the face (face tab) or the whole body (other tabs). */
const frame = (what) => {
  if (!M.three || !M.box) return;
  const { THREE, camera, controls } = M.three;
  const b = M.box;
  const c = b.getCenter(new THREE.Vector3());
  const h = b.max.y - b.min.y;
  if (what === 'face') {
    const y = b.min.y + h * 0.915;
    controls.target.set(c.x, y, c.z);
    camera.position.set(c.x + h * 0.14, y + h * 0.015, c.z + h * 0.55);
  } else {
    controls.target.set(c.x, b.min.y + h * 0.52, c.z);
    camera.position.set(c.x, b.min.y + h * 0.58, c.z + h * 2.3);
  }
  controls.update();
  redraw();
};
const redraw = () => M.three && M.three.renderer.render(M.three.scene, M.three.camera);

/** The body's skin under the clothes (vertex colour red, from export_model.py) isn't drawn while clothes show. */
const hideUnderClothes = (mesh) => {
  // the marker is the colour layer that is red on part of the body only (HG's own layers are plain white)
  const g = mesh.geometry;
  const col = Object.keys(g.attributes).filter((k) => /^color(_\d+)?$/.test(k)).map((k) => g.attributes[k]).find((a) => {
    let hi = 0;
    for (let i = 0; i < a.count; i++) if (a.getX(i) > 0.5 && a.getY(i) < 0.5) hi++;
    return hi > a.count * 0.01 && hi < a.count * 0.9;
  });
  if (!col) return;
  mesh.geometry.setAttribute('hideMask', col);
  const mat = mesh.material;
  mat.vertexColors = false; // the marker is not a colour to paint with
  M.hideUniform = { value: 1 };
  mat.onBeforeCompile = (shader) => {
    shader.uniforms.uHide = M.hideUniform;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nattribute vec4 hideMask;\nvarying float vHide;')
      .replace('#include <begin_vertex>', '#include <begin_vertex>\nvHide = hideMask.r;');
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform float uHide;\nvarying float vHide;')
      .replace('void main() {', 'void main() {\n  if (uHide > 0.5 && vHide > 0.5) discard;');
  };
  mat.needsUpdate = true;
};

const applyMorphs = () => {
  if (!M.info) return;
  let bodyOff = false;
  for (const s of M.info.sliders) {
    const val = s.key in M.pending ? M.pending[s.key] : s.value;
    const w = val - s.value;
    if (s.part !== 'face' && Math.abs(w) > 1e-3) bodyOff = true;
    for (const m of M.meshes) {
      const i = m.morphTargetDictionary[s.morph];
      if (i !== undefined) m.morphTargetInfluences[i] = w;
    }
  }
  // clothes don't follow body morphs: hidden until Apply refits them
  M.clothes.forEach((o) => (o.visible = !bodyOff));
  if (M.hideUniform) M.hideUniform.value = bodyOff ? 0 : 1; // clothes hidden: show the whole body
  $('stageNote').hidden = !bodyOff;
  redraw();
};

const sliderEl = (label, min, max, step, value, changed, onInput, fmtv = (x) => (+x).toFixed(2)) => {
  const wrap = document.createElement('label');
  wrap.className = 'slider' + (changed ? ' changed' : '');
  wrap.innerHTML = `<span>${label}<b>${fmtv(value)}</b></span><input type="range" min="${min}" max="${max}" step="${step}" value="${value}">`;
  const input = wrap.querySelector('input');
  input.addEventListener('input', () => {
    wrap.querySelector('b').textContent = fmtv(input.value);
    onInput(+input.value, wrap);
  });
  return wrap;
};

const buildPanels = () => {
  for (const part of ['face', 'body']) {
    const box = $(`tab-${part}`);
    box.innerHTML = '';
    const groups = {};
    for (const s of M.info.sliders.filter((x) => x.part === part)) (groups[s.group] ??= []).push(s);
    for (const [g, list] of Object.entries(groups)) {
      const det = document.createElement('details');
      det.className = 'group';
      det.open = part === 'face' ? ['Nose', 'Lips & mouth', 'Cheeks'].includes(g) : ['Build', 'Torso'].includes(g);
      det.innerHTML = `<summary>${g}</summary>`;
      for (const s of list) {
        const val = s.key in M.pending ? M.pending[s.key] : s.value;
        det.append(sliderEl(s.label, s.min, s.max, 0.01, val, s.key in M.pending, (x, el) => {
          if (Math.abs(x - s.value) < 1e-3) delete M.pending[s.key];
          else M.pending[s.key] = x;
          el.classList.toggle('changed', s.key in M.pending);
          applyMorphs();
          updateApplyBar();
        }));
      }
      box.append(det);
    }
  }
  $('ageSlider').value = M.age;
  $('ageVal').textContent = `${M.age} years`;
  $('heightSlider').value = Math.round(M.height);
  $('heightVal').textContent = `${Math.round(M.height)} cm`;
  buildCostume();
  buildHair();
};

/** Long-hair presets: a base hairstyle grown to a length, with waves (long_hair.py builds them on Apply). */
const LONG_PRESETS = {
  female: [['Long straight', 'center part, 70 cm', 'Long\\Medium Center Part.json', 70, 0],
    ['Long wavy', 'side part, 70 cm, waves', 'Long\\Medium Side Part.json', 70, 0.55],
    ['Waist-length', 'center part, 95 cm', 'Long\\Medium Center Part.json', 95, 0.1],
    ['Long curly', '60 cm, strong waves', 'Long\\Medium Side Part.json', 60, 1.0],
    ['Shoulder length', 'bob grown, 40 cm', 'Long\\Bob Long.json', 40, 0.2]],
  male: [['Shoulder length', '35 cm', null, 35, 0.15], ['Long straight', '55 cm', null, 55, 0],
    ['Long wavy', '55 cm, waves', null, 55, 0.6]],
};
const buildLongPresets = () => {
  const box = $('longPresets');
  box.innerHTML = '';
  for (const [name, desc, style, len, wave] of LONG_PRESETS[M.info.gender] || []) {
    const b = document.createElement('button');
    b.type = 'button';
    const on = M.hairLen === len && Math.abs(M.hairWave - wave) < 0.01 && (!style || M.hair === style || hairKey(M.info.hair?.style) === style);
    b.className = 'style' + (on ? ' on' : '');
    b.innerHTML = `<b>${name}</b><span>${desc}</span>`;
    b.addEventListener('click', () => {
      if (style && hairKey(M.info.hair?.style) !== style) M.hair = style;
      M.hairLen = len;
      M.hairWave = wave;
      updateApplyBar();
      buildHair();
    });
    box.append(b);
  }
};

/** Hair tab: HG's hairstyles (preview pictures) and its hair colour values. */
const hairKey = (style) => (style || '').replace(/^hair\\head\\(fe)?male\\/, '');
const buildHair = async () => {
  buildLongPresets();
  const list = await hairList(M.info.gender);
  const box = $('hairGrid');
  box.innerHTML = '';
  const now = hairKey(M.info.hair?.style);
  for (const h of list) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'hair' + (h.id === M.hair ? ' on' : '') + (h.id === now ? ' current' : '');
    b.title = `${h.category}: ${h.name}`;
    b.innerHTML = (h.preview ? `<img src="${h.preview}" alt="" loading="lazy">` : '<img alt="">') + `<span>${h.name}</span>`;
    b.addEventListener('click', () => {
      M.hair = h.id === now ? null : h.id;
      updateApplyBar();
      buildHair();
    });
    box.append(b);
  }
  $('hairLen').value = M.hairLen;
  $('hairLenVal').textContent = M.hairLen ? `${M.hairLen} cm` : "style's own";
  $('hairWave').value = M.hairWave;
  $('hairWaveVal').textContent = (+M.hairWave).toFixed(2);
  if (M.hairLight !== null) {
    $('hairLight').value = M.hairLight;
    $('hairLightVal').textContent = (+M.hairLight).toFixed(2);
    $('hairRed').value = M.hairRed;
    $('hairRedVal').textContent = (+M.hairRed).toFixed(2);
  }
};
$('hairLen').addEventListener('input', () => {
  M.hairLen = +$('hairLen').value;
  $('hairLenVal').textContent = M.hairLen ? `${M.hairLen} cm` : "style's own";
  updateApplyBar();
});
$('hairWave').addEventListener('input', () => {
  M.hairWave = +$('hairWave').value;
  $('hairWaveVal').textContent = M.hairWave.toFixed(2);
  updateApplyBar();
});
$('hairLight').addEventListener('input', () => {
  M.hairLight = +$('hairLight').value;
  $('hairLightVal').textContent = M.hairLight.toFixed(2);
  updateApplyBar();
});
$('hairRed').addEventListener('input', () => {
  M.hairRed = +$('hairRed').value;
  $('hairRedVal').textContent = M.hairRed.toFixed(2);
  updateApplyBar();
});

$('ageSlider').addEventListener('input', () => {
  M.age = +$('ageSlider').value;
  $('ageVal').textContent = `${M.age} years`;
  updateApplyBar();
});
$('heightSlider').addEventListener('input', () => {
  M.height = +$('heightSlider').value;
  $('heightVal').textContent = `${M.height} cm`;
  updateApplyBar();
});

let outfitCache = {};
const INDIAN = {
  female: [['lehenga', 'Lehenga choli', 'blouse, flared skirt, dupatta', '#8E1036', '#D9A520', 'paisley_butta', 'Blouse & dupatta'],
    ['saree', 'Saree', 'blouse, skirt, pallu', '#0F6E5C', '#C8961E', 'block_floral', 'Accent'],
    ['kurta', 'Kurta & churidar', 'long kurta, slim trousers', '#C2185B', '#F3E5C8', 'block_floral', 'Churidar']],
  male: [['kurta_pyjama', 'Kurta pyjama', 'long kurta, pyjama', '#F2E8D5', '#F7F3EA', 'brocade', 'Pyjama'],
    ['sherwani', 'Sherwani', 'long coat, churidar', '#E7D3A8', '#F4EEE0', 'brocade', 'Churidar']],
};
const PRINTS = ['paisley_butta', 'bandhani', 'leheriya', 'brocade', 'block_floral'];
const buildIndian = () => {
  const box = $('indianStyles');
  box.innerHTML = '';
  for (const [id, name, desc, main, accent, print, accentLabel] of INDIAN[M.info.gender] || []) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'style' + (M.indian?.style === id ? ' on' : '');
    b.innerHTML = `<b>${name}</b><span>${desc}</span>`;
    b.addEventListener('click', () => {
      M.indian = M.indian?.style === id ? null : { style: id, main, accent, pattern: print, accentLabel };
      if (M.indian) M.outfit = null;
      updateApplyBar();
      buildIndian();
      buildCostume();
    });
    box.append(b);
  }
  $('indianOptions').hidden = !M.indian;
  if (!M.indian) return;
  $('indMain').value = M.indian.main.toLowerCase();
  $('indAccent').value = M.indian.accent.toLowerCase();
  $('indAccentLabel').textContent = M.indian.accentLabel || 'Accent colour';
  const grid = $('indPrints');
  grid.innerHTML = '';
  for (const pr of [...PRINTS, 'none']) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'hair' + (M.indian.pattern === pr ? ' on' : '');
    b.innerHTML = (pr === 'none' ? '<img alt="">' : `<img src="/hg/patterns/Indian/${pr}.jpg" alt="">`) +
      `<span>${pr === 'none' ? 'Plain' : pr.replace(/_/g, ' ')}</span>`;
    b.addEventListener('click', () => {
      M.indian.pattern = pr;
      buildIndian();
      updateApplyBar();
    });
    grid.append(b);
  }
};
$('indMain').addEventListener('input', (e) => { if (M.indian) { M.indian.main = e.target.value; updateApplyBar(); } });
$('indAccent').addEventListener('input', (e) => { if (M.indian) { M.indian.accent = e.target.value; updateApplyBar(); } });

const buildCostume = async () => {
  const g = M.info.gender;
  buildIndian();
  const opts = (outfitCache[g] ??= await api(`/api/outfits?gender=${g}`));
  const grid = (box, list, kind) => {
    box.innerHTML = '';
    const now = M[kind] || M.info.current?.[kind];
    for (const o of list) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'hair' + (o.id === M[kind] ? ' on' : '') + (o.id === M.info.current?.[kind] ? ' current' : '');
      b.title = `${o.category}: ${o.name}`;
      b.innerHTML = (o.preview ? `<img src="${o.preview}" alt="" loading="lazy">` : '<img alt="">') + `<span>${o.name}</span>`;
      b.addEventListener('click', () => {
        M[kind] = o.id === M.info.current?.[kind] ? null : o.id;
        if (kind === 'outfit' && M.outfit) M.indian = null;
        updateApplyBar();
        buildCostume();
      });
      box.append(b);
    }
    return now;
  };
  grid($('outfitGrid'), opts.outfits, 'outfit');
  grid($('footGrid'), opts.footwear, 'footwear');
  const list = $('colorList');
  list.innerHTML = '';
  if (M.outfit || M.footwear) {
    list.innerHTML = '<p class="hint">Apply the new outfit first; then its colours can be changed here.</p>';
    return;
  }
  for (const gmt of M.info.garments) {
    if (!gmt.colors.length) continue;
    const h = document.createElement('div');
    h.className = 'garment';
    h.textContent = gmt.label;
    list.append(h);
    for (const c of gmt.colors) {
      const row = document.createElement('label');
      row.className = 'colorrow';
      const cur = M.colors[gmt.object]?.[c.input] || c.hex;
      row.innerHTML = `<input type="color" value="${cur.toLowerCase()}"><span>${c.label}</span>`;
      row.querySelector('input').addEventListener('input', (e) => {
        (M.colors[gmt.object] ??= {})[c.input] = e.target.value;
        if (e.target.value.toUpperCase() === c.hex.toUpperCase()) delete M.colors[gmt.object][c.input];
        updateApplyBar();
      });
      list.append(row);
    }
  }
};

const changes = () => {
  const ch = {};
  if (!M.info) return ch; // no model loaded yet
  if (Object.keys(M.pending).length) ch.keys = { ...M.pending };
  if (M.age !== M.info.age) ch.age = M.age;
  if (Math.abs(M.height - M.info.height_cm) >= 1) ch.height_cm = M.height;
  if (M.outfit) ch.outfit = M.outfit;
  if (M.footwear) ch.footwear = M.footwear;
  if (M.hair) ch.hair = M.hair;
  if (M.info.hair && (M.hairLen ?? 0) !== (M.info.hair.length_cm ?? 0)) ch.hair_length_cm = M.hairLen;
  if (M.info.hair && Math.abs((M.hairWave ?? 0) - (M.info.hair.wave ?? 0)) > 0.01) ch.hair_wave = M.hairWave;
  if (M.indian) ch.indian = { style: M.indian.style, main: M.indian.main, accent: M.indian.accent, pattern: M.indian.pattern };
  if (M.info.hair && M.hairLight !== null && Math.abs(M.hairLight - M.info.hair.lightness) > 0.005) ch.hair_lightness = M.hairLight;
  if (M.info.hair && M.hairRed !== null && Math.abs(M.hairRed - M.info.hair.redness) > 0.005) ch.hair_redness = M.hairRed;
  const cols = Object.fromEntries(Object.entries(M.colors).filter(([, v]) => Object.keys(v).length));
  if (Object.keys(cols).length) ch.colors = cols;
  return ch;
};

const updateApplyBar = () => {
  const ch = changes();
  const n = (ch.keys ? Object.keys(ch.keys).length : 0)
    + ['age', 'height_cm', 'outfit', 'footwear', 'hair', 'hair_lightness', 'hair_redness', 'hair_length_cm', 'hair_wave',
      'indian'].filter((k) => k in ch).length
    + (ch.colors ? Object.values(ch.colors).reduce((a, v) => a + Object.keys(v).length, 0) : 0);
  $('applyBar').hidden = !n || view !== 'model';
  $('applyCount').textContent = `${n} change${n === 1 ? '' : 's'}`;
  const live = ch.keys && !ch.age && !ch.height_cm && !ch.outfit && !ch.footwear && !ch.colors && !ch.hair && !ch.indian
    && !('hair_length_cm' in ch) && !('hair_wave' in ch)
    && !('hair_lightness' in ch) && !('hair_redness' in ch);
  $('applyHint').textContent = live ? 'Shown live. Apply to save them (and refit the clothes): about 3 minutes.'
    : 'Age, height and costume show after Apply: about 3 minutes.';
};

$('resetBtn').addEventListener('click', () => {
  M.pending = {};
  M.age = M.info.age;
  M.height = M.info.height_cm;
  M.outfit = M.footwear = null;
  M.colors = {};
  M.hair = null;
  M.indian = null;
  M.hairLen = M.info.hair?.length_cm ?? 0;
  M.hairWave = M.info.hair?.wave ?? 0;
  M.hairLight = M.info.hair?.lightness ?? null;
  M.hairRed = M.info.hair?.redness ?? null;
  buildPanels();
  applyMorphs();
  updateApplyBar();
});

$('applyBtn').addEventListener('click', async () => {
  $('modelError').textContent = '';
  try {
    const ch = changes();
    try { localStorage.setItem('lastApply', JSON.stringify(ch)); } catch { /* private mode */ }
    current = await api(`/api/projects/${current.id}/apply`, ch);
    $('applyBar').hidden = true;
    renderModel();
  } catch (e) {
    $('modelError').textContent = e.message;
  }
});

document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach((x) => x.classList.toggle('on', x === t));
  for (const k of ['face', 'body', 'age', 'hair', 'costume']) $(`tab-${k}`).hidden = k !== t.dataset.tab;
  frame(t.dataset.tab === 'face' ? 'face' : 'body');
}));

const renderModel = async () => {
  const p = current;
  if (!p) return;
  const j = p.job && (p.job.kind === 'model' || p.job.kind === 'apply') ? p.job : null;
  const running = j && (j.state === 'queued' || j.state === 'running');
  const hasModel = (p.modelVersion || 0) > 0;
  $('modelStart').hidden = hasModel || running || j?.state === 'failed';
  $('modelProgress').hidden = !running;
  $('modelFailed').hidden = !(j?.state === 'failed') || running;
  $('modelStudio').hidden = !hasModel;
  setStep(hasModel ? 'play' : 'model');
  if (running) renderJobList(j, $('modelProgressHead'), $('modelProgressSteps'));
  if (j?.state === 'failed') $('modelErrorText').textContent = `It stopped: ${j.error || 'unknown error'}`;
  if (hasModel && M.version !== (p.modelVersion || 0)) {
    try {
      await loadModel();
    } catch (e) {
      $('modelError').textContent = `The model didn't load: ${e.message}`;
    }
  }
  updateApplyBar();
};

$('makeModelBtn').addEventListener('click', async () => {
  $('modelError').textContent = '';
  try {
    current = await api(`/api/projects/${current.id}/model`, {});
    renderModel();
  } catch (e) {
    $('modelError').textContent = e.message;
  }
});
$('modelRetryBtn').addEventListener('click', async () => {
  try {
    current = current.job?.kind === 'apply'
      ? await api(`/api/projects/${current.id}/apply`, JSON.parse(localStorage.getItem('lastApply') || '{}'))
      : await api(`/api/projects/${current.id}/model`, {});
    renderModel();
  } catch (e) {
    $('modelError').textContent = e.message;
  }
});
$('backToFace').addEventListener('click', () => go('face'));
$('toModelBtn').addEventListener('click', () => go('model'));

refreshServices();
setInterval(refreshServices, 30000);
refreshProjects();
show('scenario');
