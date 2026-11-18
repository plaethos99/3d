/**
 * Pictures from Google Flow or Grok Imagine, in a real browser, the way the BookShorts studio makes
 * them (ported from its apps/studio/electron/flow.ts and grok.ts; the page drivers are copied unchanged).
 *
 * Each service runs in Microsoft Edge (or Chrome) with a profile of its own under tools/browser-profiles
 * and the DevTools protocol on a local port, so your everyday browser is never touched, and BookShorts'
 * windows (ports 9334, 9360) aren't either. You sign in there once; the profile keeps the session.
 * Nothing here signs in, and if a service shows a CAPTCHA, it stops for you to answer it.
 *
 *   node imagegen.mjs open  flow|grok                         open the window (sign in there; in Flow, open a project)
 *   node imagegen.mjs state flow|grok                         closed / signin / ready
 *   node imagegen.mjs image flow|grok <aspect> <out> <prompt>  make a picture
 *   node imagegen.mjs edit  flow|grok <aspect> <in> <out> <prompt>   change a picture as the prompt says
 *   node imagegen.mjs video grok <aspect> <seconds> <in> <out.mp4> <prompt>   bring a picture to life (it is the
 *        first frame; 6, 10 or 15 s, 720p, silent)
 *   node imagegen.mjs video flow <aspect> <seconds> <in> <out.mp4> <prompt>   the same with Flow (Veo): the picture
 *        is uploaded into the open Flow project, then used as the start frame of a Frames clip (Flow's lengths)
 * aspect: 16:9, 9:16, 1:1, 4:3, 3:4. Prints one line of JSON.
 */
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { FLOW_DRIVER } from './flowDriver.mjs';
import { GROK_DRIVER } from './grokDriver.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..');
const PROFILES = path.join(ROOT, 'tools', 'browser-profiles');
const SERVICES = {
  flow: { url: 'https://flow.google.com/', port: 9434, profile: path.join(PROFILES, 'flow') },
  grok: { url: 'https://grok.com/imagine', port: 9460, profile: path.join(PROFILES, 'grok') },
};
const BROWSERS = [
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  path.join(process.env.LOCALAPPDATA ?? '', 'Google\\Chrome\\Application\\chrome.exe'),
];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const tabs = async (svc) => {
  try {
    const res = await fetch(`http://127.0.0.1:${svc.port}/json`, { signal: AbortSignal.timeout(1500) });
    return await res.json();
  } catch {
    return null;
  }
};

/** Attach to a tab for the length of fn(send, listen). */
const attach = async (tab, fn) => {
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve);
    ws.addEventListener('error', () => reject(new Error('Could not connect to the tab')));
  });
  let id = 0;
  const waiting = new Set();
  ws.addEventListener('close', () => waiting.forEach((fail) => fail(new Error('The connection to the window was lost'))), { once: true });
  const send = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const mine = ++id;
      const onMessage = (e) => {
        const msg = JSON.parse(String(e.data));
        if (msg.id !== mine) return;
        ws.removeEventListener('message', onMessage);
        waiting.delete(reject);
        if (msg.error) reject(new Error(msg.error.message));
        else resolve(msg.result ?? {});
      };
      ws.addEventListener('message', onMessage);
      waiting.add(reject);
      ws.send(JSON.stringify({ id: mine, method, params }));
    });
  const listen = (handler) => {
    const onMessage = (e) => {
      const msg = JSON.parse(String(e.data));
      if (msg.method) handler(msg.method, msg.params ?? {});
    };
    ws.addEventListener('message', onMessage);
    return () => ws.removeEventListener('message', onMessage);
  };
  try {
    return await fn(send, listen);
  } finally {
    ws.close();
  }
};

const evaluate = async (send, expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) {
    const raw = r.exceptionDetails.exception?.description ?? r.exceptionDetails.text ?? 'Page script failed';
    throw new Error(raw.split('\n')[0].replace(/^Error: /, ''));
  }
  return r.result?.value;
};

const clickAt = async (send, at) => {
  await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: at.x, y: at.y });
  await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: at.x, y: at.y, button: 'left', clickCount: 1 });
  await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: at.x, y: at.y, button: 'left', clickCount: 1 });
};

const open = async (name) => {
  const svc = SERVICES[name];
  const open = await tabs(svc);
  if (open) {
    const tab = open.find((t) => t.type === 'page');
    if (tab) await attach(tab, (send) => send('Page.bringToFront'));
    else await fetch(`http://127.0.0.1:${svc.port}/json/new?${svc.url}`, { method: 'PUT' });
    return { opened: name, already_running: true };
  }
  const exe = BROWSERS.find((p) => existsSync(p));
  if (!exe) throw new Error('Needs Microsoft Edge or Google Chrome; neither was found.');
  mkdirSync(svc.profile, { recursive: true });
  spawn(exe, [`--remote-debugging-port=${svc.port}`, `--user-data-dir=${svc.profile}`, '--no-first-run',
    '--no-default-browser-check', '--disable-background-timer-throttling', '--disable-renderer-backgrounding',
    '--disable-backgrounding-occluded-windows', svc.url], { detached: true, stdio: 'ignore' }).unref();
  return { opened: name, profile: svc.profile, port: svc.port };
};

// ------------------------------------------------------------------ Flow
const isFlowProject = (t) => t.type === 'page' && t.url.includes('flow.google.com/project/');

const flowTab = async () => {
  const tab = (await tabs(SERVICES.flow))?.find(isFlowProject);
  if (!tab) throw new Error('Open Flow (node imagegen.mjs open flow), sign in, and open a project (New project) first.');
  return tab;
};

/** Flow's answer to the generate request: the new picture's signed link, or why not. */
const pictureIn = (body) => {
  const links = [...body.matchAll(/https:\/\/flow-content\.google\/image\/[^"\\]+(?:\\+u00(?:3d|26)[^"\\]*)*/g)]
    .map((m) => m[0].replace(/\\+u003d/g, '=').replace(/\\+u0026/g, '&'));
  const first = /\[\[\[\\+"([0-9a-f-]{36})\\+"/.exec(body)?.[1];
  const link = links.find((l) => first && l.includes(first)) ?? links[0];
  if (link) return link;
  const said = body.replace(/\s+/g, ' ').slice(0, 240);
  throw new Error(/polic|violat|unsafe|prohibit|safety|filter/i.test(body) ? `POLICY: Flow refused it (${said})` : `Flow failed to make it (${said})`);
};

const flowGenerate = async (send, listen, at, prompt) => {
  await send('Network.enable', { maxPostDataSize: 1_000_000 });
  const excerpt = prompt.slice(0, 40);
  let stop = () => {};
  const answer = new Promise((resolve, reject) => {
    const mine = new Set();
    stop = listen((method, params) => {
      if (method === 'Network.requestWillBeSent') {
        const request = params.request;
        if (request.method !== 'POST' || !/batchexecute/.test(request.url)) return;
        let sent = request.postData ?? '';
        try {
          sent = decodeURIComponent(sent);
        } catch {
          /* as it came */
        }
        if (sent.includes(excerpt)) mine.add(params.requestId);
      } else if (mine.has(params.requestId) && method === 'Network.loadingFinished')
        void send('Network.getResponseBody', { requestId: params.requestId })
          .then((r) => {
            const body = String(r.body);
            if (/flow-content\.google\/image|ErrorInfo/.test(body)) resolve(pictureIn(body));
          })
          .catch(reject);
    });
  });
  const challenge = evaluate(send, 'window.__bookshorts.watchForChallenge()').then(() => new Promise(() => {}));
  const late = new Promise((_, reject) => setTimeout(() => reject(new Error('Flow did not return an image in time')), 180_000));
  await clickAt(send, at);
  try {
    return await Promise.race([answer, challenge, late]);
  } finally {
    stop();
  }
};

const flowImage = async (prompt, aspect) => {
  const url = await attach(await flowTab(), async (send, listen) => {
    await send('Page.bringToFront');
    await evaluate(send, FLOW_DRIVER);
    const at = await evaluate(send, `window.__bookshorts.prepare(${JSON.stringify({ prompt, aspect })})`);
    return flowGenerate(send, listen, at, prompt);
  });
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download the image from Flow (${res.status})`);
  return Buffer.from(await res.arrayBuffer());
};

const flowEdit = async (picture, prompt, aspect) => {
  const name = `cs-${createHash('sha1').update(`${picture}|${statSync(picture).mtimeMs}`).digest('hex').slice(0, 16)}`;
  const url = await attach(await flowTab(), async (send, listen) => {
    await send('Page.bringToFront');
    await evaluate(send, FLOW_DRIVER);
    await evaluate(send, `window.__bookshorts.startEdit(${JSON.stringify({ aspect })})`);
    if (!(await evaluate(send, `window.__bookshorts.findIngredient(${JSON.stringify(name)})`))) {
      await evaluate(send, 'window.__bookshorts.armUpload()');
      const dir = mkdtempSync(path.join(tmpdir(), 'cs-flow-'));
      try {
        const file = path.join(dir, `${name}${path.extname(picture)}`);
        copyFileSync(picture, file);
        const doc = await send('DOM.getDocument', { depth: -1, pierce: true });
        const found = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: '[data-bs-upload]' });
        if (!found.nodeId) throw new Error('Flow has no picture upload on the page');
        await send('DOM.setFileInputFiles', { nodeId: found.nodeId, files: [file] });
        if (!(await evaluate(send, `window.__bookshorts.findIngredient(${JSON.stringify(name)}, 90000)`))) throw new Error("Flow didn't take the picture");
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    }
    await evaluate(send, `window.__bookshorts.useIngredient(${JSON.stringify(name)})`);
    return flowGenerate(send, listen, await evaluate(send, `window.__bookshorts.prepareEdit(${JSON.stringify({ prompt })})`), prompt);
  });
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download the image from Flow (${res.status})`);
  return Buffer.from(await res.arrayBuffer());
};

// A clip from Flow (Veo): the picture goes into the project's media (uploaded as an edit's ingredient is, under a
// name made from its contents), then is the start frame of a "Frames" clip; the clip's tile is hovered so Flow
// loads the video, whose URL is downloaded.
const flowVideo = async (picture, prompt, aspect, seconds) => {
  const name = `cs-${createHash('sha1').update(`${picture}|${statSync(picture).mtimeMs}`).digest('hex').slice(0, 16)}`;
  const url = await attach(await flowTab(), async (send) => {
    await send('Page.bringToFront');
    await evaluate(send, FLOW_DRIVER);
    await evaluate(send, `window.__bookshorts.startEdit(${JSON.stringify({ aspect })})`);
    if (!(await evaluate(send, `window.__bookshorts.findIngredient(${JSON.stringify(name)})`))) {
      await evaluate(send, 'window.__bookshorts.armUpload()');
      const dir = mkdtempSync(path.join(tmpdir(), 'cs-flow-'));
      try {
        const file = path.join(dir, `${name}${path.extname(picture)}`);
        copyFileSync(picture, file);
        const doc = await send('DOM.getDocument', { depth: -1, pierce: true });
        const found = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: '[data-bs-upload]' });
        if (!found.nodeId) throw new Error('Flow has no picture upload on the page');
        await send('DOM.setFileInputFiles', { nodeId: found.nodeId, files: [file] });
        if (!(await evaluate(send, `window.__bookshorts.findIngredient(${JSON.stringify(name)}, 90000)`))) throw new Error("Flow didn't take the picture");
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    }
    // its key in the project's media (what the start-frame picker lists)
    const key = await evaluate(send, `(() => {
      const o = [...document.querySelectorAll('[role="option"]')].find((x) => x.innerText.includes(${JSON.stringify(name)}));
      const im = o && o.querySelector('img');
      return im ? (im.src.split('/image/')[1] || im.src).split('?')[0] : null;
    })()`);
    if (!key) throw new Error('Flow took the picture but its media key was not found');
    for (const type of ['keyDown', 'keyUp']) await send('Input.dispatchKeyEvent', { type, key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
    await sleep(600);
    const at = await evaluate(send, `window.__bookshorts.prepareVideo(${JSON.stringify({ prompt, aspect, seconds, startKey: key })})`);
    await clickAt(send, at);
    const found = await evaluate(send, 'window.__bookshorts.waitForVideo()');
    if (found.url) return found.url;
    await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x: found.x, y: found.y });
    const u = await evaluate(send, 'window.__bookshorts.videoUrl()');
    if (!u) throw new Error('Flow made the clip but its video did not load');
    return u;
  });
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download the clip from Flow (${res.status})`);
  return Buffer.from(await res.arrayBuffer());
};

// ------------------------------------------------------------------ Grok
const isImagine = (t) => t.type === 'page' && /^https:\/\/grok\.com\/imagine/.test(t.url);
const GROK_SIGNED_IN = `() => {
  const signIn = [...document.querySelectorAll('a,button')].some((b) => /^\\s*sign (in|up)\\s*$/i.test(b.textContent || ''));
  const box = document.querySelector('textarea, [contenteditable="true"]');
  return (!!box || /\\/imagine\\/post\\//.test(location.pathname)) && !signIn;
}`;

const grokTab = async () => {
  const tab = (await tabs(SERVICES.grok))?.find(isImagine);
  if (!tab) throw new Error('Open Grok Imagine (node imagegen.mjs open grok) and sign in first.');
  return tab;
};

const freshGrok = async (send) => {
  await send('Page.navigate', { url: SERVICES.grok.url });
  for (let i = 0; i < 40; i++) {
    await sleep(500);
    if (await evaluate(send, `!!document.querySelector('textarea, [contenteditable="true"]')`).catch(() => false)) break;
  }
  await sleep(1000);
};

const grokImage = async (prompt, aspect) =>
  attach(await grokTab(), async (send) => {
    await send('Page.bringToFront');
    await freshGrok(send);
    await evaluate(send, GROK_DRIVER);
    const box = await evaluate(send, `window.__bsGrok.prepare(${JSON.stringify({ aspect })})`);
    await clickAt(send, box);
    await send('Input.insertText', { text: prompt });
    await sleep(500);
    await clickAt(send, await evaluate(send, 'window.__bsGrok.submitAt()'));
    return Buffer.from(await evaluate(send, 'window.__bsGrok.waitForImage()'), 'base64');
  });

const grokEdit = async (picture, prompt) =>
  attach(await grokTab(), async (send) => {
    await send('Page.bringToFront');
    await freshGrok(send);
    await evaluate(send, GROK_DRIVER);
    await evaluate(send, 'window.__bsGrok.prepareEdit()');
    const doc = await send('DOM.getDocument', { depth: -1, pierce: true });
    const found = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: '[data-bs-upload]' });
    if (!found.nodeId) throw new Error('Grok Imagine has no picture upload on the page');
    await send('DOM.setFileInputFiles', { nodeId: found.nodeId, files: [path.resolve(picture)] });
    if (!(await evaluate(send, 'window.__bsGrok.waitForAttachment()'))) throw new Error("Grok Imagine didn't take the picture");
    await clickAt(send, await evaluate(send, 'window.__bsGrok.boxAt()'));
    await send('Input.insertText', { text: prompt });
    await sleep(500);
    await clickAt(send, await evaluate(send, 'window.__bsGrok.submitAt()'));
    for (let i = 0; i < 60; i++) {
      await sleep(1000);
      if (/\/imagine\/post\//.test(await evaluate(send, 'location.href').catch(() => ''))) break;
      if (i === 59) throw new Error('Grok Imagine did not start the picture');
    }
    await sleep(1500);
    await evaluate(send, GROK_DRIVER);
    return Buffer.from(await evaluate(send, 'window.__bsGrok.grabPost()'), 'base64');
  });

// A clip that starts from a picture (Grok Imagine's video mode, the picture uploaded as the first frame).
const grokVideo = async (picture, prompt, aspect, seconds) =>
  attach(await grokTab(), async (send) => {
    await send('Page.bringToFront');
    await freshGrok(send);
    await evaluate(send, GROK_DRIVER);
    await evaluate(send, `window.__bsGrok.prepareVideo(${JSON.stringify({ aspect, seconds })})`);
    const doc = await send('DOM.getDocument', { depth: -1, pierce: true });
    const found = await send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: '[data-bs-upload]' });
    if (!found.nodeId) throw new Error('Grok Imagine has no picture upload on the page');
    await send('DOM.setFileInputFiles', { nodeId: found.nodeId, files: [path.resolve(picture)] });
    if (!(await evaluate(send, 'window.__bsGrok.waitForAttachment()'))) throw new Error("Grok Imagine didn't take the picture");
    await clickAt(send, await evaluate(send, 'window.__bsGrok.boxAt()'));
    await send('Input.insertText', { text: prompt });
    await sleep(500);
    await clickAt(send, await evaluate(send, 'window.__bsGrok.submitAt()'));
    await sleep(4000);
    // the page moves to the new post; if it reloaded, the driver is put back (it then counts every clip as new)
    if (!(await evaluate(send, '!!window.__bsGrok').catch(() => false))) await evaluate(send, GROK_DRIVER);
    return Buffer.from(await evaluate(send, 'window.__bsGrok.waitForVideo()'), 'base64');
  });

// ------------------------------------------------------------------ command line
const state = async (name) => {
  const open = await tabs(SERVICES[name]);
  if (!open) return 'closed';
  if (name === 'flow') return open.some(isFlowProject) ? 'ready' : 'signin';
  const tab = open.find(isImagine);
  if (!tab) return 'signin';
  try {
    return (await attach(tab, (send) => evaluate(send, `(${GROK_SIGNED_IN})()`))) ? 'ready' : 'signin';
  } catch {
    return 'signin';
  }
};

const save = (out, data) => {
  mkdirSync(path.dirname(path.resolve(out)), { recursive: true });
  writeFileSync(out, data);
  return { saved: path.resolve(out), bytes: data.length };
};

const [cmd, name, ...rest] = process.argv.slice(2);
try {
  if (!SERVICES[name]) throw new Error('Say which service: flow or grok');
  let result;
  if (cmd === 'open') result = await open(name);
  else if (cmd === 'state') result = { service: name, state: await state(name) };
  else if (cmd === 'options' && name === 'flow') {
    // what Flow's settings popover offers (mode, aspect, resolution, count...), for picking better ones
    result = await attach(await flowTab(), async (send) => {
      await send('Page.bringToFront');
      return evaluate(send, `(async () => {
        const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
        const t = document.querySelector('button[aria-label="Settings trigger"]');
        if (!t) throw new Error('no settings button');
        t.click(); await sleep(800);
        const opts = [...document.querySelectorAll('button[role="radio"]')].map((b) => ({
          label: b.innerText.trim().split('\\n').pop().trim(), checked: b.getAttribute('aria-checked') === 'true' }));
        t.click(); await sleep(300);
        return { options: opts };
      })()`);
    });
  }
  else if (cmd === 'image') {
    const [aspect, out, ...words] = rest;
    const prompt = words.join(' ');
    result = save(out, name === 'flow' ? await flowImage(prompt, aspect) : await grokImage(prompt, aspect));
  } else if (cmd === 'edit') {
    const [aspect, input, out, ...words] = rest;
    const prompt = words.join(' ');
    result = save(out, name === 'flow' ? await flowEdit(path.resolve(input), prompt, aspect) : await grokEdit(input, prompt));
  } else if (cmd === 'video') {
    const [aspect, seconds, input, out, ...words] = rest;
    result = save(out, name === 'flow' ? await flowVideo(path.resolve(input), words.join(' '), aspect, Number(seconds))
      : await grokVideo(input, words.join(' '), aspect, Number(seconds)));
  } else throw new Error('Commands: open, state, image, edit, video');
  console.log(JSON.stringify({ ok: true, ...result }));
} catch (e) {
  console.log(JSON.stringify({ ok: false, error: String(e.message ?? e) }));
  process.exitCode = 1;
}
