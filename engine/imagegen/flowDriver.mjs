// Copied unchanged from the BookShorts project (packages/pipeline/src/flowDriver.ts).
/**
 * Drives Google Flow's web page to make images and video clips. Flow has no
 * public API, so the app opens Flow in its own window, where the user is
 * signed in, and injects FLOW_DRIVER. One image takes three steps:
 *
 *   1. `__bookshorts.prepare({prompt, aspect})` picks the aspect ratio, types
 *      the prompt and returns where the generate button is (CSS pixels);
 *   2. the app clicks there with a native input event — Flow ignores clicks
 *      made by page script on that button;
 *   3. the app reads the new image's URL from Flow's own answer to that request.
 *
 * A clip is the same with `prepareVideo({prompt, aspect, seconds, startKey})`,
 * which switches to Video → Frames and picks the project's picture `startKey`
 * (by key, or by its look when the tile no longer shows the key) as the start
 * frame, then `waitForVideo()`, which finds the new clip's tile; the app
 * hovers it natively and reads the clip's URL with `videoUrl()`.
 *
 * A picture changed ("the same person, now sitting") goes in as an ingredient:
 * `startEdit({aspect})`, then `findIngredient(name)` for a picture already in
 * the project under its name; if it isn't there, `armUpload()` marks the upload
 * input for the app to hand the file to, and `findIngredient(name, ms)` waits
 * for it. `useIngredient(name)` adds it to the prompt, and `prepareEdit({prompt})`
 * types the prompt and returns where the generate button is, as above.
 *
 * It works the page the way a person would, at a person's pace. If Flow shows
 * a CAPTCHA challenge it stops and reports it, for the user to answer — it
 * never tries to get around one.
 *
 * Tied to Flow's current page: the prompt box is labelled "What do you want to
 * create?", the generate button "Start generation", the settings button
 * "Settings trigger", mode (Image/Video), video kind (Frames/Ingredients),
 * aspect, resolution, length and count are radio buttons in its popover, the
 * start frame is a "Start" button opening a listbox of the project's media,
 * and Agent mode is a chip labelled "Agent" with aria-pressed. When
 * Flow changes those, this is the one place to update.
 */
export const FLOW_DRIVER = `(() => {
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const radios = () => [...document.querySelectorAll('button[role="radio"]')];
  const pick = (label) => radios().find((b) => b.innerText.trim().split('\\n').pop().trim() === label);
  const challenge = () =>
    [...document.querySelectorAll('iframe')].some((f) => /recaptcha/.test(f.src) && /bframe|challenge/.test(f.src + f.title) && f.offsetHeight > 50);
  const imageKey = (src) => (src.split('/image/')[1] || src).split('?')[0];
  const visible = (e) => e.offsetParent !== null;
  // The project's grid, newest first: a generation's tile goes in at the very
  // start, showing its progress, then its picture or an error. Old tiles are
  // redrawn as the page scrolls and fills, so an element being new says nothing;
  // only the start of the grid says what this generation did.
  const toTop = () => {
    const grid = document.querySelector('cdk-virtual-scroll-viewport');
    if (grid && grid.scrollTop) grid.scrollTop = 0;
  };
  const gridTiles = () => [...document.querySelectorAll('flow-grid-tile-container')];
  const isError = (tile) => !!tile.querySelector('flow-error-tile');
  // How many failures the grid starts with: one more than before is this one.
  const leadingErrors = () => {
    let n = 0;
    for (const t of gridTiles()) {
      if (!isError(t)) break;
      n++;
    }
    return n;
  };
  let errorsBefore = 0;
  const markErrors = () => {
    toTop();
    errorsBefore = leadingErrors();
  };
  const refusal = () => {
    toTop();
    if (leadingErrors() <= errorsBefore) return null;
    const text = gridTiles()[0].textContent || '';
    return /polic|violat|safety|guideline/i.test(text) ? 'POLICY: Flow refused it: it might break Google’s policies' : 'Flow failed to make it: ' + text.replace(/\s+/g, ' ').trim().slice(0, 160);
  };

  // Sets the popover's radio options in order (a mode first: later options
  // depend on it), opening the popover if needed and closing it after.
  async function setOptions(labels) {
    // A project just opened is still drawing its page: give it a moment.
    let trigger = null;
    for (let i = 0; i < 40 && !trigger; i++) {
      trigger = document.querySelector('button[aria-label="Settings trigger"]');
      if (!trigger) await new Promise((r) => setTimeout(r, 500));
    }
    if (!trigger) throw new Error('Flow page not ready: no settings button');
    if (!pick(labels[0])) { trigger.click(); await sleep(700); }
    for (const label of labels) {
      const b = pick(label);
      if (!b) throw new Error('Flow has no option ' + label);
      if (b.getAttribute('aria-checked') !== 'true') { b.click(); await sleep(400); }
    }
    if (pick(labels[0])) { trigger.click(); await sleep(400); }
  }

  async function typePrompt(text) {
    const box = document.querySelector('[role="textbox"][contenteditable="true"][aria-label*="create"]');
    if (!box) throw new Error('Flow page not ready: no prompt box');
    box.focus();
    document.execCommand('selectAll');
    document.execCommand('delete');
    document.execCommand('insertText', false, text);
    await sleep(500);
    if (!box.innerText.trim()) throw new Error('Could not type the prompt into Flow');
  }

  // In Agent mode a prompt goes to Flow's chat agent, not straight to the
  // image model, and the aspect and count options are gone. Some accounts
  // start in it.
  async function agentOff() {
    const chip = [...document.querySelectorAll('button')].find(
      (b) => b.innerText.trim() === 'Agent' && b.getAttribute('aria-pressed') === 'true',
    );
    if (!chip) return;
    chip.click();
    await sleep(600);
    if (chip.getAttribute('aria-pressed') === 'true') throw new Error('Turn off "Agent" in the Flow prompt bar');
  }

  // ---- A picture as an ingredient ----------------------------------------------------

  const addButton = () => document.querySelector('button[aria-label="Add ingredients to the prompt box"]');
  const uploadButton = () => [...document.querySelectorAll('button')].filter(visible).find((b) => /Upload media/.test(b.innerText));
  const ingredients = () => [...document.querySelectorAll('button[aria-label="Ingredient"]')].filter(visible);

  async function closePicker() {
    if (!uploadButton()) return;
    addButton()?.click();
    await sleep(500);
  }

  // Whatever the last picture left in the prompt.
  async function clearIngredients() {
    await closePicker();
    for (let i = 0; i < 6 && ingredients().length; i++) {
      ingredients()[0].click();
      await sleep(400);
    }
  }

  async function openPicker() {
    if (uploadButton()) return;
    const add = addButton();
    if (!add) throw new Error('Flow page not ready: no way to add an ingredient');
    add.click();
    for (let i = 0; i < 20 && !uploadButton(); i++) await sleep(300);
    if (!uploadButton()) throw new Error('Flow did not open its picture picker');
  }

  const setSearch = async (text) => {
    const box = document.querySelector('input[placeholder="Search assets"]');
    if (!box) return;
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(box, text);
    box.dispatchEvent(new Event('input', { bubbles: true }));
    await sleep(900);
  };

  const ingredientOption = (name) => [...document.querySelectorAll('[role="option"]')].filter(visible).find((o) => o.innerText.includes(name));
  const ready = (o) => { const i = o.querySelector('img'); return !!i && i.naturalWidth > 0; };

  async function startEdit({ aspect }) {
    if (challenge()) throw new Error('CAPTCHA: Flow is asking for a check — answer it in the Flow window');
    await clearIngredients();
    await agentOff();
    await setOptions(['Image', aspect, 'x1']);
  }

  // Whether the project has the picture, loaded, waiting up to ms for it.
  async function findIngredient(name, ms = 0) {
    await openPicker();
    await setSearch(name);
    const started = Date.now();
    do {
      const o = ingredientOption(name);
      if (o && ready(o)) return true;
      if (ms) await sleep(1000);
    } while (Date.now() - started < ms);
    return false;
  }

  // The next file input the upload button asks for is kept for the app to
  // fill, instead of opening a file dialog.
  async function armUpload() {
    document.querySelectorAll('[data-bs-upload]').forEach((e) => e.removeAttribute('data-bs-upload'));
    await openPicker();
    const click = HTMLInputElement.prototype.click;
    let armed = false;
    HTMLInputElement.prototype.click = function () {
      if (this.type !== 'file') return click.call(this);
      this.setAttribute('data-bs-upload', '1');
      if (!this.isConnected) {
        this.style.display = 'none';
        document.body.appendChild(this);
      }
      armed = true;
    };
    try {
      uploadButton().click();
      for (let i = 0; i < 20 && !armed; i++) await sleep(200);
    } finally {
      HTMLInputElement.prototype.click = click;
    }
    if (!armed) throw new Error('Flow did not offer a file to upload');
  }

  async function useIngredient(name) {
    const option = ingredientOption(name);
    if (!option) throw new Error('Flow lost the picture to change');
    option.click();
    for (let i = 0; i < 20; i++) {
      await sleep(300);
      const add = [...document.querySelectorAll('button')].filter(visible).find((b) => /add to prompt/i.test(b.innerText));
      if (add) {
        add.click();
        break;
      }
    }
    for (let i = 0; i < 30 && !ingredients().length; i++) await sleep(300);
    if (!ingredients().length) throw new Error('Flow did not take the picture into the prompt');
    await sleep(400);
  }

  async function prepareEdit({ prompt }) {
    await typePrompt(prompt);
    const go = document.querySelector('button[aria-label="Start generation"]');
    if (!go) throw new Error('Flow page not ready: no generate button');
    const r = go.getBoundingClientRect();
    return { x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2) };
  }

  async function prepare({ prompt, aspect }) {
    if (challenge()) throw new Error('CAPTCHA: Flow is asking for a check — answer it in the Flow window');
    await clearIngredients();
    await agentOff();
    await setOptions(['Image', aspect, 'x1']);
    await typePrompt(prompt);
    const go = document.querySelector('button[aria-label="Start generation"]');
    if (!go) throw new Error('Flow page not ready: no generate button');
    const r = go.getBoundingClientRect();
    return { x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2) };
  }

  // Throws when Flow asks for a check; the app reads the picture itself from
  // Flow's answer to the generation (a tile on the page can't say whose it is).
  async function watchForChallenge(timeoutMs = 600000) {
    const started = Date.now();
    while (Date.now() - started < timeoutMs) {
      await sleep(2000);
      if (challenge()) throw new Error('CAPTCHA: Flow is asking for a check — answer it in the Flow window');
    }
  }

  // ---- Clips ------------------------------------------------------------------------

  // A finished clip shows as a picture tile ("Generated video thumbnail");
  // its <video> only loads when the tile is hovered. Tile addresses change
  // (a fresh tile shows flow-content.google/image/<key>, after a reload an
  // opaque /asb/ link), so new tiles and videos are told by what was there before.
  const videoTiles = () => [...document.querySelectorAll('img')].filter((i) => /video/i.test(i.alt) && i.src);
  const videoSrcs = () => [...document.querySelectorAll('video')].map((v) => v.currentSrc || v.src).filter(Boolean);
  let beforeTiles = new Set();
  let beforeSrcs = new Set();

  // A picture's look in 16×16 colour: the same as the app computes from the
  // file, so a thumbnail can be matched to it whatever its address. (Colour,
  // not grey: green-screen characters all look alike in grey.)
  const signature = (img) => {
    const c = document.createElement('canvas');
    c.width = c.height = 16;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(img, 0, 0, 16, 16);
    const d = g.getImageData(0, 0, 16, 16).data;
    const out = new Uint8Array(768);
    for (let i = 0; i < 256; i++) for (let k = 0; k < 3; k++) out[i * 3 + k] = d[i * 4 + k];
    return out;
  };
  const distance = (a, b) => (a.length === b.length ? a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0) / a.length : Infinity);
  const loaded = (img) => (img.complete && img.naturalWidth ? Promise.resolve() : new Promise((r) => { img.onload = img.onerror = r; setTimeout(r, 3000); }));

  // The picker only keeps the thumbnails in view: the element that scrolls them.
  const scroller = (el) => {
    for (let e = el; e && e !== document.body; e = e.parentElement) if (e.scrollHeight > e.clientHeight + 10 && getComputedStyle(e).overflowY !== 'visible') return e;
    return null;
  };

  // Picks the project picture to start the clip from: by its key while its
  // tile still shows it, else by its look (\`sig\`, base64 16×16 colour),
  // scrolling through the whole picker. The best look must be close, and
  // clearly closer than the next best.
  async function chooseStart(key, sig) {
    const startButton = () => [...document.querySelectorAll('button')].filter(visible).find((b) => b.innerText.trim() === 'Start');
    let start = startButton();
    if (!start) {
      // A start frame left from the last clip: clear it (its slot, just before
      // the swap button, has a "cancel" button).
      const swap = [...document.querySelectorAll('button')].filter(visible).find((b) => b.innerText.includes('swap_horiz'));
      const clear = swap?.previousElementSibling?.querySelector('button');
      if (clear && clear.innerText.includes('cancel')) {
        clear.click();
        for (let i = 0; i < 10 && !start; i++) {
          await sleep(300);
          start = startButton();
        }
      }
    }
    if (!start) throw new Error('Flow page not ready: no Start frame button');
    // Open the picker, unless it's already open (a click would close it).
    let options = [...document.querySelectorAll('[role="option"]')].filter(visible);
    if (!options.length) start.click();
    for (let i = 0; i < 20 && !options.length; i++) {
      await sleep(400);
      options = [...document.querySelectorAll('[role="option"]')].filter(visible);
    }
    let option = options.find((o) => [...o.querySelectorAll('img')].some((im) => imageKey(im.src) === key));
    if (!option && sig) {
      const want = Uint8Array.from(atob(sig), (ch) => ch.charCodeAt(0));
      const list = options[0] ? scroller(options[0]) : null;
      const scores = new Map();
      let best = { d: Infinity, src: '', top: 0 };
      for (let step = 0; step < 80; step++) {
        for (const o of [...document.querySelectorAll('[role="option"]')].filter(visible)) {
          const img = o.querySelector('img');
          if (!img || scores.has(img.src)) continue;
          await loaded(img);
          let d = Infinity;
          try {
            d = distance(signature(img), want);
          } catch {
            /* a thumbnail from another origin can't be read */
          }
          scores.set(img.src, d);
          if (d < best.d) best = { d, src: img.src, top: list ? list.scrollTop : 0 };
        }
        if (!list || list.scrollTop + list.clientHeight >= list.scrollHeight - 2) break;
        list.scrollTop += list.clientHeight * 0.8;
        await sleep(500);
      }
      const runnerUp = Math.min(...[...scores.entries()].filter(([src]) => src !== best.src).map(([, d]) => d));
      // What the search saw, for diagnosing a miss.
      window.__bookshortsLastPick = { seen: scores.size, best: best.d, runnerUp, scrolled: !!list };
      if (best.d <= 24 && best.d + 4 < runnerUp) {
        if (list) {
          list.scrollTop = best.top;
          await sleep(600);
        }
        option = [...document.querySelectorAll('[role="option"]')].find((o) => o.querySelector('img')?.src === best.src);
      }
    }
    if (!option) {
      start.click();
      await sleep(400);
      throw new Error('Start frame not found: the picture is not in this Flow project');
    }
    option.scrollIntoView({ block: 'center' });
    option.click();
    // Flow's picker previews the picture; "Add to prompt" makes it the start frame.
    for (let i = 0; i < 15; i++) {
      await sleep(300);
      const add = [...document.querySelectorAll('button')].filter(visible).find((b) => /add to prompt/i.test(b.innerText));
      if (add) {
        add.click();
        break;
      }
    }
    for (let i = 0; i < 15 && [...document.querySelectorAll('[role="option"]')].filter(visible).length; i++) await sleep(300);
    await sleep(500);
    const slot = [...document.querySelectorAll('button')].filter(visible).find((b) => b.innerText.trim().toLowerCase() === 'start' || b.getAttribute('aria-label') === 'Start');
    if (slot && !slot.querySelector('img') && !document.querySelector('img[alt*="Start" i]')) {
      // Some layouts show the frame beside the button rather than in it: only
      // fail when no frame picture appeared anywhere near the prompt.
      const near = [...document.querySelectorAll('img')].filter(visible).some((im) => im.closest('[class*="frame" i], [aria-label*="frame" i]'));
      if (!near) throw new Error('Flow did not take the start frame (its picker changed?)');
    }
  }

  async function prepareVideo({ prompt, aspect, seconds, startKey, startSig }) {
    if (challenge()) throw new Error('CAPTCHA: Flow is asking for a check — answer it in the Flow window');
    beforeTiles = new Set(videoTiles().map((i) => i.src));
    markErrors();
    beforeSrcs = new Set(videoSrcs());
    await agentOff();
    await setOptions(['Video', 'Frames', aspect, seconds + 's', '720p', 'x1']);
    await chooseStart(startKey, startSig);
    await typePrompt(prompt);
    const go = document.querySelector('button[aria-label="Start generation"]');
    if (!go) throw new Error('Flow page not ready: no generate button');
    const r = go.getBoundingClientRect();
    return { x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2) };
  }

  // Resolves when the new clip's tile appears: its key, and where to hover
  // (CSS pixels) so Flow loads the video. A clip already loaded as a <video>
  // comes back with its URL.
  async function waitForVideo(timeoutMs = 600000) {
    const started = Date.now();
    while (Date.now() - started < timeoutMs) {
      await sleep(3000);
      if (challenge()) throw new Error('CAPTCHA: Flow is asking for a check — answer it in the Flow window');
      const refused = refusal();
      if (refused) throw new Error(refused);
      const url = videoSrcs().find((src) => !beforeSrcs.has(src));
      if (url) return { url };
      const tile = videoTiles().find((i) => !beforeTiles.has(i.src));
      if (!tile) continue;
      tile.scrollIntoView({ block: 'center' });
      await sleep(400);
      const r = tile.getBoundingClientRect();
      return { x: Math.round(r.left + r.width / 2), y: Math.round(r.top + r.height / 2), thumb: tile.src };
    }
    throw new Error('Flow did not return a clip in time');
  }

  // The clip's URL once its tile is hovered (a <video> that wasn't there), or null.
  async function videoUrl(timeoutMs = 20000) {
    const started = Date.now();
    while (Date.now() - started < timeoutMs) {
      const url = videoSrcs().find((src) => !beforeSrcs.has(src));
      if (url) return url;
      await sleep(500);
    }
    return null;
  }

  // For checking the driver without generating: forget the clips seen so far.
  function forgetVideos() {
    beforeTiles = new Set();
    beforeSrcs = new Set();
  }

  window.__bookshorts = { startEdit, findIngredient, armUpload, useIngredient, prepareEdit, clearIngredients, prepare, watchForChallenge, prepareVideo, waitForVideo, videoUrl, forgetVideos, chooseStart, setOptions };
  return 'ready';
})()`;
