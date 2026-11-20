/**
 * Turned views of the person in a portrait, made as edits of that picture (so it stays the same person), for the
 * face step instead of a head-turn clip: sharper (about one megapixel each) and no video to wait for.
 *
 *   node face_views.mjs <flow|grok> <portrait.png> <out_dir> [views]
 * views (default left45,left75): left45 = head turned about 45 degrees, left75 = about 75 degrees, both so the
 * face points toward the left side of the picture. The other side is made by mirroring (pipeline/face).
 * Writes <out_dir>/<view>.png. Prints one line of JSON.
 */
import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const TURN = {
  left45: 'the head turned about 45 degrees so the face points toward the left side of the image (a three-quarter view)',
  left75: 'the head turned about 75 degrees so the face points toward the left side of the image (almost a side profile, the far eye just visible)',
};
const prompt = (view) =>
  `Photorealistic portrait photograph of this exact same person with ${TURN[view]}, eyes looking where the face ` +
  'points. Same face, same features, same age, same skin, same hair and hairstyle, same clothes. Head and shoulders ' +
  'in frame, nothing cut off. Neutral relaxed expression, mouth closed, eyes open. Same soft even studio lighting, ' +
  'sharp focus, natural skin texture, same plain light background, no text.';

const [service, input, outDir, viewsArg] = process.argv.slice(2);
const views = (viewsArg || 'left45,left75').split(',');
mkdirSync(outDir, { recursive: true });
const results = {};
for (const view of views) {
  if (!TURN[view]) throw new Error(`Unknown view ${view}: ${Object.keys(TURN).join(', ')}`);
  const out = path.join(outDir, `${view}.png`);
  const res = execFileSync(process.execPath, [path.join(HERE, 'imagegen.mjs'), 'edit', service, '3:4', input, out, prompt(view)],
    { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'], timeout: 600000 });
  results[view] = JSON.parse(res.trim().split('\n').pop());
}
console.log(JSON.stringify({ ok: Object.values(results).every((r) => r.ok), results }));
