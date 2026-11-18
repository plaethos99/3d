/**
 * Face close-ups for film quality: the same character's face filling the picture, from the front, three-quarter
 * and side, made by changing the head turnaround (so it stays the same person). Flow makes about one megapixel
 * a picture; framing the face tightly gives it 2-3 times the pixels it has in a turnaround.
 *
 *   node face_closeups.mjs <flow|grok> <head_turnaround.jpg> <out_dir> [views]
 * Writes <out_dir>/face_<view>.jpg for views front, three_quarter, left (default all three). Prints JSON.
 */
import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const VIEW_WORDS = {
  front: 'front view, looking straight into the camera, face perfectly frontal and level',
  three_quarter: 'three-quarter view, head turned 45 degrees so she faces toward the left side of the image, eyes looking where her face points',
  left: 'exact side profile facing the left side of the image, the whole profile visible',
};
const prompt = (view) =>
  `Photorealistic close-up portrait photograph of this exact same woman, ${VIEW_WORDS[view]}. Her head and hair fill ` +
  'the frame from just above the top of her hair to just below her chin, nothing cut off. Same face, same features, ' +
  'same short dark brown bob haircut, neutral relaxed expression, mouth closed, eyes open, no makeup change. Soft even ' +
  'studio lighting from the front, sharp focus on the whole face, fine skin detail, plain pure white background, no text.';

const [service, input, outDir, viewsArg] = process.argv.slice(2);
const views = (viewsArg || 'front,three_quarter,left').split(',');
mkdirSync(outDir, { recursive: true });
const results = {};
for (const view of views) {
  const out = path.join(outDir, `face_${view}.jpg`);
  const res = execFileSync(process.execPath, [path.join(HERE, 'imagegen.mjs'), 'edit', service, '3:4', input, out, prompt(view)],
    { encoding: 'utf8', stdio: ['ignore', 'pipe', 'inherit'], timeout: 600000 });
  results[view] = JSON.parse(res.trim().split('\n').pop());
}
console.log(JSON.stringify({ ok: Object.values(results).every((r) => r.ok), results }));
