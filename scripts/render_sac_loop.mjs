#!/usr/bin/env node
// Renders assets/animations/sac-loop.svg to assets/animations/sac-loop.gif.
//
// The SVG animates with CSS keyframes. Every frame pauses each animation and seeks it to the
// same time, so a frame is exact and the first and last frames of the loop match. Chromium comes
// from Playwright; ffmpeg builds the palette and the GIF.
//
// Usage: node scripts/render_sac_loop.mjs [--frames DIR] [--width 960] [--fps 20]
// --frames keeps the PNG frames in DIR (default: a temporary directory that is removed).
// Set PLAYWRIGHT_MODULE to the playwright package directory when it is not in node_modules.

import { execFileSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync, statSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join, dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const svgPath = join(root, 'assets', 'animations', 'sac-loop.svg');
const gifPath = join(root, 'assets', 'animations', 'sac-loop.gif');
const LOOP_MS = 10000;
const MAX_BYTES = 6 * 1024 * 1024;
const VIEW = { width: 1200, height: 560 };

const args = process.argv.slice(2);
const opt = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const outWidth = Number(opt('--width', '960'));
const fps = Number(opt('--fps', '20'));
const keepDir = opt('--frames', null);

const require = createRequire(import.meta.url);
function loadPlaywright() {
  for (const id of [process.env.PLAYWRIGHT_MODULE, 'playwright']) {
    if (!id) continue;
    try {
      return require(id);
    } catch {
      // try the next candidate
    }
  }
  throw new Error('playwright not found: install it or set PLAYWRIGHT_MODULE to its directory');
}

async function renderFrames(dir) {
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage({ viewport: VIEW, deviceScaleFactor: 1 });
    await page.goto(pathToFileURL(svgPath).href);
    const parseError = await page.evaluate(() => {
      const err = document.querySelector('parsererror');
      return err ? err.textContent : null;
    });
    if (parseError) throw new Error(`SVG does not parse: ${parseError.slice(0, 300)}`);
    const count = await page.evaluate(() => document.getAnimations().length);
    if (count === 0) throw new Error('no CSS animations found in the SVG');

    const frames = Math.round((LOOP_MS / 1000) * fps);
    for (let i = 0; i < frames; i++) {
      const t = ((i + 0.5) * 1000) / fps; // mid-frame, so steps() boundaries never round down
      await page.evaluate((time) => {
        for (const a of document.getAnimations()) {
          a.pause();
          a.currentTime = time;
        }
      }, t);
      const name = join(dir, `frame-${String(i).padStart(3, '0')}.png`);
      await page.screenshot({ path: name, clip: { x: 0, y: 0, ...VIEW } });
    }
    return { frames, animations: count };
  } finally {
    await browser.close();
  }
}

function buildGif(dir, frames) {
  const pattern = join(dir, 'frame-%03d.png');
  const palette = join(dir, 'palette.png');
  const scale = `scale=${outWidth}:-1:flags=lanczos`;
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', pattern,
    '-vf', `${scale},palettegen=stats_mode=diff`, palette], { stdio: 'inherit' });
  execFileSync('ffmpeg', ['-y', '-loglevel', 'error', '-framerate', String(fps), '-i', pattern,
    '-i', palette, '-lavfi', `${scale}[x];[x][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle`,
    '-loop', '0', gifPath], { stdio: 'inherit' });
  return frames;
}

async function main() {
  const dir = keepDir ? resolve(keepDir) : mkdtempSync(join(tmpdir(), 'sac-loop-'));
  mkdirSync(dir, { recursive: true });
  try {
    const { frames, animations } = await renderFrames(dir);
    buildGif(dir, frames);
    const bytes = statSync(gifPath).size;
    console.log(`animations: ${animations}, frames: ${frames}, fps: ${fps}, width: ${outWidth}`);
    console.log(`gif: ${gifPath}`);
    console.log(`size: ${bytes} bytes (${(bytes / 1048576).toFixed(2)} MiB)`);
    if (bytes >= MAX_BYTES) throw new Error('GIF is 6 MiB or larger; lower --width or --fps');
  } finally {
    if (!keepDir) rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((err) => {
  console.error(err.message);
  process.exit(1);
});
