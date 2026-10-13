#!/usr/bin/env node
// Renders the README header from assets/header.svg into assets/header.gif (one 8 s loop at 20 fps,
// 1000 px wide) and assets/header.png (the settled frame at 6 s, at 2x). Frames are drawn by
// setting the clock of every CSS animation, so a run produces the same frames each time.
//
// Usage: node scripts/render_hero.mjs [--preview <dir>]
//   --preview keeps the 160 source frames in <dir> for review. Needs ffmpeg and Playwright
//   with Chromium. Set PLAYWRIGHT_MODULE to the playwright package when it is not resolvable
//   from this directory.

import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const ASSETS = join(ROOT, 'assets');
const LOOP_MS = 8000;
const FPS = 20;
const GIF_WIDTH = 1000;
const GIF_MAX_BYTES = 5 * 1024 * 1024;
const VIEW = { width: 1280, height: 440 };

const previewAt = process.argv.indexOf('--preview');
const previewDir = previewAt > 0 ? resolve(process.argv[previewAt + 1]) : null;

const frameTimes = Array.from({ length: (LOOP_MS / 1000) * FPS }, (_, i) =>
  Math.round((i * 1000) / FPS),
);

async function capture(browser, svg, { scale, times, pathFor }) {
  const context = await browser.newContext({ viewport: VIEW, deviceScaleFactor: scale });
  const page = await context.newPage();
  await page.setContent(
    '<!doctype html><html><head><meta charset="utf-8"><style>' +
      'html,body{margin:0;padding:0}svg{display:block}</style></head><body>' +
      svg +
      '</body></html>',
  );
  for (const [i, ms] of times.entries()) {
    await page.evaluate((clock) => {
      for (const animation of document.getAnimations()) {
        animation.pause();
        animation.currentTime = clock;
      }
    }, ms);
    await page.screenshot({ path: pathFor(i), clip: { x: 0, y: 0, ...VIEW } });
  }
  await context.close();
}

const svg = readFileSync(join(ASSETS, 'header.svg'), 'utf8');
const workDir = previewDir ?? mkdtempSync(join(tmpdir(), 'hero-'));
mkdirSync(workDir, { recursive: true });

const browser = await chromium.launch();
try {
  await capture(browser, svg, {
    scale: 1,
    times: frameTimes,
    pathFor: (i) => join(workDir, `frame-${String(i).padStart(3, '0')}.png`),
  });
  await capture(browser, svg, {
    scale: 2,
    times: [6000],
    pathFor: () => join(ASSETS, 'header.png'),
  });
} finally {
  await browser.close();
}

const gifPath = join(ASSETS, 'header.gif');
execFileSync(
  'ffmpeg',
  [
    '-y',
    '-loglevel', 'error',
    '-start_number', '0',
    '-framerate', String(FPS),
    '-i', join(workDir, 'frame-%03d.png'),
    '-filter_complex',
    `[0:v]scale=${GIF_WIDTH}:-2:flags=lanczos,split[a][b];` +
      '[a]palettegen=max_colors=128:stats_mode=diff[p];' +
      '[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
    '-loop', '0',
    gifPath,
  ],
  { stdio: 'inherit' },
);

if (!previewDir) rmSync(workDir, { recursive: true, force: true });

const bytes = statSync(gifPath).size;
if (bytes > GIF_MAX_BYTES) {
  throw new Error(`header.gif is ${bytes} bytes, over the ${GIF_MAX_BYTES} byte limit`);
}
console.log(`header.gif ${bytes} bytes, ${frameTimes.length} frames`);
