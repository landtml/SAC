#!/usr/bin/env node
// Renders the static SVGs under assets/ to PNG at 2x, for viewers that do not draw SVG
// (the GitHub mobile app), and composes assets/social-preview.png, the 1280x640 image
// that GitHub shows in link previews.
//
// Needs Playwright and a Chromium build. When `require('playwright')` does not resolve
// from this repository, set PLAYWRIGHT_MODULE to the playwright package path.

import { mkdir, readFile, readdir, rm } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const ASSETS = path.join(ROOT, 'assets');
const PNG_DIR = path.join(ASSETS, 'png');
const SOCIAL_PREVIEW = path.join(ASSETS, 'social-preview.png');

const ICON_SCALE = 2;
const SKIP_FILE = 'header.svg';
const SKIP_DIR = 'animations';

const FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif";
const INK = '#2B2D42';
const MUTED = '#3C4B57';
const ACCENT = '#137A55';
const PAPER = '#EEF4F2';

async function listStaticSvgs(rel = '') {
  const found = [];
  for (const entry of await readdir(path.join(ASSETS, rel), { withFileTypes: true })) {
    const child = path.join(rel, entry.name);
    if (entry.isDirectory()) {
      if (entry.name !== SKIP_DIR && entry.name !== 'png') found.push(...(await listStaticSvgs(child)));
    } else if (entry.name.endsWith('.svg') && entry.name !== SKIP_FILE) {
      found.push(child);
    }
  }
  return found.sort();
}

// The root element's width and height win; a viewBox-only file uses the viewBox size.
function svgSize(svg) {
  const open = svg.match(/<svg\b[^>]*>/)[0];
  const attr = (name) => open.match(new RegExp(`\\s${name}="([\\d.]+)"`))?.[1];
  const viewBox = open.match(/\sviewBox="([^"]+)"/)?.[1].split(/[\s,]+/).map(Number);
  const width = Number(attr('width') ?? viewBox?.[2]);
  const height = Number(attr('height') ?? viewBox?.[3]);
  if (!(width > 0 && height > 0)) throw new Error('SVG has no intrinsic size');
  return { width, height };
}

// Inline markup keeps the page free of external loads. The body stays transparent so
// element screenshots keep the alpha of SVGs that have no background rectangle.
function iconPage(svg, { width, height }) {
  const markup = svg
    .replace(/^<\?xml[^>]*>\s*/, '')
    .replace(/<svg\b/, '<svg style="display:block;width:100%;height:100%"');
  return `<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;background:transparent">
<div id="art" style="width:${width}px;height:${height}px">${markup}</div>
</body></html>`;
}

function socialPreviewPage(logoSvg) {
  const logo = logoSvg
    .replace(/^<\?xml[^>]*>\s*/, '')
    .replace(/<svg\b/, '<svg x="120" y="160" width="320" height="320"');
  return `<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0">
<div id="card" style="width:1280px;height:640px">
<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="640" viewBox="0 0 1280 640" font-family="${FONT}">
<rect width="1280" height="640" fill="${PAPER}"/>
${logo}
<rect x="600" y="176" width="96" height="6" fill="${ACCENT}"/>
<text x="600" y="300" font-size="112" font-weight="700" fill="${INK}">Soft<tspan fill="${ACCENT}">Edge</tspan></text>
<text font-size="34" fill="${MUTED}">
<tspan x="600" y="362">Soft Actor-Critic agents</tspan>
<tspan x="600" y="410">in search of a trading edge,</tspan>
<tspan x="600" y="458">measured honestly</tspan>
</text>
</svg>
</div>
</body></html>`;
}

async function main() {
  const browser = await chromium.launch();
  const written = [];
  try {
    await rm(PNG_DIR, { recursive: true, force: true });

    const iconContext = await browser.newContext({ deviceScaleFactor: ICON_SCALE });
    const page = await iconContext.newPage();
    for (const rel of await listStaticSvgs()) {
      const svg = await readFile(path.join(ASSETS, rel), 'utf8');
      const size = svgSize(svg);
      await page.setViewportSize(size);
      await page.setContent(iconPage(svg, size));
      const out = path.join(PNG_DIR, rel.replace(/\.svg$/, '.png'));
      await mkdir(path.dirname(out), { recursive: true });
      await page.locator('#art').screenshot({ path: out, omitBackground: true });
      written.push(out);
    }
    await iconContext.close();

    const socialContext = await browser.newContext({
      deviceScaleFactor: 1,
      viewport: { width: 1280, height: 640 },
    });
    const social = await socialContext.newPage();
    const logo = await readFile(path.join(ASSETS, 'logo.svg'), 'utf8');
    await social.setContent(socialPreviewPage(logo));
    await social.locator('#card').screenshot({ path: SOCIAL_PREVIEW });
    await socialContext.close();
    written.push(SOCIAL_PREVIEW);
  } finally {
    await browser.close();
  }

  for (const file of written) {
    const buf = await readFile(file);
    if (buf.length === 0) throw new Error(`empty output: ${file}`);
    const width = buf.readUInt32BE(16);
    const height = buf.readUInt32BE(20);
    console.log(`${path.relative(ROOT, file)}  ${width}x${height}  ${buf.length} bytes`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
