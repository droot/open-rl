// The wireframe landscape from the OpenRL deck, drawn once behind the hero.
// A height field of low hills under a perspective camera, lit from the left.
(() => {
  'use strict';
  const canvas = document.getElementById('landscape');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const HILLS = [
    { x:  0.42, y:  0.12, a: 0.23, s: 0.30 },
    { x:  1.18, y: -0.02, a: 0.22, s: 0.30 },
    { x: -0.40, y:  0.42, a: 0.18, s: 0.30 },
    { x: -1.20, y: -0.12, a: 0.20, s: 0.27 },
    { x: -0.30, y: -0.58, a: 0.15, s: 0.32 },
    { x:  1.45, y:  0.62, a: 0.13, s: 0.25 },
    { x: -1.55, y:  0.62, a: 0.12, s: 0.24 },
  ];
  const WORLD = { x0: -3.3, x1: 3.3, y0: -1.75, y1: 1.45 };
  const GRID  = { cols: 120, rows: 64 };
  const CAM   = { pos: [0, -3.6, 1.95], yaw: -0.16, cyFrac: 0.66, fScale: 0.82 };
  const LIGHT = normalize([-0.45, -0.35, 0.82]);
  const COLORS = { paper: [0xF2, 0xEE, 0xDE], light: [0xFC, 0xFA, 0xF2], shade: [0xD3, 0xCD, 0xB8], ink: [0x1A, 0x1A, 0x1A] };

  function normalize(v) { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function mix(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }
  function css(c, alpha) { return alpha === undefined ? `rgb(${c[0]|0},${c[1]|0},${c[2]|0})` : `rgba(${c[0]|0},${c[1]|0},${c[2]|0},${alpha})`; }
  function smoothstep(a, b, x) { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); }

  function height(x, y) {
    let z = 0;
    for (const h of HILLS) { const dx = x - h.x, dy = y - h.y; z += h.a * Math.exp(-(dx * dx + dy * dy) / (2 * h.s * h.s)); }
    z += 0.012 * Math.sin(3.1 * x + 0.7) * Math.cos(2.3 * y - 0.4) + 0.008 * Math.sin(6.3 * x - 1.9 + 2.1 * y);
    return z;
  }

  const phi = Math.atan2(CAM.pos[2], -CAM.pos[1]);
  const cph = Math.cos(phi), sph = Math.sin(phi), cyw = Math.cos(CAM.yaw), syw = Math.sin(CAM.yaw);
  let W = 0, H = 0;
  function project(x, y, z, out) {
    const xr = x * cyw - y * syw, yr = x * syw + y * cyw;
    const px = xr - CAM.pos[0], py = yr - CAM.pos[1], pz = z - CAM.pos[2];
    const depth = py * cph - pz * sph, sy = py * sph + pz * cph, F = W * CAM.fScale;
    out[0] = W * 0.5 + F * px / depth; out[1] = H * CAM.cyFrac - F * sy / depth; out[2] = depth; return out;
  }

  const NV = (GRID.cols + 1) * (GRID.rows + 1);
  const SX = new Float32Array(NV), SY = new Float32Array(NV), SD = new Float32Array(NV);
  const WX = new Float32Array(NV), WY = new Float32Array(NV), WZ = new Float32Array(NV);
  const order = Array.from({ length: GRID.cols * GRID.rows }, (_, i) => i);
  const depth = new Float32Array(GRID.cols * GRID.rows);
  const tmp = [0, 0, 0];

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (let i = 0; i <= GRID.rows; i++) for (let j = 0; j <= GRID.cols; j++) {
      const n = i * (GRID.cols + 1) + j;
      const x = WORLD.x0 + (WORLD.x1 - WORLD.x0) * j / GRID.cols, y = WORLD.y0 + (WORLD.y1 - WORLD.y0) * i / GRID.rows;
      const z = height(x, y);
      WX[n] = x; WY[n] = y; WZ[n] = z;
      project(x, y, z, tmp); SX[n] = tmp[0]; SY[n] = tmp[1]; SD[n] = tmp[2];
    }
    for (let i = 0, q = 0; i < GRID.rows; i++) for (let j = 0; j < GRID.cols; j++, q++) {
      const a = i * (GRID.cols + 1) + j; depth[q] = SD[a] + SD[a + 1] + SD[a + GRID.cols + 1] + SD[a + GRID.cols + 2];
    }
    order.sort((m, n) => depth[n] - depth[m]);
    ctx.lineWidth = 0.85; ctx.lineJoin = 'round';
    const yFade0 = WORLD.y1 - 0.95, yFade1 = WORLD.y1 - 0.08;
    for (let k = 0; k < order.length; k++) {
      const qi = order[k], i = (qi / GRID.cols) | 0, j = qi - i * GRID.cols;
      const a = i * (GRID.cols + 1) + j, b = a + 1, c = a + GRID.cols + 2, d = a + GRID.cols + 1;
      const minX = Math.min(SX[a], SX[b], SX[c], SX[d]), maxX = Math.max(SX[a], SX[b], SX[c], SX[d]);
      const minY = Math.min(SY[a], SY[b], SY[c], SY[d]), maxY = Math.max(SY[a], SY[b], SY[c], SY[d]);
      if (maxX < -2 || minX > W + 2 || maxY < -2 || minY > H + 2) continue;
      const e1x = WX[c] - WX[a], e1y = WY[c] - WY[a], e1z = WZ[c] - WZ[a];
      const e2x = WX[d] - WX[b], e2y = WY[d] - WY[b], e2z = WZ[d] - WZ[b];
      let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
      const nl = Math.hypot(nx, ny, nz) || 1; nx /= nl; ny /= nl; nz /= nl;
      if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
      const lambert = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2];
      const fade = 1 - smoothstep(yFade0, yFade1, (WY[a] + WY[c]) * 0.5);
      const t = (lambert - LIGHT[2]) * 2.4;
      let fill = t >= 0 ? mix(COLORS.paper, COLORS.light, clamp(t, 0, 1)) : mix(COLORS.paper, COLORS.shade, clamp(-t, 0, 1));
      fill = mix(fill, COLORS.paper, 1 - fade);
      ctx.beginPath(); ctx.moveTo(SX[a], SY[a]); ctx.lineTo(SX[b], SY[b]); ctx.lineTo(SX[c], SY[c]); ctx.lineTo(SX[d], SY[d]); ctx.closePath();
      ctx.fillStyle = css(fill); ctx.fill();
      const alpha = 0.42 * fade;
      if (alpha > 0.02) { ctx.strokeStyle = css(COLORS.ink, alpha); ctx.stroke(); }
    }
  }

  function resize() {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  let timer = 0;
  window.addEventListener('resize', () => { clearTimeout(timer); timer = setTimeout(resize, 120); });
  resize();
})();
