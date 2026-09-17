// regular.js
//
// Mesh regulariser for Urban OpenGen tiles. Pure geometry, no three.js.
//
// Idea: find the direction the city is built along, resample the (signed distance)
// channels in a frame rotated to that direction, and do every shape operation in that
// frame, where "rectangular" means axis-aligned and everything is 1D logic:
//
//   axis     dominant direction of the tile (or of each building) from the traced
//            outlines, using the angle-doubling trick for 90 degree symmetry
//   frame    bilinear resample of the channel at `up` x resolution in the rotated frame
//   smooth   morphological open + close (square radius) to remove pimples and notches
//   pieces   connected components in the frame, optionally split into height steps
//   shape    per piece: trace (Douglas-Peucker only), ortho (snap every edge to the
//            frame axes and remove edges shorter than `minEdge`), rect (greedy cover
//            with the largest inscribed rectangles) or box (bounding rectangle)
//   back     polygons go back to native tile coordinates and are clipped to the tile
//
// Streets are handled as the negative of blocks: the not-street mask is regularised
// like buildings and rendered as block plates over one street-coloured ground plate.
//
// All coordinates in the results are native grid corner coordinates (0..W, 0..H,
// y down), as floats. Heights are metres.

export const DEFAULTS = {
  axis: 'tile',          // 'tile' | 'own' | 'none'
  axisSource: 'both',    // 'both' | 'streets' | 'buildings'
  up: 2,                 // frame resolution, samples per native pixel
  open: 0.5,             // morphological opening radius in native px (removes pimples and thin spurs)
  close: 0,              // morphological closing radius in native px (fills notches and thin gaps)
  eps: 0.6,              // Douglas-Peucker tolerance in native px
  minPx: 2,              // drop pieces smaller than this many native px
  bld: 'rect',           // 'ortho' | 'rect' | 'box' | 'trace'
  minEdge: 2.5,          // ortho: edges shorter than this (native px) are removed
  boxFill: 0.85,         // pieces filling their bounding box above this ratio become the box
  minRect: 4,            // rect: smallest rectangle kept, native px^2
  hstep: 6,              // metres; > 0 splits a building into height steps before shaping
  streets: 'blocks',     // 'blocks' | 'trace'
  blocks: 'ortho',       // 'ortho' | 'rect'
  green: 'trace',        // 'trace' | 'ortho' | 'rect'
  ownConf: 0.35,         // per-building axis: minimum confidence to trust a building's own axis
  ownSnapDeg: 6,         // per-building axis: snap to the tile axis when closer than this
};

// ---------------------------------------------------------------- raster basics
export function labelComponents(mask, W, H, eight, bbox) {
  const lab = new Int32Array(W * H); let n = 0; const stack = [];
  const x0 = bbox ? bbox.x0 : 0, y0 = bbox ? bbox.y0 : 0, x1 = bbox ? bbox.x1 : W - 1, y1 = bbox ? bbox.y1 : H - 1;
  const comps = [];
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const s = y * W + x;
    if (!mask[s] || lab[s]) continue;
    n++; lab[s] = n; stack.push(s);
    const c = { id: n, px: 0, bbox: { x0: x, y0: y, x1: x, y1: y } };
    while (stack.length) {
      const i = stack.pop(), cx = i % W, cy = (i / W) | 0;
      c.px++;
      if (cx < c.bbox.x0) c.bbox.x0 = cx; if (cx > c.bbox.x1) c.bbox.x1 = cx;
      if (cy < c.bbox.y0) c.bbox.y0 = cy; if (cy > c.bbox.y1) c.bbox.y1 = cy;
      const nb = [[cx - 1, cy], [cx + 1, cy], [cx, cy - 1], [cx, cy + 1]];
      if (eight) nb.push([cx - 1, cy - 1], [cx + 1, cy - 1], [cx - 1, cy + 1], [cx + 1, cy + 1]);
      for (const [nx, ny] of nb) {
        if (nx < x0 || ny < y0 || nx > x1 || ny > y1) continue;
        const j = ny * W + nx;
        if (mask[j] && !lab[j]) { lab[j] = n; stack.push(j); }
      }
    }
    comps.push(c);
  }
  return { lab, n, comps };
}

// Boundary loops of one labelled region as lists of grid-corner points. Edges are emitted
// clockwise around each pixel so the union chains into closed loops; at pinch vertices we
// take the sharpest right turn, which keeps loops simple.
export function traceLoops(lab, id, W, H, bbox) {
  const key = (x, y) => y * (W + 1) + x;
  const out = new Map();
  const push = (x0, y0, x1, y1) => {
    const k = key(x0, y0); if (!out.has(k)) out.set(k, []);
    out.get(k).push({ x0, y0, x1, y1, used: false });
  };
  const bx0 = bbox ? bbox.x0 : 0, by0 = bbox ? bbox.y0 : 0, bx1 = bbox ? bbox.x1 : W - 1, by1 = bbox ? bbox.y1 : H - 1;
  for (let y = by0; y <= by1; y++) for (let x = bx0; x <= bx1; x++) {
    if (lab[y * W + x] !== id) continue;
    if (y === 0 || lab[(y - 1) * W + x] !== id) push(x, y, x + 1, y);
    if (x === W - 1 || lab[y * W + x + 1] !== id) push(x + 1, y, x + 1, y + 1);
    if (y === H - 1 || lab[(y + 1) * W + x] !== id) push(x + 1, y + 1, x, y + 1);
    if (x === 0 || lab[y * W + x - 1] !== id) push(x, y + 1, x, y);
  }
  const loops = [];
  for (const edges of out.values()) for (const e0 of edges) {
    if (e0.used) continue;
    const loop = []; let e = e0;
    while (e && !e.used) {
      e.used = true; loop.push([e.x0, e.y0]);
      const cands = (out.get(key(e.x1, e.y1)) || []).filter((c) => !c.used);
      if (!cands.length) break;
      if (cands.length === 1) { e = cands[0]; continue; }
      const dx = e.x1 - e.x0, dy = e.y1 - e.y0;
      cands.sort((a, b) => (dx * (b.y1 - b.y0) - dy * (b.x1 - b.x0)) - (dx * (a.y1 - a.y0) - dy * (a.x1 - a.x0)));
      e = cands[0];
    }
    if (loop.length >= 4) loops.push(loop);
  }
  return loops;
}

export function signedArea(p) { let a = 0; for (let i = 0, n = p.length; i < n; i++) { const [x0, y0] = p[i], [x1, y1] = p[(i + 1) % n]; a += x0 * y1 - x1 * y0; } return a / 2; }

function dpOpen(pts, eps) {
  if (pts.length < 3) return pts;
  const [ax, ay] = pts[0], [bx, by] = pts[pts.length - 1];
  const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
  let best = -1, bi = 0;
  for (let i = 1; i < pts.length - 1; i++) {
    const [px, py] = pts[i];
    const d = len2 ? Math.abs(dx * (ay - py) - dy * (ax - px)) / Math.sqrt(len2) : Math.hypot(px - ax, py - ay);
    if (d > best) { best = d; bi = i; }
  }
  if (best <= eps) return [pts[0], pts[pts.length - 1]];
  const l = dpOpen(pts.slice(0, bi + 1), eps), r = dpOpen(pts.slice(bi), eps);
  return l.slice(0, -1).concat(r);
}
export function simplifyLoop(loop, eps) {
  if (eps <= 0 || loop.length < 5) return loop;
  let far = 0, fd = -1; const [x0, y0] = loop[0];
  for (let i = 1; i < loop.length; i++) { const d = Math.hypot(loop[i][0] - x0, loop[i][1] - y0); if (d > fd) { fd = d; far = i; } }
  const a = dpOpen(loop.slice(0, far + 1), eps), b = dpOpen(loop.slice(far).concat([loop[0]]), eps);
  let res = a.slice(0, -1).concat(b.slice(0, -1));
  res = res.filter((p, i) => {
    const q = res[(i + res.length - 1) % res.length], r = res[(i + 1) % res.length];
    return Math.abs((p[0] - q[0]) * (r[1] - q[1]) - (p[1] - q[1]) * (r[0] - q[0])) > 1e-9;
  });
  return res.length >= 3 ? res : loop;
}
export function pointInLoop(px, py, loop) {
  let inside = false;
  for (let i = 0, j = loop.length - 1; i < loop.length; j = i++) {
    const [xi, yi] = loop[i], [xj, yj] = loop[j];
    if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// bilinear sample with pixel centres at (x + 0.5, y + 0.5); indices clamp, so sampling
// beyond the tile continues the edge pixels outward
function bilinear(arr, W, H, x, y) {
  const fx = x - 0.5, fy = y - 0.5;
  let x0 = Math.floor(fx), y0 = Math.floor(fy);
  const tx = fx - x0, ty = fy - y0;
  let x1 = x0 + 1, y1 = y0 + 1;
  if (x0 < 0) x0 = 0; if (x0 > W - 1) x0 = W - 1; if (x1 < 0) x1 = 0; if (x1 > W - 1) x1 = W - 1;
  if (y0 < 0) y0 = 0; if (y0 > H - 1) y0 = H - 1; if (y1 < 0) y1 = 0; if (y1 > H - 1) y1 = H - 1;
  return (arr[y0 * W + x0] * (1 - tx) + arr[y0 * W + x1] * tx) * (1 - ty) + (arr[y1 * W + x0] * (1 - tx) + arr[y1 * W + x1] * tx) * ty;
}
function nearestIdx(W, H, x, y) {
  let ix = Math.floor(x), iy = Math.floor(y);
  if (ix < 0) ix = 0; if (ix > W - 1) ix = W - 1; if (iy < 0) iy = 0; if (iy > H - 1) iy = H - 1;
  return iy * W + ix;
}

// ---------------------------------------------------------------- dominant direction
// Angle-doubling trick: a boundary element at angle a contributes a unit vector at 4a
// weighted by its strength, so directions 90 degrees apart add up instead of cancelling.
// The result is the axis in (-45, 45] degrees and a confidence in [0, 1] (1 = every
// edge on one grid). The boundary direction comes from the gradient of the (blurred)
// field near its 128 contour: tracing pixel outlines would vote for 0 and 90 degrees
// through every staircase step, the gradient of a distance field does not.
function blur3(a, W, H) {
  const b = new Float32Array(W * H), c = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const x0 = Math.max(x - 1, 0), x1 = Math.min(x + 1, W - 1);
    b[y * W + x] = (a[y * W + x0] + a[y * W + x] + a[y * W + x1]) / 3;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const y0 = Math.max(y - 1, 0), y1 = Math.min(y + 1, H - 1);
    c[y * W + x] = (b[y0 * W + x] + b[y * W + x] + b[y1 * W + x]) / 3;
  }
  return c;
}
export function angleAccumulate(chan, W, H, acc, lab, id, band = 48) {
  const f = blur3(blur3(chan, W, H), W, H);
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const i = y * W + x;
    if (Math.abs(f[i] - 128) > band) continue;
    if (lab) {   // only boundary pixels that touch this component
      let hit = false;
      for (let dy = -1; dy <= 1 && !hit; dy++) for (let dx = -1; dx <= 1; dx++) if (lab[i + dy * W + dx] === id) { hit = true; break; }
      if (!hit) continue;
    }
    const gx = f[i + 1] - f[i - 1], gy = f[i + W] - f[i - W], g = Math.hypot(gx, gy);
    if (g < 1) continue;
    const a = 4 * Math.atan2(gy, gx);
    acc.C += g * Math.cos(a); acc.S += g * Math.sin(a); acc.total += g;
  }
  return acc;
}
export function angleOf(acc) {
  if (!acc.total) return { theta: 0, conf: 0, total: 0 };
  return { theta: Math.atan2(acc.S, acc.C) / 4, conf: Math.hypot(acc.S, acc.C) / acc.total, total: acc.total };
}
function angleDiff90(a, b) {   // difference of two axes (mod 90 degrees), in (-45, 45]
  let d = a - b; const q = Math.PI / 2;
  d = d - Math.round(d / q) * q;
  return d;
}

// ---------------------------------------------------------------- frames
function makeFrame(cx, cy, theta, halfW, halfH, up) {
  const Fw = Math.max(2, Math.ceil(halfW * 2 * up)), Fh = Math.max(2, Math.ceil(halfH * 2 * up));
  const c = Math.cos(theta), s = Math.sin(theta);
  return {
    Fw, Fh, up, theta, cx, cy,
    toNative(fx, fy) { const x = (fx - Fw / 2) / up, y = (fy - Fh / 2) / up; return [cx + c * x - s * y, cy + s * x + c * y]; },
  };
}
function tileFrame(W, H, theta, up) {
  const c = Math.abs(Math.cos(theta)), s = Math.abs(Math.sin(theta));
  const hw = (W / 2) * c + (H / 2) * s + 1, hh = (W / 2) * s + (H / 2) * c + 1;
  return makeFrame(W / 2, H / 2, theta, hw, hh, up);
}
function compFrame(bbox, theta, up, margin) {
  const bw = bbox.x1 + 1 - bbox.x0, bh = bbox.y1 + 1 - bbox.y0;
  const c = Math.abs(Math.cos(theta)), s = Math.abs(Math.sin(theta));
  const hw = (bw / 2) * c + (bh / 2) * s + margin, hh = (bw / 2) * s + (bh / 2) * c + margin;
  return makeFrame(bbox.x0 + bw / 2, bbox.y0 + bh / 2, theta, hw, hh, up);
}

// raster of one channel in a frame: 1 where the field says inside (>= 128), optionally
// inverted, optionally restricted to a set of native component ids
function rasterFrame(frame, chan, W, H, invert, lab, ids) {
  const { Fw, Fh, up, theta, cx, cy } = frame;
  const c = Math.cos(theta), s = Math.sin(theta);
  const m = new Uint8Array(Fw * Fh);
  for (let fy = 0; fy < Fh; fy++) {
    const y = (fy + 0.5 - Fh / 2) / up;
    for (let fx = 0; fx < Fw; fx++) {
      const x = (fx + 0.5 - Fw / 2) / up;
      const nx = cx + c * x - s * y, ny = cy + s * x + c * y;
      let inside = bilinear(chan, W, H, nx, ny) >= 128;
      if (invert) inside = !inside;
      if (inside && ids && !ids.has(lab[nearestIdx(W, H, nx, ny)])) inside = false;
      m[fy * Fw + fx] = inside ? 1 : 0;
    }
  }
  return m;
}
function sampleFrame(frame, arr, W, H) {
  const { Fw, Fh, up, theta, cx, cy } = frame;
  const c = Math.cos(theta), s = Math.sin(theta);
  const f = new Float32Array(Fw * Fh);
  for (let fy = 0; fy < Fh; fy++) {
    const y = (fy + 0.5 - Fh / 2) / up;
    for (let fx = 0; fx < Fw; fx++) {
      const x = (fx + 0.5 - Fw / 2) / up;
      f[fy * Fw + fx] = bilinear(arr, W, H, cx + c * x - s * y, cy + s * x + c * y);
    }
  }
  return f;
}

// separable min / max filters with a square window of radius r
function minMax(m, W, H, r, isMax) {
  const a = new Uint8Array(W * H), b = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = isMax ? 0 : 1;
    for (let k = -r; k <= r; k++) { const xx = x + k; if (xx < 0 || xx >= W) { if (!isMax) v = 0; continue; } const s = m[y * W + xx]; if (isMax ? s > v : s < v) v = s; }
    a[y * W + x] = v;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = isMax ? 0 : 1;
    for (let k = -r; k <= r; k++) { const yy = y + k; if (yy < 0 || yy >= H) { if (!isMax) v = 0; continue; } const s = a[yy * W + x]; if (isMax ? s > v : s < v) v = s; }
    b[y * W + x] = v;
  }
  return b;
}
function openClose(m, W, H, ro, rc) {
  let o = m;
  if (ro > 0) { o = minMax(o, W, H, ro, false); o = minMax(o, W, H, ro, true); }   // open: removes pimples
  if (rc > 0) { o = minMax(o, W, H, rc, true); o = minMax(o, W, H, rc, false); }   // close: fills notches
  return o;
}

// ---------------------------------------------------------------- shapes in a frame
function rectLoop(x0, y0, x1, y1) { return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]; }

// largest all-ones rectangle inside `m` within bbox (histogram + stack, O(area))
function largestRect(m, W, bb) {
  const w = bb.x1 - bb.x0 + 1;
  const hgt = new Int32Array(w + 1), stS = new Int32Array(w + 2), stH = new Int32Array(w + 2);
  let best = { area: 0 };
  for (let y = bb.y0; y <= bb.y1; y++) {
    for (let i = 0; i < w; i++) hgt[i] = m[y * W + bb.x0 + i] ? hgt[i] + 1 : 0;
    let top = 0;
    for (let i = 0; i <= w; i++) {
      const h = i < w ? hgt[i] : 0;
      let start = i;
      while (top > 0 && stH[top - 1] > h) {
        top--;
        const s = stS[top], sh = stH[top];
        const area = sh * (i - s);
        if (area > best.area) best = { area, x0: bb.x0 + s, x1: bb.x0 + i, y0: y - sh + 1, y1: y + 1 };
        start = s;
      }
      stS[top] = start; stH[top] = h; top++;
    }
  }
  return best;
}
function rectCover(m, W, bb, minArea, maxN = 160) {
  const work = m;
  const out = [];
  while (out.length < maxN) {
    const r = largestRect(work, W, bb);
    if (r.area < minArea) break;
    out.push(r);
    for (let y = r.y0; y < r.y1; y++) for (let x = r.x0; x < r.x1; x++) work[y * W + x] = 0;
  }
  return out;
}

function segsCross(a, b, c, d) {
  const o = (p, q, r) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
  const o1 = o(a, b, c), o2 = o(a, b, d), o3 = o(c, d, a), o4 = o(c, d, b);
  return o1 * o2 < 0 && o3 * o4 < 0;
}
function selfIntersects(p) {
  const n = p.length;
  for (let i = 0; i < n; i++) for (let j = i + 2; j < n; j++) {
    if (i === 0 && j === n - 1) continue;
    if (segsCross(p[i], p[(i + 1) % n], p[j], p[(j + 1) % n])) return true;
  }
  return false;
}

// Snap a simplified loop to the frame axes. Every edge becomes horizontal or vertical
// (whichever it is closer to) at its length-weighted position, consecutive edges with the
// same direction merge, and the polygon is then just an alternating list of x and y
// coordinates: vertex k is the corner of edge k-1 and edge k. Edges shorter than minEdge
// are removed by merging their two (parallel) neighbours, shortest first, until only a
// rectangle would be left. Returns null when the result degenerates (caller uses the box).
export function orthogonalise(loop, minEdge, maxDevDeg = 30) {
  const n = loop.length; if (n < 3) return null;
  const E = [];
  const maxDev = maxDevDeg * Math.PI / 180;
  for (let i = 0; i < n; i++) {
    const [x0, y0] = loop[i], [x1, y1] = loop[(i + 1) % n];
    const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy); if (L < 1e-9) continue;
    const h = Math.abs(dx) >= Math.abs(dy);
    // a long edge that is closer to the diagonal than to an axis means this outline is not
    // rectilinear in this frame; snapping it would fold the polygon, so leave it to the caller
    if (L >= 2 * minEdge && Math.atan2(Math.min(Math.abs(dx), Math.abs(dy)), Math.max(Math.abs(dx), Math.abs(dy))) > maxDev) return null;
    E.push({ h, c: h ? (y0 + y1) / 2 : (x0 + x1) / 2, w: L });
  }
  const M = [];
  for (const e of E) {
    const t = M[M.length - 1];
    if (t && t.h === e.h) { t.c = (t.c * t.w + e.c * e.w) / (t.w + e.w); t.w += e.w; } else M.push({ ...e });
  }
  if (M.length > 1 && M[0].h === M[M.length - 1].h) { const t = M.pop(), f = M[0]; f.c = (f.c * f.w + t.c * t.w) / (f.w + t.w); f.w += t.w; }
  if (M.length < 4) return null;
  let C = M.map((e) => e.c), Hd = M.map((e) => e.h);
  const edgeLen = (k) => { const m = C.length; return Math.abs(C[(k + 1) % m] - C[(k - 1 + m) % m]); };
  while (C.length > 4) {
    const m = C.length; let si = -1, sl = Infinity;
    for (let k = 0; k < m; k++) { const l = edgeLen(k); if (l < sl) { sl = l; si = k; } }
    if (sl >= minEdge) break;
    const a = (si - 1 + m) % m, b = (si + 1) % m;
    const la = Math.abs(C[si] - C[(a - 1 + m) % m]), lb = Math.abs(C[(b + 1) % m] - C[si]);
    C[a] = (C[a] * la + C[b] * lb) / ((la + lb) || 1);
    const drop = new Set([si, b]);
    C = C.filter((_, k) => !drop.has(k)); Hd = Hd.filter((_, k) => !drop.has(k));
  }
  const m = C.length;
  let pts = [];
  for (let k = 0; k < m; k++) { const p = (k - 1 + m) % m; pts.push(Hd[k] ? [C[p], C[k]] : [C[k], C[p]]); }
  pts = pts.filter((q, k) => { const r = pts[(k + 1) % pts.length]; return Math.abs(q[0] - r[0]) > 1e-6 || Math.abs(q[1] - r[1]) > 1e-6; });
  if (pts.length < 4 || Math.abs(signedArea(pts)) < 1e-6 || selfIntersects(pts)) return null;
  return pts;
}

// Sutherland-Hodgman clip of a polygon to the axis-aligned rectangle [x0,x1] x [y0,y1]
export function clipToRect(poly, x0, y0, x1, y1) {
  let out = poly;
  const sides = [
    (p) => p[0] >= x0, (p) => p[0] <= x1, (p) => p[1] >= y0, (p) => p[1] <= y1,
  ];
  const cut = [
    (a, b) => { const t = (x0 - a[0]) / (b[0] - a[0]); return [x0, a[1] + t * (b[1] - a[1])]; },
    (a, b) => { const t = (x1 - a[0]) / (b[0] - a[0]); return [x1, a[1] + t * (b[1] - a[1])]; },
    (a, b) => { const t = (y0 - a[1]) / (b[1] - a[1]); return [a[0] + t * (b[0] - a[0]), y0]; },
    (a, b) => { const t = (y1 - a[1]) / (b[1] - a[1]); return [a[0] + t * (b[0] - a[0]), y1]; },
  ];
  for (let s = 0; s < 4; s++) {
    const inp = out; out = [];
    if (!inp.length) break;
    let prev = inp[inp.length - 1], pin = sides[s](prev);
    for (const cur of inp) {
      const cin = sides[s](cur);
      if (cin) { if (!pin) out.push(cut[s](prev, cur)); out.push(cur); }
      else if (pin) out.push(cut[s](prev, cur));
      prev = cur; pin = cin;
    }
  }
  out = out.filter((p, i) => { const q = out[(i + 1) % out.length]; return Math.hypot(p[0] - q[0], p[1] - q[1]) > 1e-6; });
  return out.length >= 3 ? out : null;
}

function median(vals) { if (!vals.length) return 0; vals.sort((a, b) => a - b); return vals[vals.length >> 1]; }
function medianIn(hF, W, m, bb, id) {
  if (!hF) return null;
  const v = [];
  for (let y = bb.y0; y <= bb.y1; y++) for (let x = bb.x0; x <= bb.x1; x++) if (m[y * W + x] === id) v.push(hF[y * W + x]);
  return median(v);
}
function medianRect(hF, W, r) {
  if (!hF) return null;
  const v = [];
  for (let y = r.y0; y < r.y1; y++) for (let x = r.x0; x < r.x1; x++) v.push(hF[y * W + x]);
  return median(v);
}

// split one piece into sub-pieces of similar height: quantise, majority-filter the bins so
// single-pixel steps vanish, absorb bins too small to be a building step. Works on the
// piece's bbox-local crop: loc[i] = 1 inside the piece, hL[i] = height in metres.
function splitByHeight(loc, bw, bh, hL, step, minArea) {
  const n = bw * bh;
  const bins = new Int16Array(n);
  for (let i = 0; i < n; i++) bins[i] = loc[i] ? Math.round(hL[i] / step) : -32768;
  const nb = new Int16Array(n);
  const vals = new Int16Array(9), cnts = new Float32Array(9);
  const majority = () => {
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const i = y * bw + x; if (!loc[i]) { nb[i] = bins[i]; continue; }
      let best = bins[i], bc = 0, nv = 0;
      // count the up-to-9 neighbour bins (centre counts 1.5) with a tiny linear scan
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= bw || yy >= bh) continue;
        const j = yy * bw + xx; if (!loc[j]) continue;
        const b = bins[j], w = (dx === 0 && dy === 0) ? 1.5 : 1;
        let k = 0; while (k < nv && vals[k] !== b) k++;
        if (k === nv) { vals[nv] = b; cnts[nv] = 0; nv++; }
        cnts[k] += w; if (cnts[k] > bc) { bc = cnts[k]; best = b; }
      }
      nb[i] = best;
    }
    bins.set(nb);
  };
  majority(); majority();
  const rl = new Int32Array(n); const stack = [];
  const label = () => {   // connected regions of equal bin -> rl, returns [{id, px, bbox}]
    rl.fill(0); let k = 0; const regs = [];
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const s = y * bw + x; if (!loc[s] || rl[s]) continue;
      k++; rl[s] = k; stack.push(s); const b0 = bins[s];
      const r = { id: k, px: 0, bbox: { x0: x, y0: y, x1: x, y1: y } };
      while (stack.length) {
        const i = stack.pop(), cx = i % bw, cy = (i / bw) | 0; r.px++;
        if (cx < r.bbox.x0) r.bbox.x0 = cx; if (cx > r.bbox.x1) r.bbox.x1 = cx; if (cy < r.bbox.y0) r.bbox.y0 = cy; if (cy > r.bbox.y1) r.bbox.y1 = cy;
        if (cx > 0) { const j = i - 1; if (loc[j] && !rl[j] && bins[j] === b0) { rl[j] = k; stack.push(j); } }
        if (cx < bw - 1) { const j = i + 1; if (loc[j] && !rl[j] && bins[j] === b0) { rl[j] = k; stack.push(j); } }
        if (cy > 0) { const j = i - bw; if (loc[j] && !rl[j] && bins[j] === b0) { rl[j] = k; stack.push(j); } }
        if (cy < bh - 1) { const j = i + bw; if (loc[j] && !rl[j] && bins[j] === b0) { rl[j] = k; stack.push(j); } }
      }
      regs.push(r);
    }
    return regs;
  };
  for (let it = 0; it < 8; it++) {
    const regs = label();
    const small = new Set(regs.filter((r) => r.px < minArea).map((r) => r.id));
    if (!small.size || regs.length === 1) break;
    // each small region takes the most common bin among its neighbours in other regions
    const vote = new Map();
    for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      const i = y * bw + x; if (!loc[i] || !small.has(rl[i])) continue;
      const cast = (j) => { if (loc[j] && rl[j] !== rl[i]) { const key = rl[i] * 65536 + (bins[j] + 32768); vote.set(key, (vote.get(key) || 0) + 1); } };
      if (x > 0) cast(i - 1); if (x < bw - 1) cast(i + 1); if (y > 0) cast(i - bw); if (y < bh - 1) cast(i + bw);
    }
    const best = new Map();
    for (const [key, c] of vote) { const r = Math.floor(key / 65536), b = (key % 65536) - 32768; const cur = best.get(r); if (!cur || c > cur[1]) best.set(r, [b, c]); }
    if (!best.size) break;
    for (let i = 0; i < n; i++) if (loc[i] && small.has(rl[i])) { const b = best.get(rl[i]); if (b) bins[i] = b[0]; }
  }
  return { lab: rl, regions: label() };
}

// shapes for one labelled piece in a frame -> list of {outer, holes, h} in frame coords.
// Everything runs on the piece's bbox-local crop and is offset back at the end.
function shapePiece(lab, id, frame, bb, hF, method, o, fixedH, stats) {
  const { Fw, up } = frame;
  const bw = bb.x1 - bb.x0 + 1, bh = bb.y1 - bb.y0 + 1, n = bw * bh;
  const loc = new Int32Array(n); const hL = hF ? new Float32Array(n) : null;
  let area = 0;
  for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
    const g = (y + bb.y0) * Fw + x + bb.x0, i = y * bw + x;
    if (lab[g] === id) { loc[i] = 1; area++; if (hL) hL[i] = hF[g]; }
  }
  const out = [];
  let subs;
  if (o.hstep > 0 && hL && method !== 'trace') subs = splitByHeight(loc, bw, bh, hL, o.hstep, Math.max(2, o.minPx * up * up));
  else subs = { lab: loc, regions: [{ id: 1, bbox: { x0: 0, y0: 0, x1: bw - 1, y1: bh - 1 }, px: area }] };
  const full = { x0: 0, y0: 0, x1: bw - 1, y1: bh - 1 };
  for (const r of subs.regions) {
    const sl = subs.lab, sid = r.id, sb = r.bbox;
    const bbArea = (sb.x1 - sb.x0 + 1) * (sb.y1 - sb.y0 + 1);
    const hPiece = () => fixedH !== null ? fixedH : medianIn(hL, bw, sl, sb, sid);
    // trace first: the box shortcut is only taken by hole-free pieces that fill their box
    const eps = Math.max(o.eps * up, method === 'ortho' ? 1.0 : 0);
    const loops = traceLoops(sl, sid, bw, bh, sb).map((l) => ({ pts: simplifyLoop(l, eps), outer: signedArea(l) > 0 }));
    const nHoles = loops.filter((l) => !l.outer && Math.abs(signedArea(l.pts)) >= o.minPx * up * up).length;
    let m = method;
    if (m !== 'trace' && !nHoles && r.px / bbArea >= o.boxFill) m = 'box';
    if (m === 'box') {
      out.push({ outer: rectLoop(sb.x0, sb.y0, sb.x1 + 1, sb.y1 + 1), holes: [], h: hPiece() });
      stats.boxes++;
    } else if (m === 'rect') {
      const mask = new Uint8Array(n);
      for (let y = sb.y0; y <= sb.y1; y++) for (let x = sb.x0; x <= sb.x1; x++) mask[y * bw + x] = sl[y * bw + x] === sid ? 1 : 0;
      const rects = rectCover(mask, bw, sb, Math.max(1, o.minRect * up * up));
      for (const rc of rects) { out.push({ outer: rectLoop(rc.x0, rc.y0, rc.x1, rc.y1), holes: [], h: fixedH !== null ? fixedH : medianRect(hL, bw, rc) }); stats.rects++; }
    } else {
      const outers = [], holes = [];
      for (const l of loops) {
        let p = l.pts;
        if (m === 'ortho') {
          const q = orthogonalise(p, o.minEdge * up);
          if (q) p = q;
          else {   // degenerate snap: keep the simplified outline (a box only for tiny pieces)
            stats.fallback++;
            if (o.debug) (stats.failed = stats.failed || []).push(p);
            if (l.outer && p.length <= 6) p = rectLoop(...bboxOf(p));
          }
        }
        if (Math.abs(signedArea(p)) < 0.5) continue;
        (l.outer ? outers : holes).push(p);
      }
      const h = hPiece();
      for (const op of outers) {
        const hs = holes.filter((hl) => pointInLoop(hl[0][0] + 1e-3, hl[0][1] + 1e-3, op));
        out.push({ outer: op, holes: hs, h });
      }
      stats.traced += outers.length;
    }
  }
  // back to frame coordinates
  for (const s of out) {
    s.outer = s.outer.map(([x, y]) => [x + bb.x0, y + bb.y0]);
    s.holes = s.holes.map((hl) => hl.map(([x, y]) => [x + bb.x0, y + bb.y0]));
  }
  void full;
  return out;
}
function bboxOf(p) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of p) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  return [x0, y0, x1, y1];
}

// ---------------------------------------------------------------- one class, end to end
function processClass(ctx, chan, invert, eight, method, heightsArr, fixedH, o, stats, axisOverride) {
  const { W, H, ax } = ctx;
  const nat = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) nat[i] = (chan[i] >= 128) !== invert ? 1 : 0;
  const { lab, comps } = labelComponents(nat, W, H, eight);
  const minPx = o.minPx;
  // which frames: one for the tile, or one per native component with its own axis
  const frames = [];
  const axisMode = axisOverride || o.axis;
  if (axisMode === 'own') {
    for (const c of comps) {
      if (c.px < minPx) continue;
      let theta = ax.theta;
      const own = angleOf(angleAccumulate(chan, W, H, { C: 0, S: 0, total: 0 }, lab, c.id));
      if (own.conf >= o.ownConf && own.total >= 40 && Math.abs(angleDiff90(own.theta, ax.theta)) > o.ownSnapDeg * Math.PI / 180) theta = own.theta;
      frames.push({ frame: compFrame(c.bbox, theta, o.up, 2 + Math.max(o.open, o.close)), ids: new Set([c.id]) });
    }
  } else {
    frames.push({ frame: tileFrame(W, H, axisMode === 'none' ? 0 : ax.theta, o.up), ids: null });
  }
  const results = [];
  for (const { frame, ids } of frames) {
    const { Fw, Fh, up } = frame;
    let m = rasterFrame(frame, chan, W, H, invert, lab, ids);
    m = openClose(m, Fw, Fh, Math.round(o.open * up), Math.round(o.close * up));
    const hF = heightsArr ? sampleFrame(frame, heightsArr, W, H) : null;
    const fl = labelComponents(m, Fw, Fh, eight);
    for (const c of fl.comps) {
      if (c.px < minPx * up * up) continue;
      const shapes = shapePiece(fl.lab, c.id, frame, c.bbox, hF, method, o, fixedH, stats);
      for (const s of shapes) {
        const outer = clipToRect(s.outer.map(([x, y]) => frame.toNative(x, y)), 0, 0, W, H);
        if (!outer) continue;
        const holes = [];
        for (const hl of s.holes) {
          const hn = hl.map(([x, y]) => frame.toNative(x, y));
          if (hn.every(([x, y]) => x > 0.01 && y > 0.01 && x < W - 0.01 && y < H - 0.01)) holes.push(hn);
        }
        results.push({ outer, holes, h: s.h });
      }
    }
  }
  return results;
}

// ---------------------------------------------------------------- entry point
// tile: { W, H, fp, hgt, st, gr (Uint8Array bytes), heights (Float32Array metres, every pixel) }
export function regularise(tile, options = {}) {
  const o = { ...DEFAULTS, ...options };
  const t0 = performance.now();
  const { W, H, fp, st, gr, heights } = tile;
  const stats = { boxes: 0, rects: 0, traced: 0, fallback: 0 };

  // tile axis from the boundary gradients of the chosen channels
  let ax = { theta: 0, conf: 0, total: 0 };
  if (o.axis !== 'none') {
    const acc = { C: 0, S: 0, total: 0 };
    if (o.axisSource !== 'buildings') angleAccumulate(st, W, H, acc);
    if (o.axisSource !== 'streets') angleAccumulate(fp, W, H, acc);
    ax = angleOf(acc);
  }
  const ctx = { W, H, ax };

  const buildings = processClass(ctx, fp, false, false, o.bld, heights, null, o, stats);
  const greens = processClass(ctx, gr, false, false, o.green, null, 1, o, stats);
  let blocks = [], streets = [];
  if (o.streets === 'blocks') blocks = processClass(ctx, st, true, false, o.blocks, null, 1, o, stats);
  else streets = processClass(ctx, st, false, true, 'trace', null, 1, o, stats, 'none');

  return {
    axis: { deg: ax.theta * 180 / Math.PI, conf: ax.conf },
    buildings, greens, blocks, streets,
    stats: { ...stats, ms: performance.now() - t0 },
  };
}
