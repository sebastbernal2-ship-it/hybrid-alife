/* hybrid-alife immersed-boundary viewer. WebGL2, no dependencies.
 * World frame: lattice cells, node (i, j) at (i, j), y up; periodic in x.
 * Time `t` is in recorded frames; everything between records is interpolated. */
'use strict';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const canvas = $('view');
const gl = canvas.getContext('webgl2', { antialias: true, alpha: false, preserveDrawingBuffer: true });

function fail(message) {
  $('loading').hidden = false;
  $('loading').textContent = message;
}

/* ---------- replay loading (mirrors replay.py) ---------- */
async function loadReplay() {
  let source;
  if (window.REPLAY_B64) {
    const binary = atob(window.REPLAY_B64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    source = new Blob([bytes]);
  } else {
    const response = await fetch(params.get('replay') || 'replay.hvr');
    if (!response.ok) throw new Error(`replay request failed (${response.status})`);
    source = await response.blob();
  }
  const stream = source.stream().pipeThrough(new DecompressionStream('gzip'));
  const buffer = await new Response(stream).arrayBuffer();
  if (new TextDecoder().decode(new Uint8Array(buffer, 0, 4)) !== 'HVR1') throw new Error('not an HVR1 replay');
  const size = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, size)));
  const types = { uint8: Uint8Array, int16: Int16Array, int32: Int32Array, float32: Float32Array };
  const arrays = {};
  for (const [name, spec] of Object.entries(header.arrays)) {
    const Type = types[spec.dtype];
    const count = spec.shape.reduce((a, b) => a * b, 1);
    const start = 8 + size + spec.offset;
    if (spec.codec === 'delta-shuffle') {
      const raw = new Uint8Array(buffer, start, count * 2);
      const out = new Int16Array(count);
      const stride = count / spec.shape[0];
      for (let i = 0; i < count; i++) {
        const delta = raw[i] | (raw[count + i] << 8);
        out[i] = i < stride ? delta : out[i - stride] + delta; // Int16Array wraps
      }
      arrays[name] = out;
    } else {
      arrays[name] = new Type(buffer.slice(start, start + count * Type.BYTES_PER_ELEMENT));
    }
  }
  return { meta: header.meta, arrays };
}

/* ---------- colour maps, built in OKLab so lightness follows magnitude ---------- */
function oklchToRgb(L, C, h) {
  const a = C * Math.cos(h * Math.PI / 180), b = C * Math.sin(h * Math.PI / 180);
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
          -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
          -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s].map((v) => {
    v = Math.min(1, Math.max(0, v));
    return Math.round(255 * (v <= 0.0031308 ? 12.92 * v : 1.055 * v ** (1 / 2.4) - 0.055));
  });
}
function buildLut(mode) {
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const x = i / 255;
    let rgb;
    if (mode === 0) { // diverging: steel blue (clockwise), near-black, sand (counter-clockwise)
      const s = 2 * x - 1, m = Math.abs(s);
      rgb = oklchToRgb(0.178 + 0.54 * m, 0.095 * Math.min(1, 1.4 * m), s < 0 ? 245 : 62);
    } else rgb = oklchToRgb(0.165 + 0.56 * x, 0.012 + 0.06 * Math.sin(Math.PI * x), 255 - 55 * x);
    lut.set([...rgb, 255], i * 4);
  }
  return lut;
}

/* ---------- GL helpers ---------- */
function program(vs, fs, varyings) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, '#version 300 es\nprecision highp float;\nprecision highp sampler2D;\n' + src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(p, shader);
  }
  if (varyings) gl.transformFeedbackVaryings(p, varyings, gl.INTERLEAVED_ATTRIBS);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const cache = {};
  p.u = (name) => (name in cache ? cache[name] : (cache[name] = gl.getUniformLocation(p, name)));
  return p;
}
function texture(width, height, internal, format, type, data, filter = gl.LINEAR, wrapS = gl.REPEAT) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, type, data);
  for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, filter], [gl.TEXTURE_MAG_FILTER, filter],
    [gl.TEXTURE_WRAP_S, wrapS], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  return t;
}
function bind(unit, tex) { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); }

const QUAD_VS = `
uniform vec2 uCenter, uScale; // clip = (world - uCenter) * uScale
uniform vec2 uRes;
out vec2 vWorld;
void main() {
  vec2 corner = vec2(gl_VertexID & 1, gl_VertexID >> 1);
  vWorld = corner * uRes - 0.5;
  gl_Position = vec4((vWorld - uCenter) * uScale, 0.0, 1.0);
}`;

const SAMPLING = `
// Cubic B-spline reconstruction from four bilinear taps: smooth values and gradients.
vec4 bicubic(sampler2D tex, vec2 uv, vec2 res) {
  vec2 p = uv * res - 0.5, f = fract(p), i = p - f;
  vec2 f2 = f * f, f3 = f2 * f;
  vec2 w0 = (1.0 - 3.0*f + 3.0*f2 - f3) / 6.0, w1 = (4.0 - 6.0*f2 + 3.0*f3) / 6.0;
  vec2 w3 = f3 / 6.0, w2 = 1.0 - w0 - w1 - w3;
  vec2 g0 = w0 + w1, g1 = w2 + w3;
  vec2 h0 = (i - 0.5 + w1 / g0) / res, h1 = (i + 1.5 + w3 / g1) / res;
  return g0.y * (g0.x * texture(tex, h0) + g1.x * texture(tex, vec2(h1.x, h0.y)))
       + g1.y * (g0.x * texture(tex, vec2(h0.x, h1.y)) + g1.x * texture(tex, h1));
}`;

const FIELD_FS = `
uniform sampler2D uVel0, uVel1, uLut;
uniform float uMix, uVortMax, uSpeedMax;
uniform int uMode;
uniform vec2 uRes, uTex;
in vec2 vWorld;
out vec4 color;
${SAMPLING}
// Soft compression: no flat clipped plateaus. x is the value over its reference level; the
// legend below uses the same function over [0, LEGEND_MAX] reference levels.
float compress(float x) { float y = pow(x / 0.55, 0.9); return y / sqrt(1.0 + y * y); }
vec2 vel(vec2 p) { // p in cells
  vec2 uv = (p + 0.5) / uRes;
  return mix(bicubic(uVel0, uv, uTex).xy, bicubic(uVel1, uv, uTex).xy, uMix);
}
void main() {
  float value;
  if (uMode == 0) {
    float h = 1.0;
    float w = (vel(vWorld + vec2(h, 0)).y - vel(vWorld - vec2(h, 0)).y
             - vel(vWorld + vec2(0, h)).x + vel(vWorld - vec2(0, h)).x) / (2.0 * h);
    value = 0.5 + 0.5 * sign(w) * compress(abs(w) / uVortMax);
  } else value = compress(length(vel(vWorld)) / uSpeedMax);
  vec3 c = texture(uLut, vec2(value * 255.0 / 256.0 + 0.5 / 256.0, 0.5)).rgb;
  // ordered-noise dither removes banding in the dark gradients
  c += (fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715)))) - 0.5) / 255.0;
  color = vec4(c, 1.0);
}`;

// Flat-coloured triangles with per-vertex RGBA, in cells.
const MESH_VS = `
uniform vec2 uCenter, uScale;
uniform float uShift;
in vec2 aPos;
in vec4 aColor;
out vec4 vColor;
void main() { vColor = aColor; gl_Position = vec4((aPos + vec2(uShift, 0.0) - uCenter) * uScale, 0.0, 1.0); }`;
const MESH_FS = `
in vec4 vColor;
out vec4 color;
void main() { color = vColor; }`;

/* ---------- geometry ---------- */
const catmull = (p0, p1, p2, p3, s) =>
  p1 + 0.5 * s * (p2 - p0 + s * (2 * p0 - 5 * p1 + 4 * p2 - p3 + s * (3 * (p1 - p2) + p3 - p0)));
// Catmull-Rom through a list of [x, y]; `closed` wraps the ends.
function spline(points, closed, per) {
  const n = points.length, out = [];
  const at = (i) => points[closed ? (i + n) % n : Math.max(0, Math.min(n - 1, i))];
  for (let i = 0; i < (closed ? n : n - 1); i++) {
    for (let k = 0; k < per; k++) {
      const s = k / per;
      out.push([0, 1].map((a) => catmull(at(i - 1)[a], at(i)[a], at(i + 1)[a], at(i + 2)[a], s)));
    }
  }
  if (!closed) out.push(points[n - 1]);
  return out;
}
// Unit normals of a polyline (left of travel direction).
function normals(line, closed) {
  const n = line.length;
  return line.map((_, i) => {
    const a = line[closed ? (i - 1 + n) % n : Math.max(0, i - 1)], b = line[closed ? (i + 1) % n : Math.min(n - 1, i + 1)];
    const dx = b[0] - a[0], dy = b[1] - a[1], d = Math.hypot(dx, dy) || 1;
    return [-dy / d, dx / d];
  });
}
class Mesh {
  constructor() { this.data = []; }
  tri(a, b, c, ca, cb = ca, cc = ca) { this.data.push(a[0], a[1], ...ca, b[0], b[1], ...cb, c[0], c[1], ...cc); }
  strip(left, right, leftColor, rightColor = leftColor, closed = false) {
    const n = left.length;
    for (let i = 0; i < (closed ? n : n - 1); i++) {
      const j = (i + 1) % n;
      this.tri(left[i], right[i], left[j], leftColor, rightColor, leftColor);
      this.tri(left[j], right[i], right[j], leftColor, rightColor, rightColor);
    }
  }
  fan(centre, ring, centreColor, rimColor) {
    ring.forEach((p, i) => this.tri(centre, p, ring[(i + 1) % ring.length], centreColor, rimColor, rimColor));
  }
  disc(c, r, color, n = 10) {
    this.fan(c, Array.from({ length: n }, (_, i) => [c[0] + r * Math.cos(2 * Math.PI * i / n), c[1] + r * Math.sin(2 * Math.PI * i / n)]), color, color);
  }
}
const rgba = (hex, a = 1) => [...hexToRgb(hex), a];
const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
const BODY = { fill: '#aeb3b4', core: '#e2e5e2', edge: '#06080a' };
const ACCENT = { fill: '#a9d24f', core: '#e0fa9f', edge: '#0a0c0f' };

/* ---------- application ---------- */
async function main() {
  if (!gl) return fail('This viewer needs WebGL2.');
  const { meta, arrays: A } = await loadReplay();
  const NX = meta.nx, NY = meta.ny, TX = NX / 2, TY = NY / 2;
  const FRAMES = meta.frames, LAST = FRAMES - 1, PER_BEAT = meta.beat_steps / meta.stride;
  const NPOINTS = meta.bodies.reduce((a, b) => a + b.count, 0);
  const uScale = meta.velocity_range / 32767;

  const velocityCache = new Map();
  function velocityTexture(k) {
    let tex = velocityCache.get(k);
    if (tex) velocityCache.delete(k);
    else {
      const data = new Float32Array(TX * TY * 2), base = k * TX * TY * 2;
      for (let i = 0; i < data.length; i++) data[i] = A.velocity[base + i] * uScale;
      tex = texture(TX, TY, gl.RG16F, gl.RG, gl.FLOAT, data);
      if (velocityCache.size >= 48) { // bounded GPU memory
        const oldest = velocityCache.keys().next().value;
        gl.deleteTexture(velocityCache.get(oldest)); velocityCache.delete(oldest);
      }
    }
    velocityCache.set(k, tex);
    return tex;
  }
  const lutData = [buildLut(0), buildLut(1)];
  const luts = lutData.map((d) => texture(256, 1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, d, gl.LINEAR, gl.CLAMP_TO_EDGE));
  const fieldProgram = program(QUAD_VS, FIELD_FS);
  const meshProgram = program(MESH_VS, MESH_FS);
  const emptyVao = gl.createVertexArray();
  const meshBuffer = gl.createBuffer(), meshVao = gl.createVertexArray();
  gl.bindVertexArray(meshVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, meshBuffer);
  for (const [name, size, offset] of [['aPos', 2, 0], ['aColor', 4, 8]]) {
    const loc = gl.getAttribLocation(meshProgram, name);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 24, offset);
  }

  /* tracers: pathlines through the recorded velocity, computed once (RK2, 2 substeps) */
  const SPACING = 13, tracerSeeds = [];
  let seed = 7;
  const random = () => (seed = (seed * 48271) % 2147483647) / 2147483647; // fixed, reproducible
  for (let y = SPACING / 2; y < NY - 3; y += SPACING) {
    for (let x = SPACING / 2; x < NX; x += SPACING) {
      const p = [x + (random() - 0.5) * SPACING * 0.8, Math.max(2, Math.min(NY - 3, y + (random() - 0.5) * SPACING * 0.8))];
      if (Math.hypot(p[0] - meta.post[0], p[1] - meta.post[1]) > meta.post[2] + 2) tracerSeeds.push(p);
    }
  }
  const NT = tracerSeeds.length, paths = new Float32Array(FRAMES * NT * 2);
  function sampleVelocity(k, x, y, c) {
    const gx = (x + 0.5) / 2 - 0.5, gy = Math.max(0, Math.min(TY - 1.001, (y + 0.5) / 2 - 0.5));
    const ix = Math.floor(gx), iy = Math.floor(gy), fx = gx - ix, fy = gy - iy;
    const x0 = ((ix % TX) + TX) % TX, x1 = (x0 + 1) % TX, base = k * TX * TY * 2;
    const v = (i, j) => A.velocity[base + (j * TX + i) * 2 + c];
    return uScale * ((v(x0, iy) * (1 - fx) + v(x1, iy) * fx) * (1 - fy) + (v(x0, iy + 1) * (1 - fx) + v(x1, iy + 1) * fx) * fy);
  }
  tracerSeeds.forEach((p, i) => paths.set(p, i * 2));
  for (let k = 0; k < LAST; k++) {
    for (let i = 0; i < NT; i++) {
      let x = paths[(k * NT + i) * 2], y = paths[(k * NT + i) * 2 + 1];
      for (let sub = 0; sub < 2; sub++) {
        const s = sub / 2, h = meta.stride / 2;
        const at = (px, py, w, c) => sampleVelocity(k, px, py, c) * (1 - w) + sampleVelocity(k + 1, px, py, c) * w;
        const mx = x + 0.5 * h * at(x, y, s, 0), my = y + 0.5 * h * at(x, y, s, 1);
        x += h * at(mx, my, s + 0.25, 0); y += h * at(mx, my, s + 0.25, 1);
      }
      paths[((k + 1) * NT + i) * 2] = x; paths[((k + 1) * NT + i) * 2 + 1] = y; // unwrapped in x
    }
  }

  /* swimmers */
  function pointsAt(t) {
    const f = Math.min(LAST - 1, Math.floor(t)), s = t - f, out = new Float64Array(NPOINTS * 2);
    const rows = [Math.max(0, f - 1), f, f + 1, Math.min(LAST, f + 2)].map((k) => k * NPOINTS * 2);
    for (let i = 0; i < NPOINTS * 2; i++) {
      out[i] = catmull(A.points[rows[0] + i], A.points[rows[1] + i], A.points[rows[2] + i], A.points[rows[3] + i], s) / meta.point_scale;
    }
    return out;
  }
  function shape(X, body) {
    const p = (i) => [X[2 * (body.start + i)], X[2 * (body.start + i) + 1]];
    const ring = Array.from({ length: body.ring }, (_, i) => p(1 + i));
    const chain = [p(1 + body.ring / 2)];
    for (let i = 1 + body.ring; i < body.count; i++) chain.push(p(i));
    return { hub: p(0), ring, chain };
  }
  function drawSwimmer(mesh, X, body, isSelected, px) {
    const { hub, ring, chain } = shape(X, body);
    const style = isSelected ? ACCENT : BODY;
    const line = spline(chain, false, 5), n = normals(line, false);
    const width = line.map((_, i) => meta.tail_width[0] + (meta.tail_width[1] - meta.tail_width[0]) * i / (line.length - 1));
    const offset = (pts, ns, d) => pts.map((q, i) => { const w = typeof d === 'number' ? d : d[i]; return [q[0] + ns[i][0] * w, q[1] + ns[i][1] * w]; });
    // head outline: ring points, grown outward by head_pad
    let outline = spline(ring, true, 5);
    let on = normals(outline, true);
    let area = 0;
    outline.forEach((q, i) => { const r = outline[(i + 1) % outline.length]; area += q[0] * r[1] - r[0] * q[1]; });
    if (area > 0) on = on.map((v) => [-v[0], -v[1]]); // make them point outward
    const rim = offset(outline, on, meta.head_pad);
    const edge = Math.min(0.42, 1.4 / px), soft = 1.0 / px;
    const clear = rgba(style.edge, 0);
    // dark keyline just outside the drawn shape, so bodies read on both field colours
    const halo = (a, b) => mesh.strip(a, b, rgba(style.edge, 0.72), clear);
    mesh.strip(offset(line, n, width), offset(line, n, width.map((w) => -w)), rgba(style.fill));
    const tip = line[line.length - 1];
    mesh.disc(tip, meta.tail_width[1], rgba(style.fill), 12);
    for (const side of [1, -1]) {
      const at = offset(line, n, width.map((w) => side * w));
      halo(at, offset(line, n, width.map((w) => side * (w + edge))));
    }
    mesh.fan(hub, rim, rgba(style.core), rgba(style.fill));
    mesh.strip(rim, offset(outline, on, meta.head_pad + soft), rgba(style.fill), rgba(style.fill, 0), true);
    mesh.strip(offset(outline, on, meta.head_pad + 0.5 * soft), offset(outline, on, meta.head_pad + 0.5 * soft + edge), rgba(style.edge, 0.72), clear, true);
  }
  function drawSolids(mesh, px) {
    const fill = rgba('#252a31'), line = rgba('#7b8490');
    const x0 = -0.5, x1 = NX - 0.5, rim = Math.max(0.45, 1.3 / px);
    for (const [y, out] of [[meta.walls[0], -1], [meta.walls[1], 1]]) {
      mesh.strip([[x0, y], [x1, y]], [[x0, y + out * 4], [x1, y + out * 4]], fill);
      mesh.strip([[x0, y], [x1, y]], [[x0, y + out * rim], [x1, y + out * rim]], line);
    }
    const [cx, cy, r] = meta.post;
    const circle = (radius) => Array.from({ length: 96 }, (_, i) => [cx + radius * Math.cos(Math.PI * i / 48), cy + radius * Math.sin(Math.PI * i / 48)]);
    mesh.fan([cx, cy], circle(r), fill, fill);
    mesh.strip(circle(r), circle(r - rim), line, line, true);
  }
  // Each tracer is sampled once per beat at the current beat phase, so the trail shows net
  // transport over the last TRAIL_BEATS beats without the back-and-forth within a beat.
  const TRAIL_BEATS = 8;
  function drawTracers(mesh, t, px) {
    const half = Math.max(0.14, 0.6 / px);
    for (let i = 0; i < NT; i++) {
      const at = (u) => {
        const f = Math.min(LAST - 1, Math.floor(u)), s = u - f;
        return [0, 1].map((a) => paths[(f * NT + i) * 2 + a] * (1 - s) + paths[((f + 1) * NT + i) * 2 + a] * s);
      };
      const line = [];
      for (let m = TRAIL_BEATS; m >= 0; m--) if (t - m * PER_BEAT >= 0) line.push(at(t - m * PER_BEAT));
      const head = line[line.length - 1];
      const wrap = Math.floor((head[0] + 0.5) / NX) * NX;
      for (const q of line) q[0] -= wrap;
      let length = 0;
      for (let k = 1; k < line.length; k++) length += Math.hypot(line[k][0] - line[k - 1][0], line[k][1] - line[k - 1][1]);
      // tracers that have barely moved are left undrawn rather than shown as specks
      const moved = Math.min(1, Math.max(0, (length - 0.8) / 2.5));
      if (line.length > 1 && moved > 0) {
        const smooth = spline(line, false, 3), n = normals(smooth, false), m = smooth.length - 1;
        for (let k = 0; k < m; k++) {
          const c0 = [0.86, 0.89, 0.92, 0.55 * moved * k / m], c1 = [0.86, 0.89, 0.92, 0.55 * moved * (k + 1) / m];
          const q = (j, side) => [smooth[j][0] + side * half * n[j][0], smooth[j][1] + side * half * n[j][1]];
          mesh.tri(q(k, 1), q(k, -1), q(k + 1, 1), c0, c0, c1);
          mesh.tri(q(k + 1, 1), q(k, -1), q(k + 1, -1), c1, c0, c1);
        }
      }
      if (moved > 0) mesh.disc(line[line.length - 1], half, [0.88, 0.91, 0.94, 0.55 * moved], 8);
    }
  }
  function flush(mesh, shifts) {
    gl.bindVertexArray(meshVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, meshBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(mesh.data), gl.STREAM_DRAW);
    for (const shift of shifts) {
      gl.uniform1f(meshProgram.u('uShift'), shift);
      gl.drawArrays(gl.TRIANGLES, 0, mesh.data.length / 6);
    }
  }

  /* view state */
  const home = { x: (NX - 1) / 2, y: (NY - 1) / 2, zoom: 1 };
  const view = { ...home };
  const startFrame = params.has('beat') ? Number(params.get('beat')) * PER_BEAT : 0;
  let t = Math.min(LAST, Math.max(0, startFrame));
  let playing = !params.has('paused'), mode = params.get('field') === 'speed' ? 1 : 0;
  let selected = null, follow = false, current = null;
  if (params.has('zoom')) view.zoom = Number(params.get('zoom'));
  if (params.has('cx')) view.x = Number(params.get('cx'));
  if (params.has('cy')) view.y = Number(params.get('cy'));
  if (params.has('select')) selected = Number(params.get('select'));
  if (params.get('tracers') === '0') $('show-tracers').checked = false;

  // CSS pixels per cell: the whole channel, walls included, fits at zoom 1.
  const pixelsPerCell = () => Math.max(0.2, Math.min((canvas.clientWidth - 32) / NX, (canvas.clientHeight - 8) / (NY + 6))) * view.zoom;
  const centroid = (X, body) => [X[2 * body.start], X[2 * body.start + 1]];
  function draw() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    current = pointsAt(t);
    if (follow && selected !== null) {
      const c = centroid(current, meta.bodies[selected]);
      view.x = c[0] - Math.floor((c[0] + 0.5) / NX) * NX; view.y = c[1];
    }
    gl.viewport(0, 0, w, h);
    gl.disable(gl.BLEND); gl.disable(gl.SCISSOR_TEST);
    gl.clearColor(0.047, 0.055, 0.067, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    const ppc = pixelsPerCell(), px = ppc * dpr;
    const scale = [2 * ppc / canvas.clientWidth, 2 * ppc / canvas.clientHeight];
    const setView = (p) => { gl.uniform2f(p.u('uCenter'), view.x, view.y); gl.uniform2f(p.u('uScale'), ...scale); };
    const sx = (x) => Math.round(w / 2 + (x - view.x) * px), sy = (y) => Math.round(h / 2 + (y - view.y) * px);
    gl.enable(gl.SCISSOR_TEST); // clip to the domain plus the wall slabs
    const clip = (y0, y1) => gl.scissor(sx(-0.5), sy(y0), Math.max(0, sx(NX - 0.5) - sx(-0.5)), Math.max(0, sy(y1) - sy(y0)));
    clip(-0.5, NY - 0.5);

    const f = Math.min(LAST - 1, Math.floor(t));
    gl.useProgram(fieldProgram); setView(fieldProgram);
    [velocityTexture(f), velocityTexture(f + 1), luts[mode]].forEach((tex, unit) => bind(unit, tex));
    ['uVel0', 'uVel1', 'uLut'].forEach((name, unit) => gl.uniform1i(fieldProgram.u(name), unit));
    gl.uniform1f(fieldProgram.u('uMix'), t - f);
    gl.uniform1f(fieldProgram.u('uVortMax'), meta.vorticity_max);
    gl.uniform1f(fieldProgram.u('uSpeedMax'), meta.speed_max);
    gl.uniform1i(fieldProgram.u('uMode'), mode);
    gl.uniform2f(fieldProgram.u('uRes'), NX, NY); gl.uniform2f(fieldProgram.u('uTex'), TX, TY);
    gl.bindVertexArray(emptyVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(meshProgram); setView(meshProgram);
    if ($('show-tracers').checked) { const m = new Mesh(); drawTracers(m, t, px); flush(m, [-NX, 0, NX]); }
    clip(-4.5, NY + 3);
    const solids = new Mesh(); drawSolids(solids, px); flush(solids, [0]);
    const swimmers = new Mesh();
    meta.bodies.forEach((body, i) => {
      const c = centroid(current, body), wrap = Math.floor((c[0] + 0.5) / NX) * NX;
      const X = current.slice();
      for (let k = body.start; k < body.start + body.count; k++) X[2 * k] -= wrap;
      drawSwimmer(swimmers, X, body, selected === i, px);
    });
    flush(swimmers, [-NX, 0, NX]);
    gl.disable(gl.SCISSOR_TEST);
    updateScaleBar(ppc);
  }

  /* panel */
  const LEGEND_MAX = 2; // legend span in reference levels
  function updateLegend() {
    const bar = $('colorbar').getContext('2d');
    for (let x = 0; x < 400; x++) {
      let v = x / 399;
      const compress = (u) => { const y = (u * LEGEND_MAX / 0.55) ** 0.9; return y / Math.sqrt(1 + y * y); };
      if (mode === 0) { const s = 2 * v - 1; v = 0.5 + 0.5 * Math.sign(s) * compress(Math.abs(s)); } else v = compress(v);
      const i = Math.round(v * 255) * 4;
      bar.fillStyle = `rgb(${lutData[mode][i]},${lutData[mode][i + 1]},${lutData[mode][i + 2]})`;
      bar.fillRect(x, 0, 1, 8);
    }
    const e = (v) => v.toExponential(1).replace('-', '−');
    const top = LEGEND_MAX * (mode === 0 ? meta.vorticity_max : meta.speed_max);
    const labels = mode === 0 ? [`−${e(top)}`, '0', `+${e(top)}`] : ['0', e(top / 2), e(top)];
    labels.forEach((l, i) => { $('ticks').children[i].textContent = l; });
    $('unit').textContent = mode === 0
      ? 'Vorticity, per lattice step. Sand: counter-clockwise, blue: clockwise. Compressed colour scale.'
      : 'Flow speed, cells per lattice step. Compressed colour scale.';
  }
  $('subtitle').textContent = `Immersed boundary, lattice Boltzmann ${NX} × ${NY} · ${meta.bodies.length} prescribed-gait swimmers · 1 beat = ${meta.beat_steps} steps`;
  const S = meta.stats;
  $('about-stats').textContent = `Contact and walls are soft penalties. In this run, over ${S.frames} records the drawn swimmer shapes `
    + `overlapped in ${S.drawn_body_overlap_frames} (smallest gap ${S.drawn_body_min_gap} cells), outlines crossed in ${S.outline_crossing_frames}, `
    + `and a drawn shape entered a wall or the post in ${S.drawn_surface_penetration_frames} (smallest gap ${S.drawn_surface_min_gap} cells). `
    + `The penalties barely engaged: over every step the closest boundary points of two swimmers were ${S.every_step_min_point_distance} cells apart `
    + `(contact range ${meta.contact[0]}), and the closest approach to a wall or post was ${S.every_step_min_point_wall_gap} (wall range ${meta.wall[0]}). `
    + 'This run therefore does not test them under strong forcing.';

  // Mean speed of the head hub over a centred 4-beat window, in cells per beat.
  function swimSpeed(i) {
    const k = meta.bodies[i].start, half = 2 * PER_BEAT;
    const a = Math.max(0, Math.min(LAST - 2 * half, Math.round(t) - half)), b = Math.min(LAST, a + 2 * half);
    const d = [0, 1].map((c) => (A.points[(b * NPOINTS + k) * 2 + c] - A.points[(a * NPOINTS + k) * 2 + c]) / meta.point_scale);
    return Math.hypot(...d) / ((b - a) / PER_BEAT);
  }
  function updatePanel() {
    const beat = t / PER_BEAT;
    $('clock').textContent = `beat ${beat.toFixed(1)} / ${(LAST / PER_BEAT).toFixed(0)}  ·  step ${(meta.first_step + t * meta.stride).toFixed(0)}`;
    $('scrub').value = t;
    $('card').hidden = selected === null;
    if (selected === null) return;
    const body = meta.bodies[selected];
    $('card-title').textContent = `Swimmer ${selected + 1}`;
    const c = centroid(pointsAt(t), body);
    const speed = swimSpeed(selected);
    const items = [['Head position', `${(((c[0] % NX) + NX) % NX).toFixed(1)}, ${c[1].toFixed(1)}`],
      ['Speed, 4-beat mean', `${speed.toFixed(2)} cells/beat`], ['Net displacement, whole run', `${body.travel.toFixed(1)} cells`],
      ['Boundary points', body.count]];
    $('details').replaceChildren(...items.flatMap(([k, v]) => {
      const dt = document.createElement('dt'), dd = document.createElement('dd');
      dt.textContent = k; dd.textContent = v; return [dt, dd];
    }));
    $('follow').textContent = follow ? 'Stop following' : 'Follow';
  }
  function updateScaleBar(ppc) {
    const cells = [100, 50, 20, 10, 5].find((c) => c * ppc <= 150) || 2;
    $('scalebar').firstElementChild.style.width = `${cells * ppc}px`;
    $('scalebar').lastElementChild.textContent = `${cells} lattice cells`;
    const left = canvas.clientWidth / 2 + (-0.5 - view.x) * ppc;
    $('scalebar').style.left = `${Math.max(12, left + 12)}px`;
  }
  function refresh() { draw(); updatePanel(); }

  /* controls */
  $('scrub').max = LAST;
  $('scrub').oninput = (e) => { t = Number(e.target.value); };
  const setPlaying = (v) => { playing = v; $('play').textContent = v ? 'Pause' : 'Play'; };
  $('play').onclick = () => setPlaying(!playing);
  setPlaying(playing);
  const resetView = () => { Object.assign(view, home); follow = false; };
  $('reset').onclick = resetView;
  $('follow').onclick = () => { follow = !follow; if (follow) view.zoom = Math.max(view.zoom, 2.5); };
  const setMode = (m) => {
    mode = m;
    for (const b of $('field-mode').children) b.classList.toggle('on', Number(b.dataset.mode) === m);
    updateLegend();
  };
  for (const b of $('field-mode').children) b.onclick = () => setMode(Number(b.dataset.mode));
  setMode(mode);
  window.onkeydown = (e) => {
    if (e.target.matches('input,select,button')) return;
    if (e.code === 'Space') { e.preventDefault(); setPlaying(!playing); }
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      t = Math.max(0, Math.min(LAST, Math.round(t) + (e.key === 'ArrowRight' ? 1 : -1))); setPlaying(false);
    }
    if (e.key === '0') resetView();
    if (e.key === 'Escape') { selected = null; follow = false; }
  };
  const toWorld = (x, y) => [view.x + (x - canvas.clientWidth / 2) / pixelsPerCell(),
    view.y - (y - canvas.clientHeight / 2) / pixelsPerCell()];
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const [wx, wy] = toWorld(e.offsetX, e.offsetY);
    const zoom = Math.max(0.8, Math.min(12, view.zoom * Math.exp(-e.deltaY * 0.0015)));
    const k = view.zoom / zoom;
    view.x = wx + (view.x - wx) * k; view.y = wy + (view.y - wy) * k; view.zoom = zoom;
  }, { passive: false });
  let drag = null;
  canvas.onpointerdown = (e) => { drag = { x: e.clientX, y: e.clientY, moved: 0 }; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.moved += Math.abs(dx) + Math.abs(dy); drag.x = e.clientX; drag.y = e.clientY;
    if (drag.moved > 4) { canvas.classList.add('dragging'); follow = false; view.x -= dx / pixelsPerCell(); view.y += dy / pixelsPerCell(); }
  };
  function pick(wx, wy) {
    let best = null, distance = Math.max(7, 9 / pixelsPerCell());
    meta.bodies.forEach((body, i) => {
      for (let k = body.start; k < body.start + body.count; k++) {
        const dx = wx - current[2 * k];
        const d = Math.hypot(dx - NX * Math.round(dx / NX), wy - current[2 * k + 1]);
        if (d < distance) { best = i; distance = d; }
      }
    });
    selected = best;
    if (best === null) follow = false;
  }
  canvas.onpointerup = (e) => {
    if (drag && drag.moved <= 4) { pick(...toWorld(e.offsetX, e.offsetY)); refresh(); }
    drag = null; canvas.classList.remove('dragging');
  };

  /* loop */
  let last = null, panelClock = 0, frames = 0, fpsStart = performance.now();
  function advance(dt) {
    if (playing) t += dt * Number($('rate').value) * PER_BEAT;
    if (t > LAST) t %= LAST; // loop, keeping the overshoot
    draw();
  }
  function loop(now) {
    // clamp: after a stall, slow down rather than jump
    advance(last === null ? 0 : Math.min(0.1, (now - last) / 1000));
    last = now;
    frames++;
    if (now - panelClock > 150) { updatePanel(); panelClock = now; }
    if (now - fpsStart > 1000) {
      const fps = frames * 1000 / (now - fpsStart);
      $('fps').textContent = `${fps.toFixed(0)} fps`; window.viewer.fps = fps;
      frames = 0; fpsStart = now;
    }
    requestAnimationFrame(loop);
  }
  $('loading').hidden = true;
  /* automation hooks used by capture scripts */
  window.viewer = {
    meta, fps: 0, advance: (dt) => { advance(dt); updatePanel(); },
    setView: (v) => { Object.assign(view, v); refresh(); }, setMode: (m) => { setMode(m); refresh(); },
    select: (i) => { selected = i; refresh(); }, selected: () => selected,
    seekBeat(b) { t = Math.max(0, Math.min(LAST, b * PER_BEAT)); refresh(); },
    beat: () => t / PER_BEAT,
    screen(i) { const c = centroid(current, meta.bodies[i]), r = canvas.getBoundingClientRect();
      return [r.x + r.width / 2 + (c[0] - view.x) * pixelsPerCell(), r.y + r.height / 2 - (c[1] - view.y) * pixelsPerCell()]; },
    renderer: () => { const ext = gl.getExtension('WEBGL_debug_renderer_info'); return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); },
    glError: () => gl.getError(),
  };
  refresh();
  if (!params.has('manual')) requestAnimationFrame(loop);
  window.viewer.ready = true;
}
main().catch((error) => { console.error(error); fail(`Viewer error: ${error.message}`); });
