/* hybrid-alife flow viewer. WebGL2, no dependencies.
 * World frame: unit square, x = simulator axis 0 (channel flow direction), y = axis 1, y up.
 * Time `t` is in simulation steps; everything between recorded steps is interpolated. */
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

/* ---------- perceptual colour maps (built in OKLab, lightness linear in value) ---------- */
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
    if (mode === 0) { // diverging: teal-blue (clockwise) - near-black - copper (counter-clockwise)
      const s = 2 * x - 1, m = Math.abs(s);
      rgb = oklchToRgb(0.17 + 0.47 * m, 0.118 * Math.min(1, 1.35 * m), s < 0 ? 238 : 52);
    } else { // sequential: indigo to pale rose, dark to light (kept clear of the green signal)
      rgb = oklchToRgb(0.15 + 0.55 * x, 0.03 + 0.095 * Math.sin(Math.PI * Math.min(1, x * 0.9)), 272 + 68 * x);
    }
    lut.set([...rgb, 255], i * 4);
  }
  return lut;
}
// dataviz skill dark categorical steps (validated for adjacent pairs); gray folds the rest.
const LINEAGE_COLORS = ['#00a888', '#d28e05', '#e24d8b', '#0aac23', '#9887ef', '#eb6263'];
const OTHER_COLOR = '#aeb7c2';
const hexToRgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);

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
out vec2 vWorld;
void main() {
  vec2 corner = vec2(gl_VertexID & 1, gl_VertexID >> 1);
  vWorld = corner;
  gl_Position = vec4((corner - uCenter) * uScale, 0.0, 1.0);
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
uniform sampler2D uVel0, uVel1, uSig0, uSig1, uSolid, uLut;
uniform float uMix, uVortMax, uSpeedMax, uSignal;
uniform int uMode;
uniform vec2 uRes;
in vec2 vWorld;
out vec4 color;
${SAMPLING}
vec2 vel(vec2 p) { return mix(bicubic(uVel0, p, uRes).xy, bicubic(uVel1, p, uRes).xy, uMix); }
float sig(vec2 p) { // stored as sqrt(signal / max)
  float a = texture(uSig0, p).r, b = texture(uSig1, p).r;
  return mix(a * a, b * b, uMix);
}
void main() {
  vec2 e = 0.75 / uRes;
  float value;
  if (uMode == 0) {
    float dvdx = (vel(vWorld + vec2(e.x, 0)).y - vel(vWorld - vec2(e.x, 0)).y) / 1.5;
    float dudy = (vel(vWorld + vec2(0, e.y)).x - vel(vWorld - vec2(0, e.y)).x) / 1.5;
    float w = (dvdx - dudy) / uVortMax;
    // mild gamma lifts weak structure; the colour bar is drawn with the same mapping
    value = 0.5 + 0.5 * sign(w) * pow(min(abs(w), 1.0), 0.8);
  } else value = length(vel(vWorld)) / uSpeedMax;
  vec3 c = texture(uLut, vec2(clamp(value, 0.0, 1.0) * 255.0 / 256.0 + 0.5 / 256.0, 0.5)).rgb;
  if (uSignal > 0.0) { // display-only 9-tap smoothing, radius ~2 cells
    vec2 r = 2.2 / uRes;
    float s = 0.0;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++)
      s += sig(vWorld + vec2(i, j) * r) * float((2 - abs(i)) * (2 - abs(j))) / 16.0;
    float level = sqrt(clamp(s, 0.0, 1.0)); // square-root display scale (see legend)
    c = mix(c, vec3(0.62, 0.93, 0.70), 0.54 * smoothstep(0.14, 1.0, level));
  }
  float solid = bicubic(uSolid, vWorld, uRes).r;
  float aa = max(fwidth(solid), 0.02);
  float inside = smoothstep(0.5 - aa, 0.5 + aa, solid);
  c = mix(c, vec3(0.115, 0.130, 0.152), inside);
  c = mix(c, vec3(0.50, 0.55, 0.62), (1.0 - smoothstep(0.0, 2.0 * aa, abs(solid - 0.5))) * 0.8);
  color = vec4(c, 1.0);
}`;

const ADVECT_VS = `
uniform sampler2D uVel0, uVel1, uSolid;
uniform float uMix, uDt, uSeed, uPeriodicY; // uDt in LBM steps
uniform vec2 uRes;
in vec3 aState; // x, y, remaining life (simulation steps)
uniform float uSteps;
out vec3 vState;
float rand(vec2 s) { return fract(sin(dot(s, vec2(12.9898, 78.233))) * 43758.5453); }
vec2 vel(vec2 p) { return mix(texture(uVel0, p).xy, texture(uVel1, p).xy, uMix); }
void main() {
  vec2 p = aState.xy;
  vec2 mid = p + 0.5 * uDt * vel(p) / uRes;            // RK2 midpoint
  vec2 q = p + uDt * vel(mid) / uRes;
  float life = aState.z - uSteps;
  bool out_y = uPeriodicY < 0.5 && (q.y < 0.004 || q.y > 0.996);
  q = fract(q);
  if (life <= 0.0 || out_y || texture(uSolid, q).r > 0.002) {
    float id = float(gl_VertexID) * 0.0001 + uSeed;
    q = vec2(rand(vec2(id, 1.7)), rand(vec2(id, 9.2)));
    life = -1.0 - (6.0 + 24.0 * rand(vec2(id, 4.4)));   // negative marks a fresh respawn
  }
  vState = vec3(q, life < -1.0 ? -life : life);
  gl_Position = vec4(0.0);
}`;

const SEGMENT_VS = `
in vec3 aFrom, aTo;
out float vAlpha;
void main() {
  vec2 p = gl_VertexID == 0 ? aFrom.xy : aTo.xy;
  bool jumped = distance(aFrom.xy, aTo.xy) > 0.08 || aTo.z > aFrom.z + 0.001;
  vAlpha = jumped ? 0.0 : min(1.0, aTo.z / 4.0);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;
const SEGMENT_FS = `
uniform float uGain;
in float vAlpha;
out vec4 color;
void main() { color = vec4(vec3(vAlpha * uGain), 1.0); }`;
const FADE_FS = `
uniform float uKeep, uDrop;
out vec4 color;
void main() { color = vec4(vec3(uDrop), uKeep); }`;
const TRAIL_FS = `
uniform sampler2D uTrail;
in vec2 vWorld;
out vec4 color;
void main() {
  float a = clamp(texture(uTrail, vWorld).r, 0.0, 1.0);
  color = vec4(0.87, 0.91, 0.95, a * 0.46);
}`;

// Instance: aPose (x, y, heading, flagellum phase), aShape (length, width, alpha, selected), aColor
const BODY_VS = `
uniform vec2 uCenter, uScale;
uniform float uShadow;
in vec4 aPose, aShape;
in vec3 aColor;
out vec2 vLocal;
out vec4 vShape;
out vec3 vColor;
out float vHeading;
void main() {
  vec2 corner = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 2.0 - 1.0;
  vec2 local = corner * vec2(1.55, 2.0);
  vec2 p = local * aShape.xy * 0.5;
  float c = cos(aPose.z), s = sin(aPose.z);
  vec2 world = aPose.xy + vec2(c * p.x - s * p.y, s * p.x + c * p.y)
             + uShadow * vec2(0.35, -0.55) * aShape.y;
  vLocal = local; vShape = aShape; vColor = aColor; vHeading = aPose.z;
  gl_Position = vec4((world - uCenter) * uScale, 0.0, 1.0);
}`;
const BODY_FS = `
uniform float uShadow;
in vec2 vLocal;
in vec4 vShape;
in vec3 vColor;
in float vHeading;
out vec4 color;
void main() {
  // egg profile: broad rounded front (+x), narrower rear where the flagellum attaches
  float taper = 1.0 - 0.17 * clamp(-vLocal.x + 0.2, 0.0, 1.2);
  vec2 q = vec2(vLocal.x, vLocal.y / taper);
  float r = length(q);
  float aa = fwidth(r) * 1.1;
  if (uShadow > 0.5) {
    color = vec4(0.0, 0.0, 0.0, 0.42 * (1.0 - smoothstep(0.65, 1.45, r)) * vShape.z);
    return;
  }
  float body = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, r);
  vec3 n = vec3(q * 0.94, sqrt(max(0.0, 1.0 - 0.88 * min(r * r, 1.0))));
  float c = cos(vHeading), s = sin(vHeading);
  n = normalize(vec3(c * n.x - s * n.y, s * n.x + c * n.y, n.z));
  vec3 light = normalize(vec3(-0.45, 0.65, 0.62));
  float diffuse = max(dot(n, light), 0.0);
  float spec = pow(max(dot(reflect(-light, n), vec3(0, 0, 1)), 0.0), 22.0);
  vec3 shade = vColor * (0.52 + 0.58 * diffuse) + spec * 0.11;
  shade = mix(shade, vColor * 0.34, smoothstep(0.70, 1.0, r)); // membrane edge
  float outline = (1.0 - smoothstep(1.13 - aa, 1.13 + aa, r)) * (1.0 - body);
  vec4 result = vec4(mix(vec3(0.02, 0.03, 0.04), shade, body), max(body, outline * 0.8) * vShape.z);
  if (vShape.w > 0.5) {
    float ring = 1.0 - smoothstep(0.0, aa * 1.6, abs(length(vLocal * vec2(1.0, 0.92)) - 1.42));
    result = mix(result, vec4(1.0), ring);
  }
  color = result;
}`;
const TAIL_VS = `
uniform vec2 uCenter, uScale;
in vec4 aPose, aShape;
in vec3 aColor;
out vec4 vColor;
out float vSide;
void main() {
  float s = float(gl_VertexID / 2) / 24.0;
  vSide = float(gl_VertexID & 1) * 2.0 - 1.0;
  float len = aShape.x, k = 9.4;
  float amp = len * 0.13 * smoothstep(0.0, 0.45, s);
  vec2 p = vec2(-len * 0.46 - s * len * 0.95, amp * sin(k * s - aPose.w));
  float slope = len * 0.13 * (smoothstep(0.0, 0.45, s) * k * cos(k * s - aPose.w)) / (len * 0.95);
  vec2 normal = normalize(vec2(slope, 1.0));
  p += normal * vSide * aShape.y * mix(0.11, 0.035, s);
  float c = cos(aPose.z), sn = sin(aPose.z);
  vec2 world = aPose.xy + vec2(c * p.x - sn * p.y, sn * p.x + c * p.y);
  vColor = vec4(mix(aColor, vec3(0.9), 0.35), aShape.z * (0.85 - 0.5 * s));
  gl_Position = vec4((world - uCenter) * uScale, 0.0, 1.0);
}`;
const TAIL_FS = `
in vec4 vColor;
in float vSide;
out vec4 color;
void main() { color = vec4(vColor.rgb, vColor.a * (1.0 - smoothstep(0.55, 1.0, abs(vSide)))); }`;

/* ---------- application ---------- */
async function main() {
  if (!gl) return fail('This viewer needs WebGL2.');
  const { meta, arrays: A } = await loadReplay();
  const [NX, NY] = meta.grid;
  const N = meta.population_slots, STEPS = meta.steps, EVERY = meta.field_every;
  const FRAMES = STEPS + 1, KEYS = STEPS / EVERY + 1;
  const periodicY = meta.scene === 'periodic';

  /* textures */
  const velocityCache = new Map(), signalCache = new Map();
  function velocityTexture(k) {
    if (!velocityCache.has(k)) {
      const data = new Float32Array(NX * NY * 2), base = k * NX * NY * 2;
      for (let i = 0; i < data.length; i++) data[i] = A.velocity[base + i] * meta.u_scale;
      velocityCache.set(k, texture(NX, NY, gl.RG16F, gl.RG, gl.FLOAT, data));
    }
    return velocityCache.get(k);
  }
  function signalTexture(k) {
    if (!signalCache.has(k)) {
      signalCache.set(k, texture(NX, NY, gl.R8, gl.RED, gl.UNSIGNED_BYTE, A.signal.subarray(k * NX * NY, (k + 1) * NX * NY)));
    }
    return signalCache.get(k);
  }
  const solidTexture = texture(NX, NY, gl.R8, gl.RED, gl.UNSIGNED_BYTE, A.solid.map((v) => v * 255));
  const lutData = [buildLut(0), buildLut(1)];
  const luts = lutData.map((d) => texture(256, 1, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, d, gl.LINEAR, gl.CLAMP_TO_EDGE));

  /* programs */
  const fieldProgram = program(QUAD_VS, FIELD_FS);
  const advect = program(ADVECT_VS, 'out vec4 c; void main(){c=vec4(0);}', ['vState']);
  const segment = program(SEGMENT_VS, SEGMENT_FS);
  const fade = program(QUAD_VS, FADE_FS);
  const trailProgram = program(QUAD_VS, TRAIL_FS);
  const bodyProgram = program(BODY_VS, BODY_FS);
  const tailProgram = program(TAIL_VS, TAIL_FS);

  /* tracers */
  const COUNT = Number(params.get('particles')) || 9000;
  const seeds = new Float32Array(COUNT * 3);
  for (let i = 0; i < COUNT; i++) seeds.set([Math.random(), Math.random(), Math.random() * 28], i * 3);
  const particleBuffers = [0, 1].map(() => {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, seeds, gl.DYNAMIC_COPY);
    return b;
  });
  function attribute(prog, name, buffer, size, stride, offset, divisor) {
    const loc = gl.getAttribLocation(prog, name);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
    gl.vertexAttribDivisor(loc, divisor);
  }
  const advectVaos = particleBuffers.map((b) => {
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    attribute(advect, 'aState', b, 3, 12, 0, 0);
    return vao;
  });
  const segmentVaos = [0, 1].map((from) => {
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    attribute(segment, 'aFrom', particleBuffers[from], 3, 12, 0, 1);
    attribute(segment, 'aTo', particleBuffers[1 - from], 3, 12, 0, 1);
    return vao;
  });
  const feedback = gl.createTransformFeedback();
  const emptyVao = gl.createVertexArray();
  const TRAIL = Math.min(Number(params.get('trail')) || 2048, gl.getParameter(gl.MAX_TEXTURE_SIZE));
  const halfFloat = !!gl.getExtension('EXT_color_buffer_float');
  const trailTexture = texture(TRAIL, TRAIL, halfFloat ? gl.R16F : gl.R8, gl.RED,
    halfFloat ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, null, gl.LINEAR, gl.CLAMP_TO_EDGE);
  const trailBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, trailBuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, trailTexture, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  let current = 0;

  function fieldAt(t) {
    const k = Math.min(KEYS - 1, t / EVERY), k0 = Math.floor(k), k1 = Math.min(KEYS - 1, k0 + 1);
    return { k0, k1, mix: k - k0 };
  }
  function stepTracers(t, dSteps) {
    const f = fieldAt(t);
    gl.useProgram(advect);
    bind(0, velocityTexture(f.k0)); bind(1, velocityTexture(f.k1)); bind(2, solidTexture);
    gl.uniform1i(advect.u('uVel0'), 0); gl.uniform1i(advect.u('uVel1'), 1); gl.uniform1i(advect.u('uSolid'), 2);
    gl.uniform1f(advect.u('uMix'), f.mix);
    gl.uniform1f(advect.u('uDt'), dSteps * meta.lbm_substeps);
    gl.uniform1f(advect.u('uSteps'), dSteps);
    gl.uniform1f(advect.u('uSeed'), Math.random());
    gl.uniform1f(advect.u('uPeriodicY'), periodicY ? 1 : 0);
    gl.uniform2f(advect.u('uRes'), NX, NY);
    gl.bindVertexArray(advectVaos[current]);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, feedback);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, particleBuffers[1 - current]);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, COUNT);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, trailBuffer);
    gl.viewport(0, 0, TRAIL, TRAIL);
    gl.enable(gl.BLEND);
    // fade: dst * keep - drop (the small subtraction clears 8-bit residue)
    gl.useProgram(fade);
    gl.uniform2f(fade.u('uCenter'), 0.5, 0.5); gl.uniform2f(fade.u('uScale'), 2, 2);
    gl.uniform1f(fade.u('uKeep'), Math.exp(-dSteps / 4.0));
    gl.uniform1f(fade.u('uDrop'), halfFloat ? 0.0005 : 1 / 255);
    gl.blendEquation(gl.FUNC_REVERSE_SUBTRACT);
    gl.blendFunc(gl.ONE, gl.SRC_ALPHA);
    gl.bindVertexArray(emptyVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.blendEquation(gl.MAX);
    gl.useProgram(segment);
    gl.uniform1f(segment.u('uGain'), 0.85);
    gl.bindVertexArray(segmentVaos[current]);
    gl.drawArraysInstanced(gl.LINES, 0, 2, COUNT);
    gl.blendEquation(gl.FUNC_ADD);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    current = 1 - current;
  }
  function resetTrails(t) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, trailBuffer);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    for (let i = 0; i < 24; i++) stepTracers(t, 0.2); // pre-roll so a paused frame has streaks
  }

  /* organisms: lineage ranking, headings, flagellum phase */
  const presence = new Map();
  for (let i = 0; i < FRAMES * N; i++) if (A.alive[i]) presence.set(A.founder[i], (presence.get(A.founder[i]) || 0) + 1);
  const top = [...presence.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, LINEAGE_COLORS.length).map((e) => e[0]);
  const colorOf = (founder) => (top.includes(founder) ? LINEAGE_COLORS[top.indexOf(founder)] : OTHER_COLOR);
  const rgbCache = new Map([...LINEAGE_COLORS, OTHER_COLOR].map((c) => [c, hexToRgb(c)]));

  const heading = new Float32Array(FRAMES * N), phase = new Float32Array(FRAMES * N);
  let meanSwim = 0, living = 0;
  for (let i = 0; i < FRAMES * N; i++) if (A.alive[i]) { meanSwim += Math.hypot(A.swim[2 * i], A.swim[2 * i + 1]); living++; }
  meanSwim /= Math.max(living, 1);
  for (let n = 0; n < N; n++) {
    for (let f = 0; f < FRAMES; f++) {
      // swim[f] drove the move from f-1 to f; frame 0 has none, so borrow frame 1.
      const i = Math.max(f, 1) * N + n, sx = A.swim[2 * i], sy = A.swim[2 * i + 1];
      const speed = Math.hypot(sx, sy), o = f * N + n;
      heading[o] = speed > 1e-5 ? Math.atan2(sy, sx) : (f ? heading[o - N] : 0);
      phase[o] = f ? phase[o - N] + 4.2 * Math.min(2.5, 0.25 + speed / meanSwim) : n * 2.4;
    }
  }

  const sameBody = (a, b) => A.alive[a] && A.alive[b] && A.lineage[a] === A.lineage[b];
  const wrapDelta = (d) => d - Math.round(d);
  function catmull(p0, p1, p2, p3, s) {
    return p1 + 0.5 * s * (p2 - p0 + s * (2 * p0 - 5 * p1 + 4 * p2 - p3 + s * (3 * (p1 - p2) + p3 - p0)));
  }
  function angleLerp(a, b, s) { return a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * s; }
  function pose(n, t) {
    const f = Math.min(STEPS - 1, Math.floor(t)), s = Math.min(1, t - f);
    const i = f * N + n, j = i + N;
    const out = [];
    const make = (index, x, y, angle, ph, alpha, scale) => out.push({ index, x, y, angle, phase: ph, alpha, scale });
    if (sameBody(i, j)) {
      const prev = f > 0 && sameBody(i - N, i) ? i - N : i;
      const next = f + 2 < FRAMES && sameBody(j, j + N) ? j + N : j;
      const xy = [0, 1].map((a) => {
        const p1 = A.position[2 * i + a];
        const p2 = p1 + wrapDelta(A.position[2 * j + a] - p1);
        const p0 = p1 + wrapDelta(A.position[2 * prev + a] - p1);
        const p3 = p2 + wrapDelta(A.position[2 * next + a] - p2);
        const v = catmull(p0, p1, p2, p3, s);
        return v - Math.floor(v);
      });
      // each step's swim direction is centred on that step's midpoint
      const a = Math.max(0, Math.min(STEPS - 1, Math.floor(t - 0.5)));
      const angle = angleLerp(heading[(a + 1) * N + n], heading[Math.min(STEPS, a + 2) * N + n], Math.max(0, Math.min(1, t - 0.5 - a)));
      make(j, xy[0], xy[1], angle, phase[i] + (phase[j] - phase[i]) * s, 1, 1);
    } else {
      if (A.alive[i]) make(i, A.position[2 * i], A.position[2 * i + 1], heading[i], phase[i], 1 - s, 1 - 0.35 * s); // death
      if (A.alive[j]) make(j, A.position[2 * j], A.position[2 * j + 1], heading[j], phase[j], s, 0.4 + 0.6 * s); // birth
    }
    return out;
  }
  // Drawn size in grid cells: illustrative scale; proportions follow the genes.
  function bodySize(index) {
    const r = A.radius[index], a = Math.max(0.5, A.aspect[index]), stretch = Math.sqrt(2.2 * a);
    return [6.7 * r * stretch / NX, 6.7 * r / stretch / NX];
  }

  /* view state */
  const view = { x: 0.5, y: 0.5, zoom: 1 };
  let t = Math.min(STEPS, Math.max(0, Number(params.get('t') ?? 0)));
  let playing = !params.has('paused'), mode = params.get('field') === 'speed' ? 1 : 0;
  let provenanceClosed = false;
  let selected = null, follow = false, isolate = null, drawn = [], needsTrailReset = true;
  if (params.has('zoom')) view.zoom = Number(params.get('zoom'));
  if (params.has('cx')) view.x = Number(params.get('cx'));
  if (params.has('cy')) view.y = Number(params.get('cy'));

  const instanceBuffer = gl.createBuffer();
  const instances = new Float32Array(2 * N * 11);
  const bodyVaos = [bodyProgram, tailProgram].map((p) => {
    const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
    attribute(p, 'aPose', instanceBuffer, 4, 44, 0, 1);
    attribute(p, 'aShape', instanceBuffer, 4, 44, 16, 1);
    attribute(p, 'aColor', instanceBuffer, 3, 44, 32, 1);
    return vao;
  });

  const pixelsPerUnit = () => Math.max(50, Math.min(canvas.clientWidth, canvas.clientHeight) - 24) * view.zoom;
  function draw() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.BLEND); gl.disable(gl.SCISSOR_TEST);
    gl.clearColor(0.043, 0.051, 0.063, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    const ppu = pixelsPerUnit();
    const scale = [2 * ppu / canvas.clientWidth, 2 * ppu / canvas.clientHeight];
    const setView = (p) => { gl.uniform2f(p.u('uCenter'), view.x, view.y); gl.uniform2f(p.u('uScale'), ...scale); };

    const f = fieldAt(t);
    gl.useProgram(fieldProgram); setView(fieldProgram);
    [velocityTexture(f.k0), velocityTexture(f.k1), signalTexture(f.k0), signalTexture(f.k1), solidTexture, luts[mode]]
      .forEach((tex, unit) => bind(unit, tex));
    ['uVel0', 'uVel1', 'uSig0', 'uSig1', 'uSolid', 'uLut'].forEach((name, unit) => gl.uniform1i(fieldProgram.u(name), unit));
    gl.uniform1f(fieldProgram.u('uMix'), f.mix);
    gl.uniform1f(fieldProgram.u('uVortMax'), meta.vorticity_max);
    gl.uniform1f(fieldProgram.u('uSpeedMax'), meta.speed_max);
    gl.uniform1f(fieldProgram.u('uSignal'), $('show-signal').checked ? 1 : 0);
    gl.uniform1i(fieldProgram.u('uMode'), mode);
    gl.uniform2f(fieldProgram.u('uRes'), NX, NY);
    gl.bindVertexArray(emptyVao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    if ($('show-tracers').checked) {
      gl.useProgram(trailProgram); setView(trailProgram);
      bind(0, trailTexture); gl.uniform1i(trailProgram.u('uTrail'), 0);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    drawn = [];
    let count = 0;
    for (let n = 0; n < N; n++) {
      for (const b of pose(n, t)) {
        const [length, width] = bodySize(b.index);
        const founder = A.founder[b.index];
        // The simulation has no solid collision: fade bodies whose centre is on a solid cell.
        const onSolid = A.solid[Math.min(NY - 1, Math.floor(b.y * NY)) * NX + Math.min(NX - 1, Math.floor(b.x * NX))];
        const alpha = b.alpha * (onSolid ? 0.3 : 1) * (isolate === null || isolate === (top.includes(founder) ? founder : -1) ? 1 : 0.16);
        const isSelected = selected !== null && A.lineage[b.index] === selected;
        drawn.push({ ...b, slot: n, length, width });
        instances.set([b.x, b.y, b.angle, b.phase, length * b.scale, width * b.scale, alpha, isSelected ? 1 : 0,
          ...rgbCache.get(colorOf(founder))], count * 11);
        count++;
      }
    }
    if ($('show-bodies').checked) {
      gl.enable(gl.SCISSOR_TEST); // clip to the simulated domain
      const sx = (x) => Math.round(w / 2 + (x - view.x) * ppu * dpr), sy = (y) => Math.round(h / 2 + (y - view.y) * ppu * dpr);
      gl.scissor(sx(0), sy(0), Math.max(0, sx(1) - sx(0)), Math.max(0, sy(1) - sy(0)));
      gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, instances.subarray(0, count * 11), gl.DYNAMIC_DRAW);
      gl.useProgram(bodyProgram); setView(bodyProgram); gl.bindVertexArray(bodyVaos[0]);
      gl.uniform1f(bodyProgram.u('uShadow'), 1);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
      gl.useProgram(tailProgram); setView(tailProgram); gl.bindVertexArray(bodyVaos[1]);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 50, count);
      gl.useProgram(bodyProgram); gl.bindVertexArray(bodyVaos[0]);
      gl.uniform1f(bodyProgram.u('uShadow'), 0);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count);
      gl.disable(gl.SCISSOR_TEST);
    }
    updateScaleBar(ppu);
  }

  /* panel */
  const fmt = (v, d = 3) => v.toFixed(d);
  function updateLegend() {
    const bar = $('colorbar').getContext('2d');
    for (let x = 0; x < 560; x++) {
      let v = x / 559;
      if (mode === 0) { const s = 2 * v - 1; v = 0.5 + 0.5 * Math.sign(s) * Math.abs(s) ** 0.8; }
      const i = Math.round(v * 255) * 4;
      bar.fillStyle = `rgb(${lutData[mode][i]},${lutData[mode][i + 1]},${lutData[mode][i + 2]})`;
      bar.fillRect(x, 0, 1, 20);
    }
    const ticks = $('ticks').children;
    const labels = mode === 0
      ? [`≤ −${fmt(meta.vorticity_max, 4)}`, '0', `≥ +${fmt(meta.vorticity_max, 4)}`]
      : ['0', fmt(meta.speed_max / 2), `≥ ${fmt(meta.speed_max)}`];
    labels.forEach((l, i) => { ticks[i].textContent = l; });
    $('unit').textContent = mode === 0
      ? 'Vorticity, lattice units (1 / LBM step). Copper: counter-clockwise; blue: clockwise.'
      : 'Flow speed, lattice units (cells / LBM step).';
  }
  const signalContext = $('signalbar').getContext('2d');
  const gradient = signalContext.createLinearGradient(0, 0, 560, 0);
  gradient.addColorStop(0, '#171b21'); gradient.addColorStop(1, '#9eedb3');
  signalContext.fillStyle = gradient; signalContext.fillRect(0, 0, 560, 12);
  $('signal-max').textContent = `≥ ${meta.signal_max.toPrecision(2)}`;
  $('signal-mid').textContent = (meta.signal_max / 4).toPrecision(2);
  $('subtitle').textContent = `${meta.scene === 'obstacle' ? 'Channel with cylinder' : 'Periodic box'}, ${NX}×${NY} cells, `
    + `ν = ${meta.viscosity}, two-way ${meta.coupling} coupling. Recorded replay, seed ${meta.seed}. Untrained controllers.`;

  const rows = [...top, -1].map((founder) => {
    const li = document.createElement('li');
    const dot = document.createElement('i'); dot.style.background = founder < 0 ? OTHER_COLOR : colorOf(founder);
    const name = document.createElement('span'); name.textContent = founder < 0 ? 'All others' : `Founder ${founder}`;
    const total = document.createElement('b');
    li.append(dot, name, total); li.title = 'Click to isolate';
    li.onclick = () => { isolate = isolate === founder ? null : founder; refresh(); };
    $('lineages').append(li);
    return { founder, li, total };
  });

  function selectedIndex() {
    if (selected === null) return null;
    const f = Math.min(STEPS, Math.round(t));
    for (let n = 0; n < N; n++) if (A.alive[f * N + n] && A.lineage[f * N + n] === selected) return f * N + n;
    return -1;
  }
  function updatePanel() {
    const f = Math.min(STEPS, Math.round(t));
    const counts = new Map();
    let alive = 0;
    for (let n = 0; n < N; n++) {
      const i = f * N + n;
      if (!A.alive[i]) continue;
      alive++;
      const key = top.includes(A.founder[i]) ? A.founder[i] : -1;
      counts.set(key, (counts.get(key) || 0) + 1);
    }
    for (const row of rows) {
      row.total.textContent = counts.get(row.founder) || 0;
      row.li.classList.toggle('dim', isolate !== null && isolate !== row.founder);
    }
    $('clock').textContent = `step ${t.toFixed(0)} / ${STEPS}  ·  ${alive} alive`;
    $('scrub').value = t;
    const index = selectedIndex();
    $('follow').hidden = index === null || index < 0;
    $('hint').hidden = index !== null;
    // make room for the inspector; the note stays one click away
    if (selected !== null && !provenanceClosed) { $('provenance').open = false; provenanceClosed = true; }
    const details = $('details');
    details.replaceChildren();
    if (index === null) return;
    const items = index < 0 ? [['Organism', `id ${selected}`], ['Status', 'not alive at this step']] : [
      ['Organism', `id ${selected}`], ['Founder / parent', `${A.founder[index]} / ${A.parent[index] < 0 ? 'none' : A.parent[index]}`],
      ['Generation depth', A.depth[index]], ['Age', `${A.age[index]} steps`], ['Energy', fmt(A.energy[index], 1)],
      ['Body radius gene', fmt(A.radius[index], 2)], ['Body aspect gene', fmt(A.aspect[index], 2)],
      ['Swim command', `${fmt(Math.hypot(A.swim[2 * index], A.swim[2 * index + 1]) * NX, 1)} cells/step`],
      ['Emitting signal', A.emitting[index] ? 'yes' : 'no'],
    ];
    for (const [k, v] of items) {
      const dt = document.createElement('dt'), dd = document.createElement('dd');
      dt.textContent = k; dd.textContent = v; details.append(dt, dd);
    }
  }
  function updateScaleBar(ppu) {
    const cells = [5, 10, 20, 50, 100].reverse().find((c) => c / NX * ppu <= 140) || 2;
    $('scalebar').firstElementChild.style.width = `${cells / NX * ppu}px`;
    $('scalebar').lastElementChild.textContent = `${cells} lattice cells  ·  flow →`;
  }
  function refresh() { updatePanel(); draw(); }

  /* controls */
  $('scrub').max = STEPS;
  $('scrub').oninput = (e) => { t = Number(e.target.value); needsTrailReset = true; };
  const setPlaying = (v) => { playing = v; $('play').textContent = v ? 'Pause' : 'Play'; };
  $('play').onclick = () => setPlaying(!playing);
  setPlaying(playing);
  const resetView = () => { Object.assign(view, { x: 0.5, y: 0.5, zoom: 1 }); follow = false; };
  $('reset').onclick = resetView;
  $('follow').onclick = () => { follow = !follow; if (follow) view.zoom = Math.max(view.zoom, 3); };
  const setMode = (m) => {
    mode = m;
    for (const b of $('field-mode').children) b.classList.toggle('on', Number(b.dataset.mode) === m);
    updateLegend();
  };
  for (const b of $('field-mode').children) b.onclick = () => setMode(Number(b.dataset.mode));
  setMode(mode);
  if (params.get('signal') === '0') $('show-signal').checked = false;
  window.onkeydown = (e) => {
    if (e.target.matches('input,select,button')) return;
    if (e.code === 'Space') { e.preventDefault(); setPlaying(!playing); }
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      t = Math.max(0, Math.min(STEPS, Math.round(t) + (e.key === 'ArrowRight' ? 1 : -1))); needsTrailReset = true; setPlaying(false);
    }
    if (e.key === '0') resetView();
  };
  const toWorld = (px, py) => [view.x + (px - canvas.clientWidth / 2) / pixelsPerUnit(),
    view.y - (py - canvas.clientHeight / 2) / pixelsPerUnit()];
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const [wx, wy] = toWorld(e.offsetX, e.offsetY);
    const zoom = Math.max(0.8, Math.min(10, view.zoom * Math.exp(-e.deltaY * 0.0015)));
    const k = view.zoom / zoom;
    view.x = wx + (view.x - wx) * k; view.y = wy + (view.y - wy) * k; view.zoom = zoom;
  }, { passive: false });
  let drag = null;
  canvas.onpointerdown = (e) => { drag = { x: e.clientX, y: e.clientY, moved: 0 }; canvas.setPointerCapture(e.pointerId); };
  canvas.onpointermove = (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.moved += Math.abs(dx) + Math.abs(dy); drag.x = e.clientX; drag.y = e.clientY;
    if (drag.moved > 4) { canvas.classList.add('dragging'); follow = false; view.x -= dx / pixelsPerUnit(); view.y += dy / pixelsPerUnit(); }
  };
  function pick(wx, wy) {
    let best = null, bestDistance = 1.8;
    for (const b of drawn) {
      const c = Math.cos(b.angle), s = Math.sin(b.angle), dx = wx - b.x, dy = wy - b.y;
      const d = Math.hypot((c * dx + s * dy) / (b.length / 2), (-s * dx + c * dy) / Math.max(b.width / 2, 6 / pixelsPerUnit()));
      if (d < bestDistance) { best = b; bestDistance = d; }
    }
    selected = best ? A.lineage[best.index] : null;
    if (!best) follow = false;
  }
  canvas.onpointerup = (e) => {
    if (drag && drag.moved <= 4) { pick(...toWorld(e.offsetX, e.offsetY)); refresh(); }
    drag = null; canvas.classList.remove('dragging');
  };

  /* loop */
  let last = null, panelClock = 0, frames = 0, fpsStart = performance.now();
  function advance(dt) {
    const dSteps = playing ? dt * Number($('rate').value) : 0;
    t += dSteps;
    if (t >= STEPS) { t = 0; needsTrailReset = true; }
    if (needsTrailReset) { resetTrails(t); needsTrailReset = false; }
    else if (dSteps > 0) {
      const parts = Math.ceil(dSteps / 0.25);
      for (let i = 0; i < parts; i++) stepTracers(t, dSteps / parts);
    }
    if (follow) {
      const target = drawn.find((b) => A.lineage[b.index] === selected);
      if (target) { view.x = target.x; view.y = target.y; }
    }
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
  /* automation hooks used by the capture script */
  window.viewer = {
    meta, fps: 0, steps: STEPS, advance: (dt) => { advance(dt); updatePanel(); },
    selectSlot(n) { const i = Math.round(t) * N + n; selected = A.alive[i] ? A.lineage[i] : null; refresh(); return selected; },
    bodies: () => drawn, setView: (v) => { Object.assign(view, v); refresh(); }, setMode: (m) => { setMode(m); refresh(); },
    seek(v) { t = v; needsTrailReset = true; advance(0); updatePanel(); },
    time: () => t,
    renderer: () => { const ext = gl.getExtension('WEBGL_debug_renderer_info'); return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); },
    glError: () => gl.getError(),
  };
  advance(0); updatePanel();
  if (!params.has('manual')) requestAnimationFrame(loop);
  window.viewer.ready = true;
}
main().catch((error) => { console.error(error); fail(`Viewer error: ${error.message}`); });
