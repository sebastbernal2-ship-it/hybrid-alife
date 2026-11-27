/* hybrid-alife ecosystem replay viewer. Canvas 2D, no dependencies.
 * World coordinates are grid cells on a torus: x = column, y = row.
 * Time t is in recorded frames (one simulation step each). */
'use strict';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const CLADE_COLOURS = ['#ff8f70', '#58c4f5', '#f1c453', '#b79cff', '#ff82bd', '#46d7c4',
  '#e4e9ef', '#7f9cff', '#d9a07b'];
const CLADE_NAMES = ['coral', 'sky', 'amber', 'lilac', 'rose', 'lagoon', 'pearl', 'cobalt', 'clay'];
const CAUSE_TEXT = { starved: 'starved', old_age: 'died of old age',
  hazard: 'died in a toxic patch', attacked: 'died while under attack' };
const SENSE_TEXT = { food_row: 'food, down(+) / up', food_col: 'food, right(+) / left',
  food_here: 'food underfoot', other_row: 'neighbour, down(+) / up',
  other_col: 'neighbour, right(+) / left', other_size: 'neighbour bigger(+)',
  hazard_row: 'toxin, down(+) / up', hazard_col: 'toxin, right(+) / left',
  hazard_here: 'toxin underfoot', energy: 'own energy' };
const OUTPUT_TEXT = ['move down / up', 'move right / left', 'eat', 'attack', 'breed'];

let R = null;               // decoded replay
let t = 0, playing = params.get('paused') !== '1', speed = Number(params.get('speed') || 8);
let selected = -1, follow = params.get('follow') === '1', hiddenClade = -1;
const view = { x: 32, y: 32, z: 1, fitted: 10 };   // centre in cells, z = zoom over fit
const world = $('world'), ctx = world.getContext('2d');
const strip = $('strip'), sctx = strip.getContext('2d');
let dpr = 1, dirtyStrip = true, stripBase = null;

/* ---------- loading ---------- */
async function load() {
  const binary = atob(window.REPLAY_B64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  const buffer = await new Response(stream).arrayBuffer();
  const head = new DataView(buffer);
  if (new TextDecoder().decode(new Uint8Array(buffer, 0, 4)) !== 'ECO1') throw new Error('not an ECO1 replay');
  const size = head.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, size)));
  const types = { uint8: Uint8Array, int8: Int8Array, uint16: Uint16Array, int32: Int32Array, float32: Float32Array };
  const a = {};
  for (const [name, s] of Object.entries(header.arrays)) {
    a[name] = new types[s.dtype](buffer, 8 + size + s.offset, s.count);
  }
  return prepare(header.meta, a);
}

function offsets(counts) {
  const out = new Int32Array(counts.length + 1);
  for (let i = 0; i < counts.length; i++) out[i + 1] = out[i] + counts[i];
  return out;
}

function prepare(meta, a) {
  const F = meta.frames, [H, W] = meta.grid;
  const r = { meta, a, F, H, W, off: {} };
  for (const k of ['agent', 'food', 'birth', 'death', 'attack']) r.off[k] = offsets(a[`n_${k}`]);
  r.body = new Map();
  a.body_id.forEach((id, i) => r.body.set(id, i));
  r.person = new Map();
  for (let i = 0; i < a.person.length; i += 6) {
    r.person.set(a.person[i], { id: a.person[i], parent: a.person[i + 1], born: a.person[i + 2],
      died: a.person[i + 3], cause: a.person[i + 4], founder: a.person[i + 5], kids: [] });
  }
  for (const p of r.person.values()) if (r.person.has(p.parent)) r.person.get(p.parent).kids.push(p.id);
  /* row of each individual in each frame, and heading from recorded movement */
  r.rows = new Array(F);
  const total = a.agent_id.length;
  r.heading = new Float32Array(total);
  r.speed = new Float32Array(total);
  const last = new Map();
  for (let f = 0; f < F; f++) {
    const map = new Map();
    for (let k = r.off.agent[f]; k < r.off.agent[f + 1]; k++) {
      const id = a.agent_id[k];
      map.set(id, k);
      const prev = last.get(id);
      let hx = Math.cos(id * 2.399), hy = Math.sin(id * 2.399), sp = 0;
      if (prev !== undefined) {
        const dx = wrapDelta((a.agent_pos[2 * k + 1] - a.agent_pos[2 * prev + 1]) / 65535) * W;
        const dy = wrapDelta((a.agent_pos[2 * k] - a.agent_pos[2 * prev]) / 65535) * H;
        sp = Math.hypot(dx, dy);
        const keep = 0.6, ph = r.heading[prev], n = sp > 0.02 ? 1 / sp : 0;
        hx = keep * Math.cos(ph) + (1 - keep) * (n ? dx * n : Math.cos(ph));
        hy = keep * Math.sin(ph) + (1 - keep) * (n ? dy * n : Math.sin(ph));
      }
      r.heading[k] = Math.atan2(hy, hx);
      r.speed[k] = sp;
      last.set(id, k);
    }
    r.rows[f] = map;
  }
  /* give each newborn the heading of its first movement */
  for (let f = 0; f + 1 < F; f++) {
    for (const [id, k] of r.rows[f]) {
      if ((f === 0 || !r.rows[f - 1].has(id)) && r.rows[f + 1].has(id)) r.heading[k] = r.heading[r.rows[f + 1].get(id)];
    }
  }
  /* food: key frames every 100 steps, deltas between */
  r.foodKeys = [];
  const food = Uint8Array.from(a.food_first);
  for (let f = 0; f < F; f++) {
    for (let i = r.off.food[f]; i < r.off.food[f + 1]; i++) food[a.food_cell[i]] = a.food_level[i];
    if (f % 100 === 0) r.foodKeys.push(Uint8Array.from(food));
  }
  r.food = Uint8Array.from(a.food_first);
  r.foodAt = 0;
  /* clade counts per frame */
  const nc = meta.clade_roots.length;
  r.cladeCount = new Uint16Array(F * nc);
  for (let f = 0; f < F; f++) {
    for (const id of r.rows[f].keys()) r.cladeCount[f * nc + a.body_clade[r.body.get(id)]]++;
  }
  return r;
}

const wrapDelta = (d) => d - Math.round(d);
const mod = (v, m) => ((v % m) + m) % m;

function foodAt(f) {
  if (f < r_foodFrame() || f - r_foodFrame() > 100) {
    const k = Math.floor(f / 100);
    R.food.set(R.foodKeys[k]);
    R.foodAt = k * 100;
  }
  for (let g = R.foodAt + 1; g <= f; g++) {
    for (let i = R.off.food[g]; i < R.off.food[g + 1]; i++) R.food[R.a.food_cell[i]] = R.a.food_level[i];
  }
  R.foodAt = f;
  return R.food;
}
const r_foodFrame = () => R.foodAt;

/* ---------- creature look ---------- */
function bodyOf(id) {
  const i = R.body.get(id);
  const radius = R.a.body_gene[2 * i], aspect = R.a.body_gene[2 * i + 1];
  const clade = R.a.body_clade[i];
  const base = 0.78 * radius;
  return { i, radius, aspect, clade, depth: R.a.body_depth[i],
    len: base * 1.35 * Math.pow(aspect, 0.9), wid: base * 0.9 / Math.pow(aspect, 0.9) };
}

const sprites = new Map();
function sprite(clade) {
  if (sprites.has(clade)) return sprites.get(clade);
  const c = document.createElement('canvas');
  c.width = c.height = 160;
  const g = c.getContext('2d'), colour = CLADE_COLOURS[clade];
  g.translate(80, 80);
  g.scale(64, 64);
  const path = () => {           // blunt head at +x, tapering tail at -x; spans -1..1
    g.beginPath();
    g.moveTo(1, 0);
    g.bezierCurveTo(1, 0.62, 0.55, 1, 0.05, 0.98);
    g.bezierCurveTo(-0.5, 0.94, -0.78, 0.4, -1, 0);
    g.bezierCurveTo(-0.78, -0.4, -0.5, -0.94, 0.05, -0.98);
    g.bezierCurveTo(0.55, -1, 1, -0.62, 1, 0);
    g.closePath();
  };
  path();
  const fill = g.createRadialGradient(0.15, -0.15, 0.05, 0, 0, 1.1);
  fill.addColorStop(0, colour + 'f0');
  fill.addColorStop(0.62, colour + '96');
  fill.addColorStop(1, colour + '58');
  g.fillStyle = fill;
  g.fill();
  g.lineWidth = 0.11;
  g.strokeStyle = colour;
  g.stroke();
  g.globalAlpha = 0.35;          // gut and organelles: decoration
  g.fillStyle = '#071012';
  for (const [x, y, rad] of [[-0.35, 0.12, 0.2], [-0.05, -0.34, 0.13], [-0.52, -0.16, 0.1]]) {
    g.beginPath(); g.ellipse(x, y, rad * 1.3, rad, 0.4, 0, 7); g.fill();
  }
  g.globalAlpha = 1;
  for (const side of [-1, 1]) {
    g.fillStyle = '#f4fbf7';
    g.beginPath(); g.ellipse(0.56, 0.36 * side, 0.2, 0.2, 0, 0, 7); g.fill();
    g.fillStyle = '#081012';
    g.beginPath(); g.ellipse(0.63, 0.36 * side, 0.11, 0.11, 0, 0, 7); g.fill();
  }
  sprites.set(clade, c);
  return c;
}

function drawCreature(g, x, y, heading, body, scale, energy, phase, speed, alpha) {
  const L = body.len * scale, Wd = body.wid * scale, colour = CLADE_COLOURS[body.clade];
  g.save();
  g.translate(x, y);
  g.rotate(heading);
  g.globalAlpha = alpha;
  if (L > 4) {                   // tail beat: decoration, faster when moving faster
    const swing = Math.sin(phase) * Wd * (0.3 + Math.min(speed, 0.6));
    g.strokeStyle = colour;
    g.lineCap = 'round';
    g.lineWidth = Math.max(1, Wd * 0.17);
    g.beginPath();
    g.moveTo(-L * 0.92, 0);
    g.quadraticCurveTo(-L * 1.35, swing, -L * 1.8, -swing * 0.7);
    g.stroke();
  }
  g.globalAlpha = alpha * (0.5 + 0.5 * Math.min(1, energy / 20));  // pale when starving
  g.drawImage(sprite(body.clade), -L * 1.25, -Wd * 1.25, L * 2.5, Wd * 2.5);
  g.restore();
}

/* ---------- camera ---------- */
function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  for (const c of [world, strip]) {
    c.width = Math.max(2, Math.round(c.clientWidth * dpr));
    c.height = Math.max(2, Math.round(c.clientHeight * dpr));
  }
  view.fitted = (Math.min(world.width, world.height) - 16 * dpr) / 64;
  backdrop = null;
  stripBase = null;
  dirtyStrip = true;
}
const pxPerCell = () => view.fitted * view.z;
const toScreen = (x, y) => [world.width / 2 + (x - view.x) * pxPerCell(), world.height / 2 + (y - view.y) * pxPerCell()];

let backdrop = null;
function plateTexture() {
  if (backdrop) return backdrop;
  const c = document.createElement('canvas');
  c.width = c.height = 1024;
  const g = c.getContext('2d');
  g.fillStyle = '#0d191b';
  g.fillRect(0, 0, 1024, 1024);
  let seed = 7;
  const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 240; i++) {   // soft mottling, wrapped so the seam is invisible
    const x = rand() * 1024, y = rand() * 1024, rad = 20 + rand() * 90;
    const tone = rand() < 0.5 ? '70,120,110' : '3,10,14';
    for (const ox of [-1024, 0, 1024]) for (const oy of [-1024, 0, 1024]) {
      const grad = g.createRadialGradient(x + ox, y + oy, 0, x + ox, y + oy, rad);
      grad.addColorStop(0, `rgba(${tone},0.07)`);
      grad.addColorStop(1, `rgba(${tone},0)`);
      g.fillStyle = grad;
      g.fillRect(x + ox - rad, y + oy - rad, 2 * rad, 2 * rad);
    }
  }
  for (let i = 0; i < 2600; i++) {
    g.fillStyle = `rgba(190,220,205,${0.03 + rand() * 0.07})`;
    g.fillRect(rand() * 1024, rand() * 1024, 1.5, 1.5);
  }
  return (backdrop = c);
}

let pelletSprite = null;
function pellet() {
  if (pelletSprite) return pelletSprite;
  const c = document.createElement('canvas');
  c.width = c.height = 48;
  const g = c.getContext('2d');
  const glow = g.createRadialGradient(24, 24, 2, 24, 24, 23);
  glow.addColorStop(0, 'rgba(150,215,90,0.35)');
  glow.addColorStop(1, 'rgba(150,215,90,0)');
  g.fillStyle = glow;
  g.fillRect(0, 0, 48, 48);
  for (const [x, y, rad] of [[21, 22, 7], [30, 26, 5.5], [24, 31, 4.5]]) {
    g.fillStyle = '#83bd4f';
    g.beginPath(); g.arc(x, y, rad, 0, 7); g.fill();
    g.fillStyle = '#d3f59b';
    g.beginPath(); g.arc(x - rad * 0.3, y - rad * 0.3, rad * 0.4, 0, 7); g.fill();
  }
  return (pelletSprite = c);
}

let toxinSprite = null;
function toxin() {
  if (toxinSprite) return toxinSprite;
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(128, 128, 0, 128, 128, 126);
  grad.addColorStop(0, 'rgba(168,52,112,0.55)');
  grad.addColorStop(0.75, 'rgba(130,38,96,0.34)');
  grad.addColorStop(1, 'rgba(130,38,96,0.16)');
  g.fillStyle = grad;
  g.beginPath(); g.arc(128, 128, 126, 0, 7); g.fill();
  for (let i = 1; i < 190; i++) {
    const rad = 118 * Math.sqrt(i / 190), ang = i * 2.39996;
    g.fillStyle = 'rgba(236,120,180,0.34)';
    g.beginPath(); g.arc(128 + rad * Math.cos(ang), 128 + rad * Math.sin(ang), 1.2 + (i % 5) * 0.5, 0, 7); g.fill();
  }
  g.setLineDash([7, 8]);
  g.strokeStyle = 'rgba(240,128,188,0.7)';
  g.lineWidth = 2.5;
  g.beginPath(); g.arc(128, 128, 126, 0, 7); g.stroke();
  return (toxinSprite = c);
}

/* position of an individual at fractional time, in cells */
function place(id, f, u) {
  const k = R.rows[f].get(id);
  if (k === undefined) return null;
  let y = R.a.agent_pos[2 * k] / 65535, x = R.a.agent_pos[2 * k + 1] / 65535;
  let heading = R.heading[k];
  const n = f + 1 < R.F ? R.rows[f + 1].get(id) : undefined;
  if (n !== undefined && u > 0) {
    x += u * wrapDelta(R.a.agent_pos[2 * n + 1] / 65535 - x);
    y += u * wrapDelta(R.a.agent_pos[2 * n] / 65535 - y);
    const turn = Math.atan2(Math.sin(R.heading[n] - heading), Math.cos(R.heading[n] - heading));
    heading += u * turn;
  }
  return { x: mod(x, 1) * R.W, y: mod(y, 1) * R.H, heading, k };
}

/* shortest displacement from a to b on the torus */
const near = (ax, ay, bx, by) => [ax + wrapDelta((bx - ax) / R.W) * R.W, ay + wrapDelta((by - ay) / R.H) * R.H];

/* ---------- world drawing ---------- */
function drawWorld(now) {
  const f = Math.min(R.F - 1, Math.floor(t)), u = t - f;
  const s = pxPerCell(), w = world.width, h = world.height;
  if (follow && selected >= 0) {
    const p = place(selected, f, u);
    if (p) {
      const [tx, ty] = near(view.x, view.y, p.x, p.y);
      view.x += (tx - view.x) * 0.15;
      view.y += (ty - view.y) * 0.15;
    }
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#070c0e';
  ctx.fillRect(0, 0, w, h);
  const [px, py] = toScreen(0, 0), side = 64 * s;
  /* the plate is one period of the torus; zoomed in, neighbouring periods fill the view */
  const wrapView = view.z > 1.05 || follow;
  const clip = wrapView ? [0, 0, w, h] : [w / 2 - side / 2, h / 2 - side / 2, side, side];
  ctx.save();
  ctx.beginPath();
  if (wrapView) ctx.rect(...clip);
  else ctx.roundRect(...clip, 14 * dpr);
  ctx.clip();
  const tiles = [];
  const i0 = Math.floor((clip[0] - px) / side) - (wrapView ? 0 : 1);
  const j0 = Math.floor((clip[1] - py) / side) - (wrapView ? 0 : 1);
  for (let i = i0; px + i * side < clip[0] + clip[2] + side * 0.05; i++) {
    for (let j = j0; py + j * side < clip[1] + clip[3] + side * 0.05; j++) tiles.push([i * 64, j * 64]);
  }
  for (const [ox, oy] of tiles) {
    const [x, y] = toScreen(ox, oy);
    ctx.drawImage(plateTexture(), x, y, side + 0.7, side + 0.7);
  }
  const visible = (x, y, m) => x > -m && y > -m && x < w + m && y < h + m;
  const each = (cx, cy, margin, draw) => {
    for (const [ox, oy] of tiles) {
      const [x, y] = toScreen(cx + ox, cy + oy);
      if (visible(x, y, margin)) draw(x, y);
    }
  };
  /* toxins */
  const hr = R.meta.hazard_draw_radius * 64 * s, np = R.meta.hazard_patches;
  for (let i = 0; i < np; i++) {
    const k = 2 * (f * np + i);
    each(R.a.hazard_pos[k + 1] / 65535 * 64, R.a.hazard_pos[k] / 65535 * 64, hr,
      (x, y) => ctx.drawImage(toxin(), x - hr, y - hr, 2 * hr, 2 * hr));
  }
  /* food */
  const food = foodAt(f);
  for (let cell = 0; cell < food.length; cell++) {
    const level = food[cell] / 255;
    if (level <= 0) continue;
    const cx = (cell % 64) + 0.5 + 0.22 * Math.sin(cell * 12.9898), cy = Math.floor(cell / 64) + 0.5 + 0.22 * Math.cos(cell * 78.233);
    const size = s * (0.55 + 0.75 * Math.sqrt(level));
    ctx.globalAlpha = 0.45 + 0.55 * level;
    each(cx, cy, size, (x, y) => ctx.drawImage(pellet(), x - size, y - size, 2 * size, 2 * size));
  }
  ctx.globalAlpha = 1;

  const here = new Map();
  for (const id of R.rows[f].keys()) here.set(id, place(id, f, u));

  /* deaths: the body dissolves where it was */
  for (let g = Math.max(1, f - 13); g <= f; g++) {
    const age = (t - g) / 14;
    for (let i = R.off.death[g]; i < R.off.death[g + 1]; i++) {
      const id = R.a.death[2 * i], cause = R.meta.causes[R.a.death[2 * i + 1]];
      const dx = R.a.death_pos[2 * i + 1] / 65535 * 64, dy = R.a.death_pos[2 * i] / 65535 * 64;
      const body = bodyOf(id);
      each(dx, dy, 6 * s, (x, y) => {
        const k = R.rows[g - 1].get(id);
        if (k !== undefined) drawCreature(ctx, x, y, R.heading[k], body, s * (1 - 0.4 * age), 3, 0, 0, 0.5 * (1 - age));
        ctx.globalAlpha = Math.max(0, 1 - age);
        ctx.fillStyle = cause === 'hazard' ? '#f07ebb' : cause === 'attacked' ? '#ff6256' : '#b8c4c8';
        for (let j = 0; j < 9; j++) {
          const ang = j * 0.698 + id, rad = s * (0.4 + age * (1.4 + (j % 3) * 0.5));
          ctx.beginPath(); ctx.arc(x + Math.cos(ang) * rad, y + Math.sin(ang) * rad, s * 0.13 * (1 - age) + 0.6 * dpr, 0, 7); ctx.fill();
        }
        ctx.setLineDash([3 * dpr, 3 * dpr]);
        ctx.strokeStyle = ctx.fillStyle;
        ctx.lineWidth = 1.2 * dpr;
        ctx.beginPath(); ctx.arc(x, y, s * (0.9 + age), 0, 7); ctx.stroke();
        ctx.setLineDash([]);
        if (s > 22 * dpr) {
          ctx.font = `${11 * dpr}px Segoe UI, sans-serif`;
          ctx.textAlign = 'center';
          ctx.fillText(CAUSE_TEXT[cause], x, y - s * (1.6 + age));
        }
        ctx.globalAlpha = 1;
      });
    }
  }
  /* newborns and their tie to the parent */
  const born = new Map();
  for (let g = Math.max(1, f - 11); g <= f; g++) {
    for (let i = R.off.birth[g]; i < R.off.birth[g + 1]; i++) born.set(R.a.birth[2 * i], { age: (t - g) / 12, parent: R.a.birth[2 * i + 1] });
  }
  for (const [id, b] of born) {
    const c = here.get(id), p = here.get(b.parent);
    if (!c) continue;
    const fade = Math.max(0, 1 - b.age);
    each(c.x, c.y, 8 * s, (x, y) => {
      ctx.strokeStyle = `rgba(255,241,190,${0.9 * fade})`;
      ctx.lineWidth = Math.max(1, s * 0.12);
      if (p) {
        const [qx, qy] = near(c.x, c.y, p.x, p.y);
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + (qx - c.x) * s, y + (qy - c.y) * s); ctx.stroke();
      }
      ctx.lineWidth = Math.max(1, s * 0.16 * fade);
      ctx.beginPath(); ctx.arc(x, y, s * (0.7 + 2.2 * b.age), 0, 7); ctx.stroke();
    });
  }
  /* creatures */
  for (const [id, p] of here) {
    const body = bodyOf(id), k = p.k;
    const energy = R.a.agent_energy[k] / R.meta.scale.energy;
    const b = born.get(id);
    const grow = b ? 0.55 + 0.45 * Math.min(1, b.age) : 1;
    const dim = hiddenClade >= 0 && body.clade !== hiddenClade ? 0.16 : 1;
    const phase = now * 0.004 * (1 + 5 * R.speed[k]) + id;
    each(p.x, p.y, 4 * s, (x, y) => drawCreature(ctx, x, y, p.heading, body, s * grow, energy, phase, R.speed[k], dim));
  }
  /* feeding: motes drawn toward the mouth of anyone who took in food this step */
  for (const [id, p] of here) {
    const intake = R.a.agent_intake[p.k] / R.meta.scale.intake;
    if (intake < 0.2) continue;
    const body = bodyOf(id);
    const mx = p.x + Math.cos(p.heading) * body.len, my = p.y + Math.sin(p.heading) * body.len;
    each(mx, my, 3 * s, (x, y) => {
      ctx.fillStyle = '#e4ffae';
      const n = Math.min(6, 2 + Math.floor(intake * 4));
      for (let j = 0; j < n; j++) {
        const q = 1 - mod(now * 0.0022 + j / n + id * 0.37, 1);
        const ang = p.heading + (j / n - 0.5) * 2.2;
        ctx.globalAlpha = 1 - q * 0.6;
        ctx.beginPath(); ctx.arc(x + Math.cos(ang) * q * s * 1.1, y + Math.sin(ang) * q * s * 1.1, Math.max(1, s * 0.1), 0, 7); ctx.fill();
      }
      ctx.globalAlpha = 1;
    });
  }
  /* attacks: only the recorded attacker and target pairs */
  for (let i = R.off.attack[f]; i < R.off.attack[f + 1]; i++) {
    const a = here.get(R.a.attack[2 * i]), b = here.get(R.a.attack[2 * i + 1]);
    if (!a || !b) continue;
    each(a.x, a.y, 5 * s, (x, y) => {
      const [bx, by] = near(a.x, a.y, b.x, b.y);
      const tx = x + (bx - a.x) * s, ty = y + (by - a.y) * s;
      ctx.strokeStyle = '#ff4d4d';
      ctx.lineWidth = Math.max(1.5 * dpr, s * 0.17);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(tx, ty); ctx.stroke();
      const spin = now * 0.006 + i;
      ctx.lineWidth = Math.max(1, s * 0.1);
      for (let j = 0; j < 4; j++) {
        const ang = spin + j * Math.PI / 2;
        ctx.beginPath();
        ctx.moveTo(tx + Math.cos(ang) * s * 0.55, ty + Math.sin(ang) * s * 0.55);
        ctx.lineTo(tx + Math.cos(ang) * s * 1.05, ty + Math.sin(ang) * s * 1.05);
        ctx.stroke();
      }
    });
  }
  /* the selected creature: what it senses right now */
  const p = selected >= 0 ? here.get(selected) : null;
  if (p) {
    const body = bodyOf(selected);
    each(p.x, p.y, 8 * s, (x, y) => {
      ctx.strokeStyle = 'rgba(232,217,168,0.95)';
      ctx.lineWidth = 1.6 * dpr;
      ctx.beginPath(); ctx.arc(x, y, s * (body.len + 0.65), 0, 7); ctx.stroke();
      ctx.setLineDash([3 * dpr, 6 * dpr]);
      ctx.strokeStyle = 'rgba(232,217,168,0.4)';
      ctx.beginPath(); ctx.arc(x, y, s * 6, 0, 7); ctx.stroke();
      ctx.setLineDash([]);
      [[0, '#c6f386', 'food'], [3, '#ffffff', 'neighbour'], [6, '#ff8cc6', 'toxin']].forEach(([c, colour, label]) => {
        const row = R.a.agent_sense[p.k * 10 + c] / 63, col = R.a.agent_sense[p.k * 10 + c + 1] / 63;
        const len = Math.hypot(row, col);
        if (len < 0.02) return;
        /* invert closeness = 1 - 0.7 d / 6 to point at the sensed cell */
        const dist = Math.max(0.4, (1 - Math.min(len, 1)) * 6 / 0.7);
        const ex = x + col / len * dist * s, ey = y + row / len * dist * s;
        ctx.strokeStyle = colour; ctx.fillStyle = colour; ctx.lineWidth = 1.2 * dpr;
        ctx.globalAlpha = 0.85;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.beginPath(); ctx.arc(ex, ey, 3 * dpr, 0, 7); ctx.stroke();
        if (s > 14 * dpr) { ctx.font = `${10.5 * dpr}px Segoe UI, sans-serif`; ctx.textAlign = 'left'; ctx.fillText(label, ex + 5 * dpr, ey - 4 * dpr); }
        ctx.globalAlpha = 1;
      });
    });
  }
  ctx.restore();
  if (!wrapView) {
    ctx.strokeStyle = '#24363a';
    ctx.lineWidth = dpr;
    ctx.beginPath(); ctx.roundRect(...clip, 14 * dpr); ctx.stroke();
  }
}

/* ---------- timeline strip ---------- */
function stripLayout() {
  const w = strip.width, h = strip.height, runW = Math.round(w * 0.24);
  return { w, h, runW, x0: runW + 26 * dpr, x1: w - 12 * dpr, top: 16 * dpr, bottom: h - 20 * dpr };
}
function drawStripBase() {
  const L = stripLayout();
  const c = document.createElement('canvas');
  c.width = L.w; c.height = L.h;
  const g = c.getContext('2d');
  g.font = `${10.5 * dpr}px Segoe UI, sans-serif`;
  g.fillStyle = '#80959a';
  const hist = R.meta.history;
  if (hist) {
    const maxStep = hist.step[hist.step.length - 1], maxPop = Math.max(...hist.population_max);
    const X = (v) => 12 * dpr + v / maxStep * (L.runW - 12 * dpr);
    const Y = (v) => L.bottom - v / maxPop * (L.bottom - L.top - 3 * dpr);
    g.fillStyle = 'rgba(160,200,190,0.18)';
    g.beginPath();
    hist.step.forEach((s, i) => (i ? g.lineTo(X(s), Y(hist.population_max[i])) : g.moveTo(X(s), Y(hist.population_max[i]))));
    for (let i = hist.step.length - 1; i >= 0; i--) g.lineTo(X(hist.step[i]), Y(hist.population_min[i]));
    g.fill();
    g.strokeStyle = '#b6d8cd'; g.lineWidth = 1.4 * dpr; g.beginPath();
    hist.step.forEach((s, i) => (i ? g.lineTo(X(s), Y(hist.population_mean[i])) : g.moveTo(X(s), Y(hist.population_mean[i]))));
    g.stroke();
    const wx = X(R.meta.start_step), ww = Math.max(3 * dpr, X(R.meta.start_step + R.F) - wx);
    g.fillStyle = '#e8d9a8'; g.fillRect(wx - ww, L.top, ww, L.bottom - L.top);
    g.fillStyle = '#80959a';
    g.textAlign = 'left';
    g.fillText(`whole run: population (peak ${maxPop}), founder lines ${hist.founders} → ${hist.founder_lineages[hist.founder_lineages.length - 1]}`, 12 * dpr, 11 * dpr);
    g.fillText('step 0', 12 * dpr, L.h - 6 * dpr);
    g.textAlign = 'right';
    g.fillText(`${maxStep.toLocaleString()}  (gold = this replay)`, L.runW, L.h - 6 * dpr);
  }
  /* this replay: stacked families */
  const nc = R.meta.clade_roots.length;
  let maxN = 1;
  for (let f = 0; f < R.F; f++) maxN = Math.max(maxN, R.off.agent[f + 1] - R.off.agent[f]);
  const X = (f) => L.x0 + f / (R.F - 1) * (L.x1 - L.x0);
  const popH = (L.bottom - L.top) * 0.72;
  const base = L.top + popH;
  const stack = new Float32Array(R.F);
  const every = Math.max(1, Math.floor(R.F / 900));
  for (let k = 0; k < nc; k++) {
    const lower = Float32Array.from(stack);
    for (let f = 0; f < R.F; f++) stack[f] += R.cladeCount[f * nc + k];
    g.fillStyle = CLADE_COLOURS[k] + 'cc';
    g.beginPath();
    for (let f = 0; f < R.F; f += every) g.lineTo(X(f), base - stack[f] / maxN * popH);
    for (let f = R.F - 1; f >= 0; f -= every) g.lineTo(X(f), base - lower[f] / maxN * popH);
    g.fill();
  }
  const bins = Math.min(220, R.F), mid = base + (L.bottom - base) * 0.52;
  const sums = { birth: new Array(bins).fill(0), death: new Array(bins).fill(0) };
  for (const k of ['birth', 'death']) for (let f = 0; f < R.F; f++) sums[k][Math.floor(f * bins / R.F)] += R.a[`n_${k}`][f];
  const top = Math.max(1, ...sums.birth, ...sums.death), half = (L.bottom - base) * 0.5, bw = (L.x1 - L.x0) / bins;
  for (let i = 0; i < bins; i++) {
    g.fillStyle = '#fff1be';
    g.fillRect(L.x0 + i * bw, mid - sums.birth[i] / top * half, Math.max(1, bw - 1), sums.birth[i] / top * half);
    g.fillStyle = '#74868c';
    g.fillRect(L.x0 + i * bw, mid + 1, Math.max(1, bw - 1), sums.death[i] / top * half);
  }
  g.fillStyle = '#80959a'; g.textAlign = 'left';
  g.fillText(`this replay: alive by family (peak ${maxN}) · births above the line, deaths below`, L.x0, 11 * dpr);
  g.fillText(`step ${R.meta.start_step.toLocaleString()}`, L.x0, L.h - 6 * dpr);
  g.textAlign = 'right';
  g.fillText(`step ${(R.meta.start_step + R.F - 1).toLocaleString()}`, L.x1, L.h - 6 * dpr);
  return c;
}
function drawStrip() {
  const L = stripLayout();
  stripBase = stripBase || drawStripBase();
  sctx.clearRect(0, 0, L.w, L.h);
  sctx.drawImage(stripBase, 0, 0);
  const x = L.x0 + t / (R.F - 1) * (L.x1 - L.x0);
  sctx.fillStyle = '#fff';
  sctx.fillRect(x - dpr, L.top - 3 * dpr, 2 * dpr, L.bottom - L.top + 6 * dpr);
}

/* ---------- side panels ---------- */
let panelFrame = -1, treeFor = null, feedFrame = -1;
function chip(id, extra) {
  const p = R.person.get(id);
  const el = document.createElement('span');
  const bi = R.body.get(id);
  el.className = `chip ${extra || ''}`;
  if (bi === undefined) el.classList.add('old');
  const alive = R.rows[Math.floor(t)].has(id);
  if (!alive && !extra) el.classList.add('gone');
  const dot = document.createElement('i');
  dot.style.background = bi === undefined ? '#566' : CLADE_COLOURS[R.a.body_clade[bi]];
  const fate = alive ? '' : p.born > R.meta.start_step + t ? ' not yet born' : p.died >= 0 && p.died <= R.meta.start_step + t ? ' †' : '';
  el.append(dot, `#${id}${fate}`);
  el.title = `born step ${p.born}` + (p.died >= 0 ? `, ${CAUSE_TEXT[R.meta.causes[p.cause]]} at step ${p.died}` : ', alive at the end of the replay');
  if (bi !== undefined && extra !== 'me') el.onclick = () => select(id, true);
  return el;
}
function buildTree() {
  const root = $('tree'), me = R.person.get(selected);
  const chain = [];
  for (let p = R.person.get(me.parent); p; p = R.person.get(p.parent)) chain.push(p.id);
  root.textContent = '';
  const line = (parent, ...nodes) => { const d = document.createElement('div'); d.append(...nodes); parent.append(d); return d; };
  if (chain.length) {
    const shown = chain.slice(0, 4).reverse();
    if (chain.length > 4) line(root, `↑ ${chain.length - 4} earlier ancestors, back to founder #${chain[chain.length - 1]}`);
    let host = root;
    for (const id of shown) {
      line(host, chip(id), host === root ? '' : '');
      const next = document.createElement('div'); next.className = 'trunk'; host.append(next); host = next;
    }
    line(host, chip(selected, 'me'), ' this one');
    appendKids(host, me, chain);
  } else {
    line(root, chip(selected, 'me'), ' a founder');
    appendKids(root, me, chain);
  }
}
function appendKids(host, me) {
  const trunk = document.createElement('div'); trunk.className = 'trunk'; host.append(trunk);
  const now = R.meta.start_step + Math.floor(t);
  const kids = me.kids.filter((id) => R.person.get(id).born <= now);
  const d = document.createElement('div');
  if (!kids.length) d.append('no children recorded yet');
  else {
    d.append(`${kids.length} ${kids.length > 1 ? 'children' : 'child'} with a living line or seen here: `);
    kids.slice(-14).forEach((id) => d.append(chip(id)));
  }
  trunk.append(d);
  const parent = R.person.get(me.parent);
  if (parent) {
    const sibs = parent.kids.filter((id) => id !== me.id && R.person.get(id).born <= now);
    if (sibs.length) {
      const s = document.createElement('div'); s.style.marginTop = '5px';
      s.append(`siblings: `); sibs.slice(-10).forEach((id) => s.append(chip(id))); host.parentElement.append(s);
    }
  }
}

const diverge = (v, a = 1) => (v >= 0 ? `rgba(255,170,92,${a})` : `rgba(92,176,255,${a})`);
function drawBrain(k) {
  const c = $('brain'), g = c.getContext('2d'), a = R.a, H = R.meta.hidden, no = OUTPUT_TEXT.length;
  g.setTransform(2, 0, 0, 2, 0, 0);
  g.clearRect(0, 0, 330, 280);
  const bi = R.body.get(selected);
  const inY = (i) => 14 + i * 28, hidY = (i) => 30 + i * 31.5, outY = (i) => 32 + i * 54;
  const xi = 128, xh = 190, xo = 248;
  const sense = (i) => a.agent_sense[k * 10 + i] / 63, hid = (i) => a.agent_hidden[k * H + i] / 127;
  const out = (i) => a.agent_output[k * no + i] / 127;
  g.lineCap = 'round';
  const edge = (x1, y1, x2, y2, v) => {
    const m = Math.min(1, Math.abs(v) / 1.1);
    if (m < 0.09) return;
    g.strokeStyle = diverge(v, 0.12 + 0.78 * m);
    g.lineWidth = 0.4 + 2.2 * m;
    g.beginPath(); g.moveTo(x1, y1); g.bezierCurveTo(x1 + 30, y1, x2 - 30, y2, x2, y2); g.stroke();
  };
  for (let i = 0; i < 10; i++) for (let h = 0; h < H; h++) edge(xi, inY(i), xh, hidY(h), a.brain_in[(bi * 10 + i) * H + h] / 32 * sense(i));
  for (let h = 0; h < H; h++) for (let o = 0; o < no; o++) edge(xh, hidY(h), xo, outY(o), a.brain_out[(bi * H + h) * no + o] / 32 * hid(h));
  const node = (x, y, v, rad) => {
    g.fillStyle = '#0f171a'; g.beginPath(); g.arc(x, y, rad, 0, 7); g.fill();
    g.fillStyle = diverge(v, Math.min(1, Math.abs(v))); g.fill();
    g.strokeStyle = '#7f959b'; g.lineWidth = 0.8; g.stroke();
  };
  g.font = '10px Segoe UI, sans-serif';
  g.textAlign = 'right';
  R.meta.senses.forEach((name, i) => {
    g.fillStyle = Math.abs(sense(i)) > 0.05 ? '#dde8e4' : '#62767b';
    g.fillText(SENSE_TEXT[name], xi - 9, inY(i) + 1);
    g.fillStyle = '#80959a';
    g.fillText(sense(i).toFixed(2), xi - 9, inY(i) + 11);
    node(xi, inY(i), sense(i), 5);
  });
  for (let h = 0; h < H; h++) node(xh, hidY(h), hid(h), 6.5);
  g.textAlign = 'left';
  OUTPUT_TEXT.forEach((name, o) => {
    const v = out(o);
    node(xo, outY(o), v, 6.5);
    g.fillStyle = '#dde8e4'; g.fillText(name, xo + 11, outY(o) - 3);
    g.fillStyle = '#1a272b'; g.fillRect(xo + 11, outY(o) + 3, 66, 5);
    g.fillStyle = diverge(v);
    if (o < 2) g.fillRect(xo + 44 + Math.min(0, v) * 33, outY(o) + 3, Math.abs(v) * 33, 5);
    else g.fillRect(xo + 11, outY(o) + 3, v * 66, 5);
    if (o === 3 || o === 4) {         // action thresholds used by the simulator
      const th = o === 3 ? 0.5 : 0.4;
      g.fillStyle = '#fff'; g.fillRect(xo + 11 + th * 66, outY(o) + 1, 1, 9);
      g.fillStyle = v > th ? '#fff1be' : '#62767b';
      g.fillText(v > th ? (o === 3 ? 'attacking' : 'willing') : 'off', xo + 11, outY(o) + 19);
    }
  });
}
function drawGenes(f) {
  const c = $('genes'), g = c.getContext('2d'), me = bodyOf(selected);
  g.setTransform(2, 0, 0, 2, 0, 0);
  g.clearRect(0, 0, 330, 58);
  const alive = [...R.rows[f].keys()].map(bodyOf);
  [['size', 'radius', 0.6, 1.8, 0], ['elongation', 'aspect', 0.7, 1.4, 168]].forEach(([label, key, lo, hi, x]) => {
    const bins = new Array(18).fill(0);
    for (const b of alive) bins[Math.min(17, Math.max(0, Math.floor((b[key] - lo) / (hi - lo) * 18)))]++;
    const top = Math.max(1, ...bins);
    g.fillStyle = '#3d565d';
    bins.forEach((n, i) => g.fillRect(x + i * 9, 40 - n / top * 28, 8, n / top * 28));
    const mx = x + (me[key] - lo) / (hi - lo) * 160;
    g.fillStyle = '#e8d9a8'; g.fillRect(mx, 9, 2, 33);
    g.font = '10px Segoe UI, sans-serif'; g.fillStyle = '#dde8e4'; g.textAlign = 'left';
    g.fillText(`${label} ${me[key].toFixed(2)}`, x, 53);
    g.fillStyle = '#80959a'; g.textAlign = 'right';
    g.fillText(`${lo}–${hi}`, x + 160, 53);
  });
}

function select(id, jump) {
  selected = id;
  treeFor = null;
  panelFrame = -1;
  $('empty').hidden = id >= 0;
  $('card').hidden = id < 0;
  if (id < 0) { follow = false; $('follow').classList.remove('on'); return; }
  if (jump && !R.rows[Math.floor(t)].has(id)) {
    const p = R.person.get(id);
    const from = Math.max(0, p.born - R.meta.start_step);
    if (from < R.F && R.rows[from].has(id)) t = from;
  }
}

function updatePanels(now) {
  const f = Math.min(R.F - 1, Math.floor(t));
  if (f !== panelFrame) {
    panelFrame = f;
    const nc = R.meta.clade_roots.length;
    $('c-pop').textContent = R.off.agent[f + 1] - R.off.agent[f];
    $('c-births').textContent = R.off.birth[f + 1];
    $('c-deaths').textContent = R.off.death[f + 1];
    [...$('clades').children].forEach((li, k) => { li.lastChild.textContent = R.cladeCount[f * nc + k]; });
    $('clock').textContent = `step ${(R.meta.start_step + f).toLocaleString()}  ·  ${f + 1} / ${R.F}  ·  seed ${R.meta.seed}`;
    if (selected >= 0) {
      const body = bodyOf(selected), p = R.person.get(selected), k = R.rows[f].get(selected);
      $('who').textContent = `Creature #${selected}`;
      $('who').style.color = CLADE_COLOURS[body.clade];
      const now_ = R.meta.start_step + f;
      let status = `${CLADE_NAMES[body.clade]} family · generation ${body.depth}`;
      if (k === undefined) status += p.born > now_ ? ' · not yet born' : ` · ${CAUSE_TEXT[R.meta.causes[p.cause]] || 'gone'} at step ${p.died.toLocaleString()}`;
      else status += ' · alive';
      $('status').textContent = status;
      const energy = k === undefined ? 0 : R.a.agent_energy[k] / R.meta.scale.energy;
      const age = k === undefined ? Math.max(0, (p.died >= 0 ? Math.min(p.died, now_) : now_) - p.born) : R.a.agent_age[k];
      $('v-energy').textContent = `${energy.toFixed(1)} / ${R.meta.max_energy}`;
      $('b-energy').style.width = `${100 * energy / R.meta.max_energy}%`;
      $('m-breed').style.left = `${100 * R.meta.breed_energy / R.meta.max_energy}%`;
      $('v-age').textContent = `${age} steps`;
      $('b-age').style.width = `${Math.min(100, age / (R.meta.senescence_age * 2) * 100)}%`;
      $('m-old').style.left = '50%';
      const pc = $('portrait').getContext('2d');
      pc.clearRect(0, 0, 192, 128);
      drawCreature(pc, 108, 64, 0, body, 54 / 1.2, k === undefined ? 4 : energy, f * 0.5, 0.3, k === undefined ? 0.4 : 1);
      drawGenes(f);
      if (k === undefined) $('brain').getContext('2d').clearRect(0, 0, 660, 560);
      else drawBrain(k);
      const living = p.kids.filter((i) => R.person.get(i).born <= now_);
      const key = `${selected}:${living.length}:${living.filter((i) => R.rows[f].has(i)).length}:${k === undefined}`;
      if (key !== treeFor) { treeFor = key; buildTree(); }
    }
  }
  if (Math.abs(f - feedFrame) >= 4) {
    feedFrame = f;
    const feed = $('feed');
    feed.textContent = '';
    let shown = 0;
    for (let g = f; g >= Math.max(1, f - 80) && shown < 6; g--) {
      const add = (text, id) => { if (shown++ < 6) { const li = document.createElement('li'); li.textContent = `${R.meta.start_step + g}  ${text}`; li.onclick = () => { t = Math.max(0, g - 2); select(id, false); }; feed.append(li); } };
      for (let i = R.off.birth[g]; i < R.off.birth[g + 1]; i++) add(`#${R.a.birth[2 * i + 1]} bred #${R.a.birth[2 * i]}`, R.a.birth[2 * i + 1]);
      for (let i = R.off.death[g]; i < R.off.death[g + 1]; i++) add(`#${R.a.death[2 * i]} ${CAUSE_TEXT[R.meta.causes[R.a.death[2 * i + 1]]]}`, R.a.death[2 * i]);
    }
  }
}

/* ---------- input ---------- */
function setZoom(z, sx, sy) {
  const s0 = pxPerCell();
  const wx = view.x + (sx - world.width / 2) / s0, wy = view.y + (sy - world.height / 2) / s0;
  view.z = Math.min(9, Math.max(1, z));
  if (view.z === 1) { view.x = 32; view.y = 32; return; }
  const s1 = pxPerCell();
  view.x = wx - (sx - world.width / 2) / s1;
  view.y = wy - (sy - world.height / 2) / s1;
}
function pickAt(sx, sy) {
  const f = Math.floor(t), s = pxPerCell();
  const wx = view.x + (sx - world.width / 2) / s, wy = view.y + (sy - world.height / 2) / s;
  let best = -1, bd = Math.max(1.6, 14 * dpr / s);
  for (const id of R.rows[f].keys()) {
    const p = place(id, f, t - f);
    const d = Math.hypot(wrapDelta((p.x - wx) / 64) * 64, wrapDelta((p.y - wy) / 64) * 64);
    if (d < bd) { bd = d; best = id; }
  }
  return best;
}
function setFollow(on) {
  follow = on && selected >= 0;
  $('follow').classList.toggle('on', follow);
  if (follow && view.z < 3) view.z = 3.2;
}
function fitView() { setFollow(false); view.z = 1; view.x = 32; view.y = 32; }
function setPlaying(on) { playing = on; $('play').textContent = on ? 'Pause' : 'Play'; }

function wire() {
  let drag = null;
  world.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, moved: false }; world.setPointerCapture(e.pointerId); });
  world.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (drag.moved || Math.hypot(dx, dy) > 4) {
      drag.moved = true;
      if (view.z > 1) { setFollow(false); view.x -= dx * dpr / pxPerCell(); view.y -= dy * dpr / pxPerCell(); }
      drag.x = e.clientX; drag.y = e.clientY;
    }
  });
  world.addEventListener('pointerup', (e) => {
    if (drag && !drag.moved) {
      const box = world.getBoundingClientRect();
      const id = pickAt((e.clientX - box.left) * dpr, (e.clientY - box.top) * dpr);
      if (id >= 0) select(id, false);
    }
    drag = null;
  });
  world.addEventListener('wheel', (e) => {
    e.preventDefault();
    const box = world.getBoundingClientRect();
    setZoom(view.z * Math.exp(-e.deltaY * 0.0015), (e.clientX - box.left) * dpr, (e.clientY - box.top) * dpr);
  }, { passive: false });
  const scrub = (e) => {
    const box = strip.getBoundingClientRect(), L = stripLayout();
    const x = (e.clientX - box.left) * dpr;
    if (x >= L.x0 - 8 * dpr) t = Math.min(R.F - 1, Math.max(0, (x - L.x0) / (L.x1 - L.x0) * (R.F - 1)));
    panelFrame = -1;
  };
  strip.addEventListener('pointerdown', (e) => { strip.setPointerCapture(e.pointerId); scrub(e); });
  strip.addEventListener('pointermove', (e) => { if (e.buttons) scrub(e); });
  $('play').onclick = () => setPlaying(!playing);
  $('fit').onclick = fitView;
  $('follow').onclick = () => setFollow(!follow);
  $('close').onclick = () => select(-1);
  $('pick-old').onclick = () => {
    const f = Math.floor(t);
    let best = -1, age = -1;
    for (const [id, k] of R.rows[f]) if (R.a.agent_age[k] > age) { age = R.a.agent_age[k]; best = id; }
    select(best, false); setFollow(true);
  };
  $('pick-kin').onclick = () => { select(mostKids(Math.floor(t)), false); setFollow(true); };
  for (const v of [2, 8, 24, 80]) {
    const b = document.createElement('button');
    b.textContent = `${v} steps/s`;
    b.classList.toggle('on', v === speed);
    b.onclick = () => { speed = v; for (const o of $('speeds').children) o.classList.toggle('on', o === b); };
    $('speeds').append(b);
  }
  R.meta.clade_roots.forEach((root, k) => {
    const li = document.createElement('li'), dot = document.createElement('i'), name = document.createElement('span'), n = document.createElement('span');
    dot.style.background = CLADE_COLOURS[k];
    name.textContent = `${CLADE_NAMES[k]} · from #${root}`;
    li.append(dot, name, n);
    li.onclick = () => { hiddenClade = hiddenClade === k ? -1 : k; [...$('clades').children].forEach((o, j) => o.classList.toggle('off', hiddenClade >= 0 && j !== hiddenClade)); };
    $('clades').append(li);
  });
  $('clade-note').textContent = `A family is everyone descended from one creature alive by step ${R.meta.clade_anchor_step.toLocaleString()}. Click to isolate.`;
  $('subtitle').textContent = `A recorded stretch of one evolving run, ${R.meta.start_step.toLocaleString()} steps after random founders. Nothing is scripted or respawned.`;
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); setPlaying(!playing); }
    else if (e.code === 'ArrowRight') { setPlaying(false); t = Math.min(R.F - 1, Math.floor(t) + 1); }
    else if (e.code === 'ArrowLeft') { setPlaying(false); t = Math.max(0, Math.ceil(t) - 1); }
    else if (e.key === 'f') setFollow(!follow);
    else if (e.key === '0') fitView();
    else if (e.code === 'Escape') select(-1);
  });
  window.addEventListener('resize', resize);
}
function mostKids(f) {
  let best = -1, n = -1;
  const now = R.meta.start_step + f;
  for (const id of R.rows[f].keys()) {
    const c = R.person.get(id).kids.filter((k) => R.person.get(k).born <= now).length;
    if (c > n) { n = c; best = id; }
  }
  return best;
}

/* ---------- main loop and frame-rate probe ---------- */
let lastTime = null;
const frameTimes = [], drawTimes = [];
function tick(now) {
  const dt = lastTime === null ? 0 : now - lastTime;
  lastTime = now;
  if (playing) {
    t += Math.min(dt, 100) / 1000 * speed;
    if (t >= R.F - 1) t = 0;
  }
  const start = performance.now();
  drawWorld(now);
  drawStrip();
  updatePanels(now);
  if (params.has('bench')) { frameTimes.push(dt); drawTimes.push(performance.now() - start); }
  requestAnimationFrame(tick);
}
function bench(url) {
  setTimeout(() => {
    frameTimes.length = 0; drawTimes.length = 0;
    setTimeout(() => {
      const sorted = [...frameTimes].sort((a, b) => a - b), draw = [...drawTimes].sort((a, b) => a - b);
      const gl = document.createElement('canvas').getContext('webgl');
      const info = gl && gl.getExtension('WEBGL_debug_renderer_info');
      fetch(url, { method: 'POST', mode: 'no-cors', body: JSON.stringify({
        fps: 1000 * sorted.length / sorted.reduce((a, b) => a + b, 0),
        p95_ms: sorted[Math.floor(0.95 * sorted.length)], p99_ms: sorted[Math.floor(0.99 * sorted.length)],
        worst_ms: sorted[sorted.length - 1], draw_p95_ms: draw[Math.floor(0.95 * draw.length)],
        canvas: [world.width, world.height], zoom: view.z, selected, alive: R.rows[Math.floor(t)].size,
        renderer: info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : 'unknown' }) });
    }, 12000);
  }, 5000);
}

load().then((replay) => {
  R = replay;
  $('loading').hidden = true;
  wire();
  resize();
  t = Math.min(R.F - 1, Math.max(0, Number(params.get('t') || 0)));
  const want = params.get('select');
  if (want) select(want === 'kin' ? mostKids(Math.floor(t)) : Number(want), false);
  if (params.has('zoom')) view.z = Number(params.get('zoom'));
  if (params.has('cx')) { view.x = Number(params.get('cx')); view.y = Number(params.get('cy')); }
  if (follow) {
    setFollow(true);
    if (params.has('zoom')) view.z = Number(params.get('zoom'));
    const p = place(selected, Math.floor(t), 0);
    if (p) { view.x = p.x; view.y = p.y; }
  }
  setPlaying(playing);
  window.replayApi = { seek(v) { t = v; panelFrame = -1; }, frames: R.F };
  requestAnimationFrame(tick);
  if (params.has('bench')) bench(params.get('bench'));
}).catch((error) => { $('loading').hidden = false; $('loading').textContent = error.message; console.error(error); });
