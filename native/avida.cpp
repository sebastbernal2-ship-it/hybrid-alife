// Native Avida-style VM update: runs each organism's merit-budgeted cycles.
// Mirrors hybrid_alife.agents.avida_vm.vm_cycle (organisms are independent
// within an update, so the cycle loop is organism-major). Copy mutations use
// a splitmix64 stream, so they match the JAX path in distribution, not bits.
#include <cstdint>
#include <vector>

namespace {
enum Op { NOP, INC, DEC, ADD, NAND, SHIFT_L, SHIFT_R, LOAD_ENV, EMIT, H_ALLOC,
          H_COPY, H_DIVIDE, JUMP, JUMP_IF_ZERO, SET_READ, SET_WRITE, N_OPS };

struct Rng {
  uint64_t s;
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

inline int32_t fmod_pos(int32_t a, int32_t m) {  // floor-mod, m > 0
  int32_t r = a % m;
  return r < 0 ? r + m : r;
}
inline int32_t wrap_add(int32_t a, int32_t b) { return (int32_t)((uint32_t)a + (uint32_t)b); }
}  // namespace

extern "C" {

// Opcode count, checked against the Python VM at load time so a stale build cannot silently
// disagree about the instruction set.
int avida_n_ops() { return N_OPS; }

// All arrays are C-contiguous. registers is [n, nreg] with nreg >= 2.
// rewards: 9 floats. div_mask (out): 1 where the organism executed H_DIVIDE.
void avida_run_update(int n, int max_len, int nreg, int32_t* genomes, const int32_t* lengths,
                      int32_t* registers, int32_t* ip, int32_t* read_head, int32_t* write_head,
                      int32_t* copied, int32_t* copied_length, float* merit, int32_t* tasks,
                      int32_t* in_a, int32_t* in_b, const uint8_t* alive, const int32_t* budget,
                      const int32_t* env_a, const int32_t* env_b, const float* rewards,
                      double point_mutation_prob, uint64_t seed, uint8_t* div_mask,
                      float* energy, float energy_cost, float energy_eff, double* totals) {
  Rng rng{seed};
  for (int o = 0; o < n; ++o) {
    div_mask[o] = 0;
    if (!alive[o]) continue;
    const int32_t len = lengths[o] > 1 ? lengths[o] : 1;
    int32_t* g = genomes + (int64_t)o * max_len;
    int32_t* cp = copied + (int64_t)o * max_len;
    int32_t* r = registers + (int64_t)o * nreg;
    int32_t ipo = ip[o], rh = read_head[o], wh = write_head[o], cl = copied_length[o];
    int32_t ta = tasks[o], ia = in_a[o], ib = in_b[o];
    float mer = merit[o];
    float en = energy ? energy[o] : 0.0f;
    double tot_in = 0.0, tot_heat = 0.0;
    for (int c = 0; c < budget[o]; ++c) {
      if (energy && !(en > 0.0f)) break;  // out of energy: stops executing
      const int32_t op = g[fmod_pos(ipo, len)];
      const int32_t r0 = r[0], r1 = r[1];
      int32_t n0 = r0, n1 = r1;
      switch (op) {
        case INC: n0 = wrap_add(r0, 1); break;
        case DEC: n0 = wrap_add(r0, -1); break;
        case ADD: n0 = wrap_add(r0, r1); break;
        case NAND: n0 = ~(r0 & r1); break;
        case SHIFT_L: n0 = (int32_t)((uint32_t)r0 << 1); break;
        case SHIFT_R: n0 = r0 >> 1; break;
        case LOAD_ENV: n0 = env_a[o]; n1 = env_b[o]; ia = n0; ib = n1; break;
        default: break;
      }
      if (op == SET_READ) rh = fmod_pos(r0, max_len);
      if (op == SET_WRITE) wh = fmod_pos(r0, max_len);
      if (op == H_ALLOC) { cl = 0; wh = 0; rh = 0; }
      if (op == H_COPY) {
        int32_t val = g[fmod_pos(rh, len)];
        if (point_mutation_prob > 0 && rng.uniform() < point_mutation_prob)
          val = (int32_t)(rng.next() % N_OPS);
        cp[fmod_pos(wh, max_len)] = val;
        rh = fmod_pos(rh + 1, max_len);
        wh = fmod_pos(wh + 1, max_len);
        cl = cl + 1 < max_len ? cl + 1 : max_len;
      }
      int32_t nip = ipo + 1;
      if (op == JUMP || (op == JUMP_IF_ZERO && r0 == 0)) nip = fmod_pos(r0, len);
      ipo = fmod_pos(nip, len);

      float gain = 0.0f;
      if (op == EMIT) {
        const int32_t expect[9] = {~ia, ~(ia & ib), ia & ib, ia | ~ib, ia | ib,
                                   ia & ~ib, ~(ia | ib), ia ^ ib, ~(ia ^ ib)};
        for (int t = 0; t < 9; ++t)
          if (expect[t] == n0 && !((ta >> t) & 1)) { gain += rewards[t]; ta |= (1 << t); }
        gain += 0.05f;
      }
      if (op == NAND) gain += 0.01f;
      mer += gain;
      if (energy) {
        const float intake = energy_eff * gain;
        const float avail = en + intake;
        const float paid = energy_cost < avail ? energy_cost : avail;
        en = avail - paid;
        tot_in += intake;
        tot_heat += paid;
      }
      r[0] = n0; r[1] = n1;
      if (op == H_DIVIDE) div_mask[o] = 1;
    }
    ip[o] = ipo; read_head[o] = rh; write_head[o] = wh; copied_length[o] = cl;
    tasks[o] = ta; in_a[o] = ia; in_b[o] = ib; merit[o] = mer;
    if (energy) { energy[o] = en; totals[0] += tot_in; totals[1] += tot_heat; }
  }
}


// Offspring placement at the update boundary; mirrors avida_vm.apply_h_divide.
// Parents (executed H_DIVIDE with copied_length >= min_copy) are matched in index
// order to dead slots in index order. JAX scatters with duplicate index 0 for the
// unmatched rows, so when matches < n the slot-0 child writes and slot-0 parent
// buffer reset are undone unless scatter_fix is enabled.
int avida_abi_version() { return 2; }

void avida_divide(int n, int max_len, int nreg, int mem, int min_len, int min_copy,
                  double ins_p, double del_p, double dup_p, uint64_t seed,
                  int32_t next_lineage_start, const uint8_t* div_mask, int scatter_fix,
                  int32_t* genomes,
                  int32_t* lengths, int32_t* registers, int32_t* memory, int32_t* ip,
                  int32_t* rh, int32_t* wh, int32_t* copied, int32_t* cl, float* merit,
                  int32_t* age, uint8_t* alive, int32_t* tasks, int32_t* lineage_id,
                  int32_t* parent_id, int32_t* depth, float* energy) {
  Rng rng{seed};
  std::vector<int> parents, dead;
  for (int i = 0; i < n; ++i) {
    if (div_mask[i] && cl[i] >= min_copy) parents.push_back(i);
    if (!alive[i]) dead.push_back(i);
  }
  const int m = (int)(parents.size() < dead.size() ? parents.size() : dead.size());
  if (m == 0) return;

  // Snapshot restores slot 0 only for legacy duplicate-index scatter behaviour.
  std::vector<int32_t> g0(genomes, genomes + max_len), r0(registers, registers + nreg),
      m0(memory, memory + mem), c0(copied, copied + max_len);
  const int32_t len0 = lengths[0], ip0 = ip[0], rh0 = rh[0], wh0 = wh[0], cl0 = cl[0],
                age0 = age[0], tasks0 = tasks[0], lid0 = lineage_id[0], pid0 = parent_id[0],
                dep0 = depth[0];
  const float merit0 = merit[0];
  const uint8_t alive0 = alive[0];

  std::vector<int32_t> tmp(max_len);
  for (int k = 0; k < m; ++k) {
    const int p = parents[k], ch = dead[k];
    const int32_t pl = cl[p];
    const int32_t* pc = copied + (int64_t)p * max_len;
    for (int j = 0; j < max_len; ++j) tmp[j] = pc[j];
    const bool ins = ins_p > 0 && rng.uniform() < ins_p;
    const bool del = del_p > 0 && rng.uniform() < del_p;
    const bool dup = dup_p > 0 && rng.uniform() < dup_p;
    int32_t dup_len = pl / 4;
    dup_len = dup_len < 1 ? 1 : (dup_len > max_len ? max_len : dup_len);
    if (dup) {
      const int32_t denom = pl > 1 ? pl : 1;
      const int32_t start = (int32_t)(rng.next() % (uint64_t)max_len) % denom;
      for (int j = pl; j < pl + dup_len && j < max_len; ++j) {
        int32_t off = j - pl;
        off = off > dup_len - 1 ? dup_len - 1 : off;
        tmp[j] = pc[(start + off) % denom];
      }
    }
    int32_t child_len = pl + (ins ? 1 : 0) - (del ? 1 : 0) + (dup ? dup_len : 0);
    child_len = child_len < min_len ? min_len : child_len;
    child_len = child_len > max_len ? max_len : child_len;

    for (int j = 0; j < max_len; ++j) {
      genomes[(int64_t)ch * max_len + j] = tmp[j];
      copied[(int64_t)ch * max_len + j] = 0;
    }
    lengths[ch] = child_len;
    for (int j = 0; j < nreg; ++j) registers[(int64_t)ch * nreg + j] = 0;
    for (int j = 0; j < mem; ++j) memory[(int64_t)ch * mem + j] = 0;
    ip[ch] = rh[ch] = wh[ch] = cl[ch] = age[ch] = tasks[ch] = 0;
    merit[ch] = 1.0f;
    alive[ch] = 1;
    lineage_id[ch] = wrap_add(next_lineage_start, k);  // int32 wrap, like JAX
    parent_id[ch] = lineage_id[p];
    depth[ch] = wrap_add(depth[p], 1);
    if (energy && (scatter_fix || !(ch == 0 && m < n))) {
      const float half = 0.5f * energy[p];
      energy[p] = energy[p] - half;
      energy[ch] = energy[ch] + half;
    }
  }
  if (!scatter_fix && m < n) {  // duplicate-index-0 writes from unmatched rows
    for (int j = 0; j < max_len; ++j) { genomes[j] = g0[j]; copied[j] = c0[j]; }
    for (int j = 0; j < nreg; ++j) registers[j] = r0[j];
    for (int j = 0; j < mem; ++j) memory[j] = m0[j];
    lengths[0] = len0; ip[0] = ip0; rh[0] = rh0; wh[0] = wh0; cl[0] = cl0; age[0] = age0;
    tasks[0] = tasks0; lineage_id[0] = lid0; parent_id[0] = pid0; depth[0] = dep0;
    merit[0] = merit0; alive[0] = alive0;
  }

  std::vector<int32_t> pc0(copied, copied + max_len);
  const int32_t pcl0 = cl[0];
  for (int k = 0; k < m; ++k) {
    const int p = parents[k];
    for (int j = 0; j < max_len; ++j) copied[(int64_t)p * max_len + j] = 0;
    cl[p] = 0;
  }
  if (!scatter_fix && m < n) {
    for (int j = 0; j < max_len; ++j) copied[j] = pc0[j];
    cl[0] = pcl0;
  }
}

// Metabolite uptake / concentration deposit; mirrors the coupling block of
// step_avida_population. gr/gc are precomputed grid cells (float32 JAX indexing).
void avida_couple(int n, int h, int w, int M, int C, const int32_t* gr, const int32_t* gc,
                  const uint8_t* alive, float uptake, float deposit, float* metab,
                  float* conc, float* merit) {
  (void)h;
  if (uptake > 0.0f) {
    std::vector<float> take((size_t)n * M);
    for (int i = 0; i < n; ++i) {
      const float a = alive[i] ? 1.0f : 0.0f;
      const float* av = metab + ((int64_t)gr[i] * w + gc[i]) * M;
      for (int c = 0; c < M; ++c) take[(size_t)i * M + c] = av[c] * uptake * a;
    }
    for (int i = 0; i < n; ++i) {
      float* cell = metab + ((int64_t)gr[i] * w + gc[i]) * M;
      for (int c = 0; c < M; ++c) cell[c] = cell[c] + (-take[(size_t)i * M + c]);
    }
    const int64_t total = (int64_t)h * w * M;
    for (int64_t j = 0; j < total; ++j) {
      float x = metab[j];
      x = x < 0.0f ? 0.0f : x;
      metab[j] = x > 1.0f ? 1.0f : x;
    }
    for (int i = 0; i < n; ++i) {
      float s = take[(size_t)i * M];
      for (int c = 1; c < M; ++c) s = s + take[(size_t)i * M + c];
      merit[i] = merit[i] + s * 5.0f;
    }
  }
  if (deposit > 0.0f) {
    for (int i = 0; i < n; ++i) {
      const float a = alive[i] ? 1.0f : 0.0f;
      const float d = (deposit * a) * a;
      float* cell = conc + ((int64_t)gr[i] * w + gc[i]) * C;
      for (int c = 0; c < C; ++c) cell[c] = cell[c] + d;
    }
  }
}

}  // extern "C"
