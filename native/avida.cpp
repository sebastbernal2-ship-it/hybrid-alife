// Native Avida-style VM update: runs each organism's merit-budgeted cycles.
// Mirrors hybrid_alife.agents.avida_vm.vm_cycle (organisms are independent
// within an update, so the cycle loop is organism-major). Copy mutations use
// a splitmix64 stream, so they match the JAX path in distribution, not bits.
#include <cstdint>

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

// All arrays are C-contiguous. registers is [n, nreg] with nreg >= 2.
// rewards: 9 floats. div_mask (out): 1 where the organism executed H_DIVIDE.
void avida_run_update(int n, int max_len, int nreg, int32_t* genomes, const int32_t* lengths,
                      int32_t* registers, int32_t* ip, int32_t* read_head, int32_t* write_head,
                      int32_t* copied, int32_t* copied_length, float* merit, int32_t* tasks,
                      int32_t* in_a, int32_t* in_b, const uint8_t* alive, const int32_t* budget,
                      const int32_t* env_a, const int32_t* env_b, const float* rewards,
                      double point_mutation_prob, uint64_t seed, uint8_t* div_mask) {
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
    for (int c = 0; c < budget[o]; ++c) {
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
      r[0] = n0; r[1] = n1;
      if (op == H_DIVIDE) div_mask[o] = 1;
    }
    ip[o] = ipo; read_head[o] = rh; write_head[o] = wh; copied_length[o] = cl;
    tasks[o] = ta; in_a[o] = ia; in_b[o] = ib; merit[o] = mer;
  }
}

}  // extern "C"
