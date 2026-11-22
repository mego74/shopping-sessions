// Shopping Sessions: native pipeline.
// Phases: small tables -> links (union-find) -> parse events+clicks (scatter)
//         -> dedup by event_id -> per-person sessionize/attribute -> parquet.
#if defined(__x86_64__)
#include <immintrin.h>
#else
#include <arm_neon.h>
#include <arm_acle.h>
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

static const i64 DAYMS = 86400000LL;
static const u32 GAP = 1800000u;
static const u32 WEEK = 604800000u;

static double T0;
static double nowd() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static bool VERBOSE = false;
static void logt(const char* what) {
  if (VERBOSE) fprintf(stderr, "[%8.2f ms] %s\n", (nowd() - T0) * 1000.0, what);
}
static void die(const char* msg) {
  fprintf(stderr, "FATAL: %s\n", msg);
  _exit(2);
}

// ---------------------------------------------------------------- memory
static void* big_alloc(size_t n) {
  if (n == 0) n = 64;
  n = (n + 4095) & ~(size_t)4095;
  void* p = mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) die("mmap anon");
#ifdef MADV_HUGEPAGE
  if (n >= (4u << 20)) madvise(p, n, MADV_HUGEPAGE);
#endif
  return p;
}

template <class T>
struct Buf {
  T* p = nullptr;
  size_t cap = 0;
  void ensure(size_t n) {
    if (n <= cap) return;
    size_t nc = std::max(n, cap * 2 + 64);
    T* q = (T*)malloc(nc * sizeof(T));
    if (!q) die("malloc");
    if (p) {
      memcpy(q, p, cap * sizeof(T));
      free(p);
    }
    p = q;
    cap = nc;
  }
  T& operator[](size_t i) { return p[i]; }
};

// ---------------------------------------------------------------- files
struct MFile {
  const char* p;
  u64 n;
};

static MFile map_file(const std::string& path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) die(("open " + path).c_str());
  struct stat st;
  if (fstat(fd, &st) != 0) die("fstat");
  u64 n = (u64)st.st_size;
  u64 mapn = (n + 4095) & ~(u64)4095;
  u64 total = mapn + 3 * 4096;
  char* base = (char*)mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (base == MAP_FAILED) die("mmap region");
  if (n > 0) {
    void* r = mmap(base, n, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd, 0);
    if (r == MAP_FAILED) die("mmap file");
  }
  close(fd);
  if (n == 0 || base[n - 1] != '\n') {
    if (n % 4096 != 0) {
      if (mprotect(base + (n & ~(u64)4095), 4096, PROT_READ | PROT_WRITE) != 0) die("mprotect");
    }
    base[n] = '\n';
    n++;
  }
  return {base, n};
}

static std::vector<std::string> list_csv(const std::string& dir) {
  std::vector<std::string> out;
  DIR* d = opendir(dir.c_str());
  if (!d) return out;
  while (dirent* e = readdir(d)) {
    std::string nm = e->d_name;
    if (nm.size() < 4 || nm.compare(nm.size() - 4, 4, ".csv") != 0) continue;
    std::string full = dir + "/" + nm;
    struct stat st;
    if (stat(full.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) continue;
    out.push_back(full);
  }
  closedir(d);
  std::sort(out.begin(), out.end());
  return out;
}

static std::string read_small(const std::string& path) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) die(("open " + path).c_str());
  std::string s;
  char buf[65536];
  size_t k;
  while ((k = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, k);
  fclose(f);
  return s;
}

// ---------------------------------------------------------------- hashing
static inline u64 load64(const char* p) {
  u64 v;
  memcpy(&v, p, 8);
  return v;
}
static inline u64 loadp(const char* p, u32 n) {  // 0 <= n <= 8, may over-read
  if (n == 0) return 0;
  u64 v = load64(p);
  return n >= 8 ? v : (v & ((1ULL << (8 * n)) - 1));
}
static inline u64 mum(u64 a, u64 b) {
  __uint128_t r = (__uint128_t)a * b;
  return (u64)r ^ (u64)(r >> 64);
}
static inline u64 fmix(u64 h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;
  return h;
}
static inline u64 hstr(const char* p, u32 n) {
  u64 seed = 0xa0761d6478bd642fULL ^ ((u64)n * 0xe7037ed1a0b428dbULL);
  while (n > 16) {
    seed = mum(load64(p) ^ 0x8ebc6af09c88c6e3ULL, load64(p + 8) ^ seed);
    p += 16;
    n -= 16;
  }
  u64 a, b;
  if (n > 8) {
    a = load64(p);
    b = loadp(p + 8, n - 8);
  } else {
    a = loadp(p, n);
    b = 0;
  }
  u64 h = fmix(mum(a ^ 0x589965cc75374cc3ULL, b ^ seed ^ 0x1d8e4e27c47d124fULL));
  return h ? h : 1;
}
static inline u32 crcstr(const char* p, u32 n) {
  u64 c = n;
  while (n >= 8) {
#if defined(__x86_64__)
    c = _mm_crc32_u64(c, load64(p));
#else
    c = __crc32cd((u32)c, load64(p));
#endif
    p += 8;
    n -= 8;
  }
#if defined(__x86_64__)
  if (n) c = _mm_crc32_u64(c, loadp(p, n));
#else
  if (n) c = __crc32cd((u32)c, loadp(p, n));
#endif
  return (u32)c;
}
static inline u64 encp(const char* p, u32 n) { return ((u64)(uintptr_t)p << 16) | (u64)(n > 65535 ? 65535 : n); }
static inline const char* decp(u64 v) { return (const char*)(uintptr_t)(v >> 16); }
static inline u32 decn(u64 v) { return (u32)(v & 0xFFFF); }
static inline int cmpstr(u64 a, u64 b) {
  u32 na = decn(a), nb = decn(b);
  int c = memcmp(decp(a), decp(b), na < nb ? na : nb);
  if (c) return c;
  return na < nb ? -1 : (na > nb ? 1 : 0);
}

// ---------------------------------------------------------------- CSV scanning
// sep = ',' or '\n' positions, quo = '"' positions, for 64 bytes at p (64-byte aligned)
static inline void block_masks(const char* p, u64& sep, u64& quo) {
#if defined(__AVX2__)
  __m256i a = _mm256_load_si256((const __m256i*)p);
  __m256i b = _mm256_load_si256((const __m256i*)(p + 32));
  const __m256i C = _mm256_set1_epi8(','), N = _mm256_set1_epi8('\n'), Q = _mm256_set1_epi8('"');
  u64 sa = (u32)_mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi8(a, C), _mm256_cmpeq_epi8(a, N)));
  u64 sb = (u32)_mm256_movemask_epi8(_mm256_or_si256(_mm256_cmpeq_epi8(b, C), _mm256_cmpeq_epi8(b, N)));
  u64 qa = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(a, Q));
  u64 qb = (u32)_mm256_movemask_epi8(_mm256_cmpeq_epi8(b, Q));
  sep = sa | (sb << 32);
  quo = qa | (qb << 32);
#elif defined(__aarch64__)
  const uint8x16_t C = vdupq_n_u8(','), N = vdupq_n_u8('\n'), Q = vdupq_n_u8('"');
  static const uint8_t bm[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
  const uint8x16_t B = vld1q_u8(bm);
  uint8x16_t ts[4], tq[4];
  for (int i = 0; i < 4; i++) {
    uint8x16_t a = vld1q_u8((const uint8_t*)p + 16 * i);
    ts[i] = vandq_u8(vorrq_u8(vceqq_u8(a, C), vceqq_u8(a, N)), B);
    tq[i] = vandq_u8(vceqq_u8(a, Q), B);
  }
  uint8x16_t s0 = vpaddq_u8(ts[0], ts[1]), s1 = vpaddq_u8(ts[2], ts[3]);
  uint8x16_t q0 = vpaddq_u8(tq[0], tq[1]), q1 = vpaddq_u8(tq[2], tq[3]);
  s0 = vpaddq_u8(s0, s1);
  q0 = vpaddq_u8(q0, q1);
  uint8x16_t r = vpaddq_u8(s0, q0);
  sep = vgetq_lane_u64(vreinterpretq_u64_u8(r), 0);
  quo = vgetq_lane_u64(vreinterpretq_u64_u8(r), 1);
#else
  const __m128i C = _mm_set1_epi8(','), N = _mm_set1_epi8('\n'), Q = _mm_set1_epi8('"');
  sep = 0;
  quo = 0;
  for (int i = 0; i < 4; i++) {
    __m128i a = _mm_load_si128((const __m128i*)(p + 16 * i));
    sep |= (u64)(u32)_mm_movemask_epi8(_mm_or_si128(_mm_cmpeq_epi8(a, C), _mm_cmpeq_epi8(a, N))) << (16 * i);
    quo |= (u64)(u32)_mm_movemask_epi8(_mm_cmpeq_epi8(a, Q)) << (16 * i);
  }
#endif
}

struct Scanner {
  const char* base;
  u64 blk;   // current block offset
  u64 sm;    // remaining separator bits of current block
  u64 qm;    // all quote bits of current block
  u64 cumq;  // quotes in blocks before the current one
  inline void init(const char* b, u64 pos) {
    base = b;
    blk = pos & ~(u64)63;
    u64 s, q;
    block_masks(base + blk, s, q);
    sm = s & (~0ULL << (pos & 63));
    qm = q;
    cumq = 0;
  }
  inline u64 next() {
    while (UNLIKELY(sm == 0)) {
      cumq += (u64)__builtin_popcountll(qm);
      blk += 64;
      block_masks(base + blk, sm, qm);
    }
    u64 p = blk + (u64)__builtin_ctzll(sm);
    sm &= sm - 1;
    return p;
  }
  // number of quotes strictly before pos; pos must be < blk + 128 and >= blk
  inline u64 qbefore(u64 pos) const {
    u64 k = pos - blk;
    u64 m = k >= 64 ? ~0ULL : ((1ULL << k) - 1);
    return cumq + (u64)__builtin_popcountll(qm & m);
  }
};

struct Fld {
  const char* p;
  u32 n;
};

// Parses one line starting at ls. Returns true iff well-formed with exactly NF fields.
// Fields split on every ','; a field is quoted iff its first byte is '"', then its last byte must be
// '"' too (length >= 2), and the line's quote count must equal 2 x (number of quoted fields).
template <int NF>
static inline bool parse_line(Scanner& sc, const char* base, Fld* f, u64 ls, u64& next_ls) {
  const u64 q0 = sc.qbefore(ls);
  u64 fs = ls;
  u32 expq = 0;
  u32 bad = 0;
  for (int i = 0;; i++) {
    u64 p = sc.next();
    char c = base[p];
    if (i < NF - 1) {
      if (UNLIKELY(c == '\n')) {
        next_ls = p + 1;
        return false;
      }
    } else if (UNLIKELY(c != '\n')) {
      do p = sc.next();
      while (base[p] != '\n');
      next_ls = p + 1;
      return false;
    }
    u32 len = (u32)(p - fs);
    u32 isq = base[fs] == '"';
    bad |= isq & ((len < 2) | (base[p - 1] != '"'));
    f[i].p = base + fs + isq;
    f[i].n = len - 2 * isq;
    expq += 2 * isq;
    fs = p + 1;
    if (i == NF - 1) {
      next_ls = p + 1;
      return !bad && (sc.qbefore(p) - q0) == expq;
    }
  }
}

static inline u64 align_start(const MFile& f, u64 s) {
  if (s == 0) {
    const char* nl = (const char*)memchr(f.p, '\n', f.n);
    return (u64)(nl - f.p) + 1;
  }
  if (f.p[s - 1] == '\n') return s;
  const char* nl = (const char*)memchr(f.p + s, '\n', f.n - s);
  return (u64)(nl - f.p) + 1;
}

// ---------------------------------------------------------------- time parsing
static const int DIM[13] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
static inline bool is_leap(int y) { return (y % 4 == 0 && y % 100 != 0) || y % 400 == 0; }
static inline i64 days_from_civil(i64 y, unsigned m, unsigned d) {
  y -= m <= 2;
  const i64 era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = (unsigned)(y - era * 400);
  const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + (i64)doe - 719468;
}

// month-start day numbers for years [YBASE, YBASE+YSPAN): days since epoch, and days in month
static const int YBASE = 1900, YSPAN = 400;
static i32 MSTART[YSPAN * 13];
static u8 MDIM[YSPAN * 13];
static void init_cal() {
  for (int y = 0; y < YSPAN; y++)
    for (int m = 1; m <= 12; m++) {
      int yy = YBASE + y;
      MSTART[y * 13 + m] = (i32)days_from_civil(yy, (unsigned)m, 1);
      MDIM[y * 13 + m] = (u8)(DIM[m] + (m == 2 && is_leap(yy)));
    }
}

// "YYYY-MM-DDTHH:MM:SS" at s -> seconds since epoch (validated). Reads s[0..24).
static inline bool dt19(const char* s, i64& secs) {
  const u64 a = load64(s), b = load64(s + 8), c = load64(s + 16);
  const u64 SA = 0xFF0000FF00000000ULL, VA = ((u64)'-' << 32) | ((u64)'-' << 56);
  const u64 SB = 0x0000FF0000FF0000ULL, VB = ((u64)'T' << 16) | ((u64)':' << 40);
  if (((a & SA) ^ VA) | ((b & SB) ^ VB) | ((c & 0xFF) ^ (u64)':')) return false;
  const u64 Z = 0x3030303030303030ULL, SIX = 0x0606060606060606ULL, HI = 0xF0F0F0F0F0F0F0F0ULL;
  const u64 DA = 0x00FFFF00FFFFFFFFULL, DB = 0xFFFF00FFFF00FFFFULL, DC = 0x0000000000FFFF00ULL;
  const u64 ta = (a ^ Z) & DA, tb = (b ^ Z) & DB, tc = (c ^ Z) & DC;
  if (((ta | (ta + SIX)) & HI & DA) | ((tb | (tb + SIX)) & HI & DB) | ((tc | (tc + SIX)) & HI & DC)) return false;
  const unsigned y = (unsigned)((ta & 0xFF) * 1000 + ((ta >> 8) & 0xFF) * 100 + ((ta >> 16) & 0xFF) * 10 + ((ta >> 24) & 0xFF));
  const unsigned mo = (unsigned)(((ta >> 40) & 0xFF) * 10 + ((ta >> 48) & 0xFF));
  const unsigned dd = (unsigned)((tb & 0xFF) * 10 + ((tb >> 8) & 0xFF));
  const unsigned hh = (unsigned)(((tb >> 24) & 0xFF) * 10 + ((tb >> 32) & 0xFF));
  const unsigned mi = (unsigned)(((tb >> 48) & 0xFF) * 10 + (tb >> 56));
  const unsigned ss = (unsigned)(((tc >> 8) & 0xFF) * 10 + ((tc >> 16) & 0xFF));
  if ((mo - 1u > 11u) | (dd == 0) | (hh > 23) | (mi > 59) | (ss > 59)) return false;
  i64 days;
  unsigned yi = y - (unsigned)YBASE;
  if (LIKELY(yi < (unsigned)YSPAN)) {
    if (dd > MDIM[yi * 13 + mo]) return false;
    days = MSTART[yi * 13 + mo] + (i64)dd - 1;
  } else {
    unsigned dim = (unsigned)DIM[mo] + (mo == 2 && is_leap((int)y));
    if (dd > dim) return false;
    days = days_from_civil((i64)y, mo, dd);
  }
  secs = days * 86400 + (i64)(hh * 3600 + mi * 60 + ss);
  return true;
}

// F1 (n==20) or F2 (n==25). Returns ms.
static inline bool parse_f12(const char* s, u32 n, i64& ms) {
  i64 secs;
  if (n == 20) {
    if (s[19] != 'Z' || !dt19(s, secs)) return false;
    ms = secs * 1000;
    return true;
  }
  if (n != 25) return false;
  const unsigned char* u = (const unsigned char*)s;
  unsigned char sg = u[19];
  if ((sg != '+' && sg != '-') || u[22] != ':') return false;
  unsigned a = u[20] - 48u, b = u[21] - 48u, c = u[23] - 48u, e = u[24] - 48u;
  if ((a > 9) | (b > 9) | (c > 9) | (e > 9)) return false;
  unsigned oh = a * 10 + b, om = c * 10 + e;
  if (oh > 23 || om > 59) return false;
  if (!dt19(s, secs)) return false;
  i64 off = (i64)(oh * 3600 + om * 60);
  if (sg == '-') off = -off;
  ms = (secs - off) * 1000;
  return true;
}

static inline bool all_digits8(u64 v) {
  const u64 t = v ^ 0x3030303030303030ULL;
  return ((t | (t + 0x0606060606060606ULL)) & 0xF0F0F0F0F0F0F0F0ULL) == 0;
}
static inline u64 swar8(u64 v) {  // 8 ASCII digits (validated), first char in lowest byte
  v -= 0x3030303030303030ULL;
  v = (v * 10) + (v >> 8);
  v = (((v & 0x000000FF000000FFULL) * (100 + (1000000ULL << 32))) +
       (((v >> 16) & 0x000000FF000000FFULL) * (1 + (10000ULL << 32)))) >>
      32;
  return v;
}

static inline bool parse_ts(const char* s, u32 n, i64& ms) {
  if (n == 0) return false;
  if (n >= 5 && s[4] == '-') return parse_f12(s, n, ms);
  if (LIKELY(n == 13)) {
    u64 a = load64(s), b = load64(s + 5);  // bytes 0..7 and 5..12
    if (all_digits8(a) & all_digits8(b)) {
      ms = (i64)(swar8(a) * 100000 + (swar8(b) % 100000));
      return true;
    }
    return false;
  }
  u32 i = 0;
  while (i < n && s[i] == '0') i++;
  u64 v = 0;
  u32 sig = n - i;
  for (u32 k = i; k < n; k++) {
    unsigned d = (unsigned char)s[k] - 48u;
    if (d > 9) return false;
    v = v * 10 + d;
  }
  if (sig > 18) v = (u64)INT64_MAX;
  ms = (i64)v;
  return true;
}

static inline bool parse_price(const char* s, u32 n, u32& v) {
  if (n == 0) return false;
  u32 i = 0;
  while (i < n && s[i] == '0') i++;
  if (n - i > 8) return false;
  u32 x = 0;
  for (; i < n; i++) {
    unsigned d = (unsigned char)s[i] - 48u;
    if (d > 9) return false;
    x = x * 10 + d;
  }
  if (x < 1 || x > 10000000) return false;
  v = x;
  return true;
}

// ---------------------------------------------------------------- small exact string map
struct SMap {
  std::vector<u64> key;
  std::vector<i32> val;
  std::vector<std::string> str;
  u64 mask = 0;
  void init(size_t n) {
    size_t c = 16;
    while (c < n * 3) c <<= 1;
    key.assign(c, 0);
    val.assign(c, -1);
    mask = c - 1;
  }
  inline i32 find(const char* p, u32 n) const {
    u64 h = hstr(p, n);
    u64 i = h & mask;
    for (;;) {
      i32 v = val[i];
      if (v < 0) return -1;
      if (key[i] == h && str[v].size() == n && memcmp(str[v].data(), p, n) == 0) return v;
      i = (i + 1) & mask;
    }
  }
  i32 insert(const char* p, u32 n) {
    i32 f = find(p, n);
    if (f >= 0) return f;
    u64 h = hstr(p, n);
    u64 i = h & mask;
    while (val[i] >= 0) i = (i + 1) & mask;
    key[i] = h;
    val[i] = (i32)str.size();
    str.emplace_back(p, n);
    return val[i];
  }
};

// ---------------------------------------------------------------- globals
static int NT = 8;
static i64 WS, WE, CWS;      // event window, click window start (ms)
static i64 DAY0;             // epoch day of WS
static u64 WSOFF;            // WS - DAY0*DAYMS
static int ND;               // number of days
static int PBITS = 8, QBITS = 8;
static u32 TSHIFT = 29;
static const u32 INVALID = 0xFFFFFFFFu;
static const u64 DEAD = 1;

static SMap PRODS;                // product -> index
static std::vector<u8> PRODCAT;   // product index -> cat index
static std::vector<std::string> CATS;
static int UNKCAT = 0;
static SMap CAMPS;
static std::vector<std::string> CHANNELS;
static int NC = 0;  // number of campaigns; organic = NC

// ID table (links)
struct IdSlot {
  std::atomic<u64> key;
  u64 pkey;
};
static IdSlot* IDS = nullptr;
static u64* IDP = nullptr;            // string ptr per slot
static std::atomic<u32>* PAR = nullptr;
static std::atomic<u32>* BEST = nullptr;
static u64 IDCAP = 16;
static inline u64 id_home(u64 h) { return (u64)(((__uint128_t)h * IDCAP) >> 64); }

// ---------------------------------------------------------------- records & arenas
struct PRec {
  u64 pk;
  u64 packed;  // ts_rel<<32 | X<<24 | price   X=0xFF non-purchase else cat
  u64 np;      // name ptr (0 = linked, look up), DEAD = deduped away
};
struct ERec {
  u64 key;
  u32 ing;
  u32 gidx;
};
struct CRec {
  u64 pk;
  u64 key;  // ts_rel(click base)<<32 | campaign
  u64 idp;
};
struct LRec {
  u64 a, b, pa, pb;
};

static const u32 BLK = 512;
static const u32 CHB = 4096;  // blocks per chunk

template <class T>
struct Arena {
  T* chunks[2048];
  u32 nblocks = 0;
  Arena() { memset(chunks, 0, sizeof chunks); }
  inline u32 alloc() {
    u32 b = nblocks++;
    u32 c = b / CHB;
    if (c >= 2048) die("arena overflow");
    if (!chunks[c]) chunks[c] = (T*)big_alloc(sizeof(T) * (size_t)BLK * CHB);
    return b;
  }
  inline T* blk(u32 b) const { return chunks[b / CHB] + (size_t)(b % CHB) * BLK; }
  inline T* at(u32 idx) const { return chunks[idx / (BLK * CHB)] + idx % (BLK * CHB); }
};

template <class T>
struct Strm {
  T* wp = nullptr;
  T* we = nullptr;
  T* bp = nullptr;
  u32 cur = 0;
  std::vector<u32> blocks;
  inline u32 push(const T& v, Arena<T>& a) {
    if (UNLIKELY(wp == we)) {
      cur = a.alloc();
      blocks.push_back(cur);
      bp = wp = a.blk(cur);
      we = wp + BLK;
    }
    u32 idx = cur * BLK + (u32)(wp - bp);
    *wp++ = v;
    return idx;
  }
  size_t count() const { return blocks.empty() ? 0 : (blocks.size() - 1) * (size_t)BLK + (size_t)(wp - bp); }
  inline u32 bcount(size_t i) const { return i + 1 == blocks.size() ? (u32)(wp - bp) : BLK; }
};

struct Pers {
  u64 pk;
  u64 np;
  u32 cnt, off, ccnt, coff;
};
struct CK {
  u64 key;
  u64 idp;
};
struct CampState {
  u32 stamp, a, b, pad;
};

struct TCtx {
  int id = 0;
  Arena<PRec> pa;
  Arena<ERec> ea;
  Arena<CRec> ca;
  std::vector<Strm<PRec>> ps;
  std::vector<Strm<ERec>> es;
  std::vector<Strm<CRec>> cs;
  std::vector<LRec> lrecs;
  // counters
  u64 n_ev_lines = 0, n_ev_valid = 0, n_ck_valid = 0;
  // dedup scratch
  Buf<ERec> dtab;
  // person scratch
  Buf<u64> ptk;
  Buf<u32> ptv;
  std::vector<Pers> persons;
  Buf<u32> lp_of;
  Buf<u64> tmp64;
  Buf<u64> ev;
  Buf<u64> radtmp;
  Buf<u32> clp;
  Buf<CK> cktmp;
  Buf<CK> ck;
  Buf<u32> s_start;
  Buf<u16> s_day;
  Buf<u64> s_rev;
  Buf<u32> s_nds;
  std::vector<u64> catrev;
  std::vector<u32> cattouch;
  std::vector<CampState> cst;
  std::vector<u32> ctouch;
  u32 stamp = 0;
  // outputs
  std::vector<char> names;
  std::vector<u64> o_name;  // (offset into names)<<16 | len
  std::vector<i32> o_day, o_sess, o_ev;
  std::vector<i64> o_rev;
  std::vector<i16> o_cat;
  std::vector<u32> c_sess, c_purch, c_assist;
  std::vector<u64> c_rev;
};
static TCtx* TC = nullptr;

template <class F>
static void par_run(F&& f) {
  std::vector<std::thread> th;
  th.reserve(NT);
  for (int t = 1; t < NT; t++) th.emplace_back([&f, t] { f(t); });
  f(0);
  for (auto& x : th) x.join();
}

struct Chunk {
  int file;
  u64 s, e;
};

static std::vector<Chunk> make_chunks(const std::vector<MFile>& fs, u64 target) {
  std::vector<Chunk> out;
  for (size_t i = 0; i < fs.size(); i++) {
    u64 n = fs[i].n;
    u64 k = (n + target - 1) / target;
    if (k == 0) k = 1;
    u64 step = (n + k - 1) / k;
    for (u64 s = 0; s < n; s += step) out.push_back({(int)i, s, std::min(n, s + step)});
  }
  return out;
}

// ---------------------------------------------------------------- links
static inline u32 uf_find(u32 x) {
  for (;;) {
    u32 p = PAR[x].load(std::memory_order_relaxed);
    if (p == x) return x;
    u32 gp = PAR[p].load(std::memory_order_relaxed);
    if (p != gp) PAR[x].compare_exchange_weak(p, gp, std::memory_order_relaxed);
    x = gp;
  }
}
static inline void uf_unite(u32 a, u32 b) {
  for (;;) {
    a = uf_find(a);
    b = uf_find(b);
    if (a == b) return;
    if (a < b) std::swap(a, b);
    u32 exp = a;
    if (PAR[a].compare_exchange_strong(exp, b, std::memory_order_relaxed)) return;
  }
}
static inline u32 id_insert(u64 h, u64 ptr) {
  u64 i = id_home(h);
  for (;;) {
    u64 k = IDS[i].key.load(std::memory_order_relaxed);
    if (k == h) return (u32)i;
    if (k == 0) {
      u64 exp = 0;
      if (IDS[i].key.compare_exchange_strong(exp, h, std::memory_order_relaxed)) {
        IDP[i] = ptr;
        PAR[i].store((u32)i, std::memory_order_relaxed);
        BEST[i].store(INVALID, std::memory_order_relaxed);
        return (u32)i;
      }
      if (exp == h) return (u32)i;
    }
    if (++i == IDCAP) i = 0;
  }
}

static void links_phase(const std::vector<MFile>& lf) {
  std::vector<Chunk> ch = make_chunks(lf, 1 << 20);
  std::atomic<size_t> next{0};
  par_run([&](int t) {
    TCtx& W = TC[t];
    Fld fl[2];
    for (;;) {
      size_t ci = next.fetch_add(1);
      if (ci >= ch.size()) break;
      const MFile& f = lf[ch[ci].file];
      u64 ls = align_start(f, ch[ci].s), e = ch[ci].e;
      if (ls >= e) continue;
      Scanner sc;
      sc.init(f.p, ls);
      while (ls < e) {
        u64 nx;
        bool ok = parse_line<2>(sc, f.p, fl, ls, nx);
        ls = nx;
        if (!ok || fl[0].n == 0 || fl[1].n == 0) continue;
        W.lrecs.push_back({hstr(fl[0].p, fl[0].n), hstr(fl[1].p, fl[1].n), encp(fl[0].p, fl[0].n),
                           encp(fl[1].p, fl[1].n)});
      }
    }
  });
  u64 nl = 0;
  for (int t = 0; t < NT; t++) nl += TC[t].lrecs.size();
  IDCAP = std::max<u64>(64, nl * 3 + 64);
  IDS = (IdSlot*)big_alloc(IDCAP * sizeof(IdSlot));
  IDP = (u64*)big_alloc(IDCAP * sizeof(u64));
  PAR = (std::atomic<u32>*)big_alloc(IDCAP * sizeof(u32));
  BEST = (std::atomic<u32>*)big_alloc(IDCAP * sizeof(u32));
  logt("links parsed");
  par_run([&](int t) {
    for (LRec& r : TC[t].lrecs) {
      u32 sa = id_insert(r.a, r.pa);
      u32 sb = id_insert(r.b, r.pb);
      r.a = sa;
      r.b = sb;
    }
  });
  par_run([&](int t) {
    for (LRec& r : TC[t].lrecs)
      if (r.a != r.b) uf_unite((u32)r.a, (u32)r.b);
  });
  auto range = [&](int t, u64& lo, u64& hi) {
    lo = IDCAP * (u64)t / NT;
    hi = IDCAP * (u64)(t + 1) / NT;
  };
  par_run([&](int t) {
    u64 lo, hi;
    range(t, lo, hi);
    for (u64 i = lo; i < hi; i++) {
      if (IDS[i].key.load(std::memory_order_relaxed) == 0) continue;
      u32 r = uf_find((u32)i);
      u32 cur = BEST[r].load(std::memory_order_relaxed);
      for (;;) {
        if (cur != INVALID && cmpstr(IDP[i], IDP[cur]) >= 0) break;
        if (BEST[r].compare_exchange_weak(cur, (u32)i, std::memory_order_relaxed)) break;
      }
    }
  });
  par_run([&](int t) {
    u64 lo, hi;
    range(t, lo, hi);
    for (u64 i = lo; i < hi; i++) {
      if (IDS[i].key.load(std::memory_order_relaxed) == 0) continue;
      u32 r = uf_find((u32)i);
      u32 b = BEST[r].load(std::memory_order_relaxed);
      IDS[i].pkey = IDS[b].key.load(std::memory_order_relaxed);
    }
  });
  logt("links done");
}

// ---------------------------------------------------------------- events & clicks parse
struct PendE {
  u64 huid, np, packed, ekey;
  u32 ing, q;
};
struct PendC {
  u64 huid, key, idp;
};

static inline void lookup_person(u64 h, u64& pk, bool& linked) {
  u64 j = id_home(h);
  for (;;) {
    u64 k = IDS[j].key.load(std::memory_order_relaxed);
    if (k == h) {
      pk = IDS[j].pkey;
      linked = true;
      return;
    }
    if (k == 0) {
      pk = h;
      linked = false;
      return;
    }
    if (++j == IDCAP) j = 0;
  }
}

static inline void flushE(TCtx& W, PendE* pe, int n) {
  const u32 tid = (u32)W.id << TSHIFT;
  const u32 lim = 1u << TSHIFT;
  for (int i = 0; i < n; i++) {
    u64 pk;
    bool linked;
    lookup_person(pe[i].huid, pk, linked);
    u64 np = (linked && pk != pe[i].huid) ? 0 : pe[i].np;
    u32 p = (u32)(pk >> (64 - PBITS));
    u32 idx = W.ps[p].push({pk, pe[i].packed, np}, W.pa);
    if (UNLIKELY(idx >= lim - 1)) die("too many records per thread");
    W.es[pe[i].q].push({pe[i].ekey, pe[i].ing, tid | idx}, W.ea);
  }
}
static inline void flushC(TCtx& W, PendC* pc, int n) {
  for (int i = 0; i < n; i++) {
    u64 pk;
    bool linked;
    lookup_person(pc[i].huid, pk, linked);
    u32 p = (u32)(pk >> (64 - PBITS));
    W.cs[p].push({pk, pc[i].key, pc[i].idp}, W.ca);
  }
}

static void events_chunk(TCtx& W, const MFile& f, u64 s, u64 e) {
  u64 ls = align_start(f, s);
  if (ls >= e) return;
  Scanner sc;
  sc.init(f.p, ls);
  Fld fl[7];
  PendE pend[16];
  int npd = 0;
  const u32 qmask = (1u << QBITS) - 1;
  while (ls < e) {
    u64 nx;
    bool ok = parse_line<7>(sc, f.p, fl, ls, nx);
    ls = nx;
    if (!ok || fl[0].n == 0) continue;
    W.n_ev_lines++;
#if STAGE == 1
    W.n_ev_valid += fl[2].n + fl[1].n; continue;
#endif
    u64 h1 = hstr(fl[0].p, fl[0].n);
    u32 crc = crcstr(fl[0].p, fl[0].n);
#if STAGE == 2
    W.n_ev_valid += h1 + crc + hstr(fl[1].p, fl[1].n); continue;
#endif
    u32 q = (u32)(h1 >> (64 - QBITS));
    u64 ekey = (h1 << QBITS) | (crc & qmask);
    if (ekey == 0) ekey = 1;
    i64 ingms;
    u32 ing = INVALID;
    if (parse_f12(fl[3].p, fl[3].n, ingms)) ing = (u32)(ingms / 1000);
    // validate
    bool valid = fl[1].n > 0 && fl[5].n > 0;
    i64 ts = 0;
    u32 X = 0xFF, price = 0;
    if (valid) valid = parse_ts(fl[2].p, fl[2].n, ts) && ts >= WS && ts < WE;
    if (valid) {
      const char* tp = fl[4].p;
      u32 tn = fl[4].n;
      if (tn == 4) {
        u32 w;
        memcpy(&w, tp, 4);
        valid = (w == 0x77656976u /*view*/) || (w == 0x74726163u /*cart*/);
      } else if (tn == 8 && load64(tp) == 0x6573616863727570ULL /*purchase*/) {
        valid = parse_price(fl[6].p, fl[6].n, price);
        if (valid) {
          i32 pi = PRODS.find(fl[5].p, fl[5].n);
          X = pi >= 0 ? PRODCAT[pi] : (u32)UNKCAT;
        }
      } else
        valid = false;
    }
    if (valid && (ing == INVALID || (i64)ing * 1000 - ts > DAYMS)) valid = false;
#if STAGE == 3
    W.n_ev_valid += valid + h1 + crc + ts + hstr(fl[1].p, fl[1].n); continue;
#endif
    if (!valid) {
      W.es[q].push({ekey, ing, INVALID}, W.ea);
      continue;
    }
    W.n_ev_valid++;
    PendE& pd = pend[npd++];
    pd.huid = hstr(fl[1].p, fl[1].n);
    __builtin_prefetch(&IDS[id_home(pd.huid)]);
    pd.np = encp(fl[1].p, fl[1].n);
    pd.packed = ((u64)(u32)(ts - WS) << 32) | ((u64)X << 24) | price;
    pd.ekey = ekey;
    pd.ing = ing;
    pd.q = q;
    if (npd == 16) {
      flushE(W, pend, npd);
      npd = 0;
    }
  }
  flushE(W, pend, npd);
}

static void clicks_chunk(TCtx& W, const MFile& f, u64 s, u64 e) {
  u64 ls = align_start(f, s);
  if (ls >= e) return;
  Scanner sc;
  sc.init(f.p, ls);
  Fld fl[4];
  PendC pend[16];
  int npd = 0;
  while (ls < e) {
    u64 nx;
    bool ok = parse_line<4>(sc, f.p, fl, ls, nx);
    ls = nx;
    if (!ok || fl[0].n == 0 || fl[1].n == 0) continue;
    i64 ts;
    if (!parse_ts(fl[3].p, fl[3].n, ts) || ts < CWS || ts >= WE) continue;
    i32 c = CAMPS.find(fl[2].p, fl[2].n);
    if (c < 0) continue;
    W.n_ck_valid++;
    PendC& pd = pend[npd++];
    pd.huid = hstr(fl[1].p, fl[1].n);
    __builtin_prefetch(&IDS[id_home(pd.huid)]);
    pd.key = ((u64)(u32)(ts - CWS) << 32) | (u32)c;
    pd.idp = encp(fl[0].p, fl[0].n);
    if (npd == 16) {
      flushC(W, pend, npd);
      npd = 0;
    }
  }
  flushC(W, pend, npd);
}

// ---------------------------------------------------------------- dedup
static void dedup_partition(TCtx& W, u32 q) {
  size_t n = 0;
  for (int t = 0; t < NT; t++) n += TC[t].es[q].count();
  if (n == 0) return;
  size_t cap = 16;
  while (cap < n * 2) cap <<= 1;
  W.dtab.ensure(cap);
  ERec* tab = W.dtab.p;
  memset(tab, 0, cap * sizeof(ERec));
  const u64 mask = cap - 1;
  int lg = __builtin_ctzll(cap);
  auto kill = [&](u32 g) {
    if (g == INVALID) return;
    u32 t = g >> TSHIFT, idx = g & ((1u << TSHIFT) - 1);
    TC[t].pa.at(idx)->np = DEAD;
  };
  for (int t = 0; t < NT; t++) {
    Strm<ERec>& st = TC[t].es[q];
    for (size_t b = 0; b < st.blocks.size(); b++) {
      const ERec* r = TC[t].ea.blk(st.blocks[b]);
      u32 cnt = st.bcount(b);
      for (u32 k = 0; k < cnt; k++) {
        const ERec& e = r[k];
        u64 i = (e.key >> (64 - lg)) & mask;
        for (;;) {
          ERec& s = tab[i];
          if (s.key == 0) {
            s = e;
            break;
          }
          if (s.key == e.key) {
            if (e.ing < s.ing) {
              kill(s.gidx);
              s = e;
            } else {
              kill(e.gidx);
            }
            break;
          }
          i = (i + 1) & mask;
        }
      }
    }
  }
}

// ---------------------------------------------------------------- per person
static inline void sort_u64(u64* a, u32 n, TCtx& W) {
  if (n < 2) return;
  if (n <= 24) {
    for (u32 i = 1; i < n; i++) {
      u64 v = a[i];
      u32 j = i;
      while (j > 0 && a[j - 1] > v) {
        a[j] = a[j - 1];
        j--;
      }
      a[j] = v;
    }
    return;
  }
  if (n < 4096) {
    std::sort(a, a + n);
    return;
  }
  // LSD radix on bits 32..63 (11/11/10)
  W.radtmp.ensure(n);
  u64* b = W.radtmp.p;
  static const int SH[3] = {32, 43, 54};
  static const int BW[3] = {11, 11, 10};
  u32 cnt[2048];
  u64* src = a;
  u64* dst = b;
  for (int pass = 0; pass < 3; pass++) {
    int sh = SH[pass];
    u32 m = (1u << BW[pass]) - 1;
    memset(cnt, 0, sizeof(u32) * (m + 1));
    for (u32 i = 0; i < n; i++) cnt[(src[i] >> sh) & m]++;
    u32 sum = 0;
    for (u32 i = 0; i <= m; i++) {
      u32 c = cnt[i];
      cnt[i] = sum;
      sum += c;
    }
    for (u32 i = 0; i < n; i++) dst[cnt[(src[i] >> sh) & m]++] = src[i];
    std::swap(src, dst);
  }
  if (src != a) memcpy(a, src, n * sizeof(u64));
}

static inline void sort_ck(CK* a, u32 n) {
  if (n < 2) return;
  if (n <= 24) {
    for (u32 i = 1; i < n; i++) {
      CK v = a[i];
      u32 j = i;
      while (j > 0 && a[j - 1].key > v.key) {
        a[j] = a[j - 1];
        j--;
      }
      a[j] = v;
    }
    return;
  }
  std::sort(a, a + n, [](const CK& x, const CK& y) { return x.key < y.key; });
}

static inline u64 resolve_name(u64 pk) {
  u64 j = id_home(pk);
  for (;;) {
    u64 k = IDS[j].key.load(std::memory_order_relaxed);
    if (k == pk) return IDP[j];
    if (k == 0) die("name not found");
    if (++j == IDCAP) j = 0;
  }
}

static void process_person(TCtx& W, Pers& P, u64* ev, CK* ck) {
  const u32 n = P.cnt, m = P.ccnt;
  sort_u64(ev, n, W);
  if (m > 1) sort_ck(ck, m);
  W.s_start.ensure(n);
  W.s_day.ensure(n);
  W.s_rev.ensure(n);
  u32* ss = W.s_start.p;
  u16* sd = W.s_day.p;
  u64* sr = W.s_rev.p;
  const u64 name = P.np;  // arena ref
  u32 ns = 0, prev = 0;
  i32 cur_day = -1;
  u32 d_sess = 0, d_ev = 0;
  u64 d_rev = 0;
  auto emit = [&]() {
    i16 topc = -1;
    if (d_rev > 0) {
      u64 best = 0;
      for (u32 c : W.cattouch) {
        u64 v = W.catrev[c];
        if (v > best || (v == best && (i16)c < topc)) {
          best = v;
          topc = (i16)c;
        }
        W.catrev[c] = 0;
      }
      W.cattouch.clear();
    }
    W.o_name.push_back(name);
    W.o_day.push_back((i32)(DAY0 + cur_day));
    W.o_sess.push_back((i32)d_sess);
    W.o_ev.push_back((i32)d_ev);
    W.o_rev.push_back((i64)d_rev);
    W.o_cat.push_back(topc);
  };
  for (u32 k = 0; k < n; k++) {
    u64 x = ev[k];
    u32 ts = (u32)(x >> 32);
    if (k == 0 || ts - prev > GAP) {
      i32 day = (i32)((WSOFF + ts) / (u64)DAYMS);
      if (day != cur_day) {
        if (cur_day >= 0) emit();
        cur_day = day;
        d_sess = d_ev = 0;
        d_rev = 0;
      }
      d_sess++;
      ss[ns] = ts;
      sd[ns] = (u16)day;
      sr[ns] = 0;
      ns++;
    }
    prev = ts;
    d_ev++;
    u32 X = (u32)(x >> 24) & 0xFF;
    if (X != 0xFF) {
      u32 pr = (u32)(x & 0xFFFFFF);
      sr[ns - 1] += pr;
      d_rev += pr;
      if (W.catrev[X] == 0) W.cattouch.push_back(X);
      W.catrev[X] += pr;
    }
  }
  if (cur_day >= 0) emit();

  // attribution
  const u32 org = (u32)NC;
  auto add = [&](u32 c, u32 d, u64 rev) {
    size_t ix = (size_t)c * ND + d;
    W.c_sess[ix]++;
    if (rev) {
      W.c_purch[ix]++;
      W.c_rev[ix] += rev;
    }
  };
  if (m == 0) {
    for (u32 s = 0; s < ns; s++) add(org, sd[s], sr[s]);
    return;
  }
  u32 j = 0;
  for (u32 s = 0; s < ns; s++) {
    u32 S = ss[s] + WEEK;
    while (j < m && (u32)(ck[j].key >> 32) <= S) j++;
    u32 c = org;
    if (j > 0) {
      u32 t = (u32)(ck[j - 1].key >> 32);
      if (S - t <= WEEK) {
        u32 k = j - 1;
        c = (u32)(ck[k].key & 0xFFFF);
        if (k > 0 && (u32)(ck[k - 1].key >> 32) == t) {
          u64 best = ck[k].idp;
          for (u32 r = k; r-- > 0 && (u32)(ck[r].key >> 32) == t;) {
            if (cmpstr(ck[r].idp, best) < 0) {
              best = ck[r].idp;
              c = (u32)(ck[r].key & 0xFFFF);
            }
          }
        }
      }
    }
    add(c, sd[s], sr[s]);
  }
  // assists: union of [t, t+WEEK] per campaign (click base), sessions S = ss+WEEK inside
  W.s_nds.ensure(ns);
  u32* nds = W.s_nds.p;
  nds[ns - 1] = ns;
  for (u32 s = ns - 1; s-- > 0;) nds[s] = (sd[s + 1] != sd[s]) ? s + 1 : nds[s + 1];
  u32 stamp = ++W.stamp;
  W.ctouch.clear();
  auto flush = [&](u32 c, u32 a, u32 b) {
    u32 lo = a >= WEEK ? a - WEEK : 0;
    u32 hi = b - WEEK;
    u32 i0 = (u32)(std::lower_bound(ss, ss + ns, lo) - ss);
    u32 i1 = (u32)(std::upper_bound(ss + i0, ss + ns, hi) - ss);
    u32* asst = W.c_assist.data() + (size_t)c * ND;
    while (i0 < i1) {
      u32 e2 = std::min(i1, nds[i0]);
      asst[sd[i0]] += e2 - i0;
      i0 = e2;
    }
  };
  for (u32 k = 0; k < m; k++) {
    u32 c = (u32)(ck[k].key & 0xFFFF);
    u32 t = (u32)(ck[k].key >> 32);
    CampState& st = W.cst[c];
    if (st.stamp == stamp) {
      if (t <= st.b) {
        st.b = t + WEEK;
        continue;
      }
      flush(c, st.a, st.b);
      st.a = t;
      st.b = t + WEEK;
    } else {
      st.stamp = stamp;
      st.a = t;
      st.b = t + WEEK;
      W.ctouch.push_back(c);
    }
  }
  for (u32 c : W.ctouch) flush(c, W.cst[c].a, W.cst[c].b);
}

static void person_partition(TCtx& W, u32 p) {
  size_t nrec = 0, ncl = 0;
  for (int t = 0; t < NT; t++) {
    nrec += TC[t].ps[p].count();
    ncl += TC[t].cs[p].count();
  }
  if (nrec == 0) return;
  size_t cap = 64;
  while (cap < nrec / 4 + 16) cap <<= 1;
  W.ptk.ensure(cap);
  W.ptv.ensure(cap);
  u64* tk = W.ptk.p;
  u32* tv = W.ptv.p;
  memset(tk, 0, cap * sizeof(u64));
  u64 mask = cap - 1;
  std::vector<Pers>& PS = W.persons;
  PS.clear();
  W.lp_of.ensure(nrec);
  W.tmp64.ensure(nrec);
  u32* lpo = W.lp_of.p;
  u64* tmp = W.tmp64.p;
  size_t k = 0;
  auto grow = [&]() {
    size_t nc = cap * 2;
    u64* nk = (u64*)malloc(nc * sizeof(u64));
    u32* nv = (u32*)malloc(nc * sizeof(u32));
    memset(nk, 0, nc * sizeof(u64));
    u64 nm = nc - 1;
    for (size_t i = 0; i < cap; i++)
      if (tk[i]) {
        u64 j = (tk[i] >> 20) & nm;
        while (nk[j]) j = (j + 1) & nm;
        nk[j] = tk[i];
        nv[j] = tv[i];
      }
    free(W.ptk.p);
    free(W.ptv.p);
    W.ptk.p = nk;
    W.ptv.p = nv;
    W.ptk.cap = W.ptv.cap = nc;
    tk = nk;
    tv = nv;
    cap = nc;
    mask = nm;
  };
  for (int t = 0; t < NT; t++) {
    Strm<PRec>& st = TC[t].ps[p];
    for (size_t b = 0; b < st.blocks.size(); b++) {
      const PRec* r = TC[t].pa.blk(st.blocks[b]);
      u32 cnt = st.bcount(b);
      for (u32 i = 0; i < cnt; i++) {
        const PRec& x = r[i];
        if (x.np == DEAD) continue;
        u64 j = (x.pk >> 20) & mask;
        u32 lp;
        for (;;) {
          u64 kk = tk[j];
          if (kk == x.pk) {
            lp = tv[j];
            break;
          }
          if (kk == 0) {
            lp = (u32)PS.size();
            tk[j] = x.pk;
            tv[j] = lp;
            PS.push_back({x.pk, 0, 0, 0, 0, 0});
            if (PS.size() * 2 > cap) grow();
            break;
          }
          j = (j + 1) & mask;
        }
        Pers& P = PS[lp];
        P.cnt++;
        if (P.np == 0 && x.np) P.np = x.np;
        lpo[k] = lp;
        tmp[k] = x.packed;
        k++;
      }
    }
  }
  size_t live = k;
  u32 off = 0;
  for (Pers& P : PS) {
    P.off = off;
    off += P.cnt;
    P.cnt = 0;
  }
  W.ev.ensure(live);
  u64* ev = W.ev.p;
  for (size_t i = 0; i < live; i++) {
    Pers& P = PS[lpo[i]];
    ev[P.off + P.cnt++] = tmp[i];
  }
  // clicks
  W.clp.ensure(ncl + 1);
  W.cktmp.ensure(ncl + 1);
  u32* clp = W.clp.p;
  CK* ckt = W.cktmp.p;
  size_t kc = 0;
  for (int t = 0; t < NT; t++) {
    Strm<CRec>& st = TC[t].cs[p];
    for (size_t b = 0; b < st.blocks.size(); b++) {
      const CRec* r = TC[t].ca.blk(st.blocks[b]);
      u32 cnt = st.bcount(b);
      for (u32 i = 0; i < cnt; i++) {
        const CRec& x = r[i];
        u64 j = (x.pk >> 20) & mask;
        for (;;) {
          u64 kk = tk[j];
          if (kk == x.pk) {
            u32 lp = tv[j];
            PS[lp].ccnt++;
            clp[kc] = lp;
            ckt[kc] = {x.key, x.idp};
            kc++;
            break;
          }
          if (kk == 0) break;
          j = (j + 1) & mask;
        }
      }
    }
  }
  u32 coff = 0;
  for (Pers& P : PS) {
    P.coff = coff;
    coff += P.ccnt;
    P.ccnt = 0;
  }
  W.ck.ensure(kc + 1);
  CK* ck = W.ck.p;
  for (size_t i = 0; i < kc; i++) {
    Pers& P = PS[clp[i]];
    ck[P.coff + P.ccnt++] = ckt[i];
  }
  // names: copy each person's name once into the thread arena (prefetched)
  {
    const size_t NP = PS.size();
    const size_t D = 8;
    for (size_t i = 0; i < NP; i++) {
      if (i + D < NP) {
        const Pers& Q = PS[i + D];
        if (Q.np)
          __builtin_prefetch(decp(Q.np));
        else
          __builtin_prefetch(&IDS[id_home(Q.pk)]);
      }
      Pers& P = PS[i];
      u64 np = P.np ? P.np : resolve_name(P.pk);
      const char* sp = decp(np);
      u32 sn = decn(np);
      u64 off = W.names.size();
      W.names.insert(W.names.end(), sp, sp + sn);
      P.np = (off << 16) | sn;
    }
  }
  for (Pers& P : PS) process_person(W, P, ev + P.off, ck + P.coff);
}

// ---------------------------------------------------------------- parquet writer
struct Out {
  std::vector<char> b;
  inline void u8_(u8 v) { b.push_back((char)v); }
  inline void put(const void* p, size_t n) {
    const char* c = (const char*)p;
    b.insert(b.end(), c, c + n);
  }
  inline void varint(u64 v) {
    while (v >= 0x80) {
      b.push_back((char)(v | 0x80));
      v >>= 7;
    }
    b.push_back((char)v);
  }
  inline void u32le(u32 v) { put(&v, 4); }
  size_t size() const { return b.size(); }
};
static inline u64 zz64(i64 v) { return ((u64)v << 1) ^ (u64)(v >> 63); }
struct Thr {
  Out& o;
  int last = 0;
  std::vector<int> st;
  explicit Thr(Out& out) : o(out) {}
  void fh(int id, int t) {
    int d = id - last;
    if (d > 0 && d <= 15)
      o.u8_((u8)((d << 4) | t));
    else {
      o.u8_((u8)t);
      o.varint(zz64(id));
    }
    last = id;
  }
  void i32f(int id, i32 v) {
    fh(id, 5);
    o.varint(zz64(v));
  }
  void i64f(int id, i64 v) {
    fh(id, 6);
    o.varint(zz64(v));
  }
  void binf(int id, const std::string& s) {
    fh(id, 8);
    o.varint(s.size());
    o.put(s.data(), s.size());
  }
  void sbeg(int id) {
    fh(id, 12);
    st.push_back(last);
    last = 0;
  }
  void send() {
    o.u8_(0);
    last = st.back();
    st.pop_back();
  }
  void lbeg(int id, int et, int n) {
    fh(id, 9);
    if (n < 15)
      o.u8_((u8)((n << 4) | et));
    else {
      o.u8_((u8)(0xF0 | et));
      o.varint(n);
    }
  }
  void ebeg() {
    st.push_back(last);
    last = 0;
  }
  void eend() { send(); }
};

enum { PT_INT32 = 1, PT_INT64 = 2, PT_BYTE_ARRAY = 6 };
struct ColDef {
  std::string name;
  int ptype;
  int conv;     // -1 none, 0 UTF8, 6 DATE
  int logical;  // 0 none, 1 STRING, 6 DATE
  bool optional;
};
struct ColOut {
  Out data;
  u32 nvals = 0;
  u64 offset = 0, size = 0;
};

static void write_parquet(const std::string& path, const std::vector<ColDef>& defs, std::vector<ColOut>& cols,
                          u64 nrows) {
  Out f;
  size_t est = 8;
  for (auto& c : cols) est += c.data.size() + 64;
  f.b.reserve(est + 4096);
  f.put("PAR1", 4);
  for (size_t ci = 0; ci < cols.size(); ci++) {
    ColOut& c = cols[ci];
    Out h;
    Thr t(h);
    t.i32f(1, 0);
    t.i32f(2, (i32)c.data.size());
    t.i32f(3, (i32)c.data.size());
    t.sbeg(5);
    t.i32f(1, (i32)nrows);
    t.i32f(2, 0);
    t.i32f(3, 3);
    t.i32f(4, 3);
    t.send();
    h.u8_(0);
    c.offset = f.size();
    f.put(h.b.data(), h.size());
    f.put(c.data.b.data(), c.data.size());
    c.size = h.size() + c.data.size();
  }
  Out m;
  Thr t(m);
  t.i32f(1, 1);
  t.lbeg(2, 12, (int)defs.size() + 1);
  t.ebeg();
  t.binf(4, "schema");
  t.i32f(5, (i32)defs.size());
  t.eend();
  for (auto& d : defs) {
    t.ebeg();
    t.i32f(1, d.ptype);
    t.i32f(3, d.optional ? 1 : 0);
    t.binf(4, d.name);
    if (d.conv >= 0) t.i32f(6, d.conv);
    if (d.logical) {
      t.sbeg(10);
      t.sbeg(d.logical);
      t.send();
      t.send();
    }
    t.eend();
  }
  t.i64f(3, (i64)nrows);
  if (nrows == 0) {
    t.lbeg(4, 12, 0);
  } else {
    t.lbeg(4, 12, 1);
    t.ebeg();
    t.lbeg(1, 12, (int)defs.size());
    u64 total = 0;
    for (size_t ci = 0; ci < cols.size(); ci++) {
      ColOut& c = cols[ci];
      total += c.size;
      t.ebeg();
      t.i64f(2, (i64)c.offset);
      t.sbeg(3);
      t.i32f(1, defs[ci].ptype);
      t.lbeg(2, 5, 2);
      m.varint(zz64(0));
      m.varint(zz64(3));
      t.lbeg(3, 8, 1);
      m.varint(defs[ci].name.size());
      m.put(defs[ci].name.data(), defs[ci].name.size());
      t.i32f(4, 0);
      t.i64f(5, (i64)nrows);
      t.i64f(6, (i64)c.size);
      t.i64f(7, (i64)c.size);
      t.i64f(9, (i64)c.offset);
      t.send();
      t.eend();
    }
    t.i64f(2, (i64)total);
    t.i64f(3, (i64)nrows);
    t.eend();
  }
  t.binf(6, "shopping-sessions native");
  m.u8_(0);
  f.put(m.b.data(), m.size());
  f.u32le((u32)m.size());
  f.put("PAR1", 4);
  if (nrows == 0) {
    // drop the column chunks: rewrite as header + footer only
    Out g;
    g.put("PAR1", 4);
    g.put(m.b.data(), m.size());
    g.u32le((u32)m.size());
    g.put("PAR1", 4);
    f.b.swap(g.b);
  }
  int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) die(("create " + path).c_str());
  size_t done = 0;
  while (done < f.size()) {
    ssize_t w = write(fd, f.b.data() + done, f.size() - done);
    if (w <= 0) die("write");
    done += (size_t)w;
  }
  close(fd);
}

static void put_str_col(Out& o, const char* p, u32 n) {
  o.u32le(n);
  o.put(p, n);
}
static void put_defs(Out& o, const std::vector<u8>& defs) {
  u64 n = defs.size();
  u64 ng = (n + 7) / 8;
  Out enc;
  enc.varint((ng << 1) | 1);
  for (u64 g = 0; g < ng; g++) {
    u8 byte = 0;
    for (int b = 0; b < 8; b++) {
      u64 i = g * 8 + b;
      if (i < n && defs[i]) byte |= (u8)(1 << b);
    }
    enc.u8_(byte);
  }
  o.u32le((u32)enc.size());
  o.put(enc.b.data(), enc.size());
}

static void write_user_daily(TCtx& W, const std::string& path) {
  double tw0 = nowd();
  std::vector<ColDef> defs = {{"user_id", PT_BYTE_ARRAY, 0, 1, false}, {"day", PT_INT32, 6, 6, false},
                              {"sessions", PT_INT32, -1, 0, false},    {"events", PT_INT32, -1, 0, false},
                              {"revenue_cents", PT_INT64, -1, 0, false}, {"top_category", PT_BYTE_ARRAY, 0, 1, true}};
  std::vector<ColOut> cols(6);
  size_t n = W.o_name.size();
  {
    size_t tot = 0;
    for (size_t i = 0; i < n; i++) tot += 4 + (W.o_name[i] & 0xFFFF);
    std::vector<char>& b = cols[0].data.b;
    b.resize(tot);
    char* o = b.data();
    const char* nb = W.names.data();
    for (size_t i = 0; i < n; i++) {
      u32 len = (u32)(W.o_name[i] & 0xFFFF);
      memcpy(o, &len, 4);
      memcpy(o + 4, nb + (W.o_name[i] >> 16), len);
      o += 4 + len;
    }
  }
  cols[1].data.put(W.o_day.data(), n * 4);
  cols[2].data.put(W.o_sess.data(), n * 4);
  cols[3].data.put(W.o_ev.data(), n * 4);
  cols[4].data.put(W.o_rev.data(), n * 8);
  {
    Out& c5 = cols[5].data;
    u64 ng = (n + 7) / 8;
    Out enc;
    enc.varint((ng << 1) | 1);
    size_t hdr = enc.size();
    enc.b.resize(hdr + ng, 0);
    size_t vt = 0;
    for (size_t i = 0; i < n; i++)
      if (W.o_cat[i] >= 0) {
        enc.b[hdr + i / 8] |= (char)(1 << (i % 8));
        vt += 4 + CATS[W.o_cat[i]].size();
      }
    c5.u32le((u32)enc.size());
    c5.put(enc.b.data(), enc.size());
    size_t base = c5.size();
    c5.b.resize(base + vt);
    char* o = c5.b.data() + base;
    for (size_t i = 0; i < n; i++)
      if (W.o_cat[i] >= 0) {
        const std::string& cs = CATS[W.o_cat[i]];
        u32 len = (u32)cs.size();
        memcpy(o, &len, 4);
        memcpy(o + 4, cs.data(), len);
        o += 4 + len;
      }
  }
  double tb = nowd();
  write_parquet(path, defs, cols, n);
  if (VERBOSE) fprintf(stderr, "  t%d user_daily rows=%zu build=%.1fms write=%.1fms\n", W.id, n, (tb - tw0) * 1000, (nowd() - tb) * 1000);
}

// ---------------------------------------------------------------- main
static std::string json_str(const std::string& j, const char* key) {
  size_t k = j.find(std::string("\"") + key + "\"");
  if (k == std::string::npos) die("manifest key");
  size_t c = j.find(':', k);
  size_t q1 = j.find('"', c + 1);
  size_t q2 = j.find('"', q1 + 1);
  return j.substr(q1 + 1, q2 - q1 - 1);
}

static void load_small_table(const std::string& path, std::vector<std::pair<std::string, std::string>>& rows) {
  MFile f = map_file(path);
  u64 ls = align_start(f, 0);
  if (ls >= f.n) return;
  Scanner sc;
  sc.init(f.p, ls);
  Fld fl[2];
  while (ls < f.n) {
    u64 nx;
    bool ok = parse_line<2>(sc, f.p, fl, ls, nx);
    ls = nx;
    if (!ok) continue;
    rows.emplace_back(std::string(fl[0].p, fl[0].n), std::string(fl[1].p, fl[1].n));
  }
}

int main(int argc, char** argv) {
  T0 = nowd();
  init_cal();
  VERBOSE = getenv("PIPE_VERBOSE") != nullptr;
  std::string DATA = getenv("DATA_DIR") ? getenv("DATA_DIR") : "/data";
  std::string OUTD = getenv("OUTPUT_DIR") ? getenv("OUTPUT_DIR") : "/output";
  if (argc >= 3) {
    DATA = argv[argc - 2];
    OUTD = argv[argc - 1];
  }
  NT = getenv("NCPU") ? atoi(getenv("NCPU")) : (int)std::thread::hardware_concurrency();
  if (NT < 1) NT = 1;
  if (NT > 64) NT = 64;
  {
    int tb = 1;
    while ((1 << tb) < NT) tb++;
    TSHIFT = 32 - tb;
  }

  // manifest
  {
    std::string j = read_small(DATA + "/manifest.json");
    std::string a = json_str(j, "window_start"), b = json_str(j, "window_end");
    i64 x, y;
    if (!parse_f12(a.data(), (u32)a.size(), x) || !parse_f12(b.data(), (u32)b.size(), y)) die("manifest ts");
    WS = x;
    WE = y;
    CWS = WS - (i64)WEEK;
    DAY0 = WS >= 0 ? WS / DAYMS : -((-WS + DAYMS - 1) / DAYMS);
    WSOFF = (u64)(WS - DAY0 * DAYMS);
    ND = (int)(((WE - 1) - DAY0 * DAYMS) / DAYMS + 1);
    if (WE - WS > 49LL * DAYMS) die("window too long");
  }
  // products & categories
  {
    std::vector<std::pair<std::string, std::string>> rows;
    load_small_table(DATA + "/products.csv", rows);
    std::vector<std::string> cats;
    for (auto& r : rows) cats.push_back(r.second);
    cats.push_back("unknown");
    std::sort(cats.begin(), cats.end());
    cats.erase(std::unique(cats.begin(), cats.end()), cats.end());
    if (cats.size() > 250) die("too many categories");
    CATS = cats;
    UNKCAT = (int)(std::lower_bound(CATS.begin(), CATS.end(), std::string("unknown")) - CATS.begin());
    PRODS.init(rows.size());
    PRODCAT.clear();
    for (auto& r : rows) {
      i32 idx = PRODS.insert(r.first.data(), (u32)r.first.size());
      if ((size_t)idx == PRODCAT.size())
        PRODCAT.push_back(
            (u8)(std::lower_bound(CATS.begin(), CATS.end(), r.second) - CATS.begin()));
    }
  }
  // campaigns
  {
    std::vector<std::pair<std::string, std::string>> rows;
    load_small_table(DATA + "/campaigns.csv", rows);
    CAMPS.init(rows.size());
    for (auto& r : rows) {
      i32 idx = CAMPS.insert(r.first.data(), (u32)r.first.size());
      if ((size_t)idx == CHANNELS.size()) CHANNELS.push_back(r.second);
    }
    NC = (int)CAMPS.str.size();
    if (NC > 65000) die("too many campaigns");
  }
  logt("small tables");

  std::vector<std::string> evp = list_csv(DATA + "/events"), ckp = list_csv(DATA + "/clicks"),
                           lkp = list_csv(DATA + "/links");
  std::vector<MFile> evf, ckf, lkf;
  for (auto& s : evp) evf.push_back(map_file(s));
  for (auto& s : ckp) ckf.push_back(map_file(s));
  for (auto& s : lkp) lkf.push_back(map_file(s));
  u64 evbytes = 0;
  for (auto& f : evf) evbytes += f.n;
  {
    double est = (double)evbytes / 85.0;
    int pb = (int)std::lround(std::log2(std::max(1.0, est / 30000.0)));
    PBITS = std::min(12, std::max(4, pb));
    int qb = (int)std::lround(std::log2(std::max(1.0, est / 12000.0)));
    QBITS = std::min(13, std::max(4, qb));
  }
  TC = new TCtx[NT];
  for (int t = 0; t < NT; t++) {
    TC[t].id = t;
    TC[t].ps.resize(1u << PBITS);
    TC[t].cs.resize(1u << PBITS);
    TC[t].es.resize(1u << QBITS);
  }
  logt("mapped");

  links_phase(lkf);

  // parse events + clicks
  {
    std::vector<Chunk> ec = make_chunks(evf, 4 << 20), cc = make_chunks(ckf, 4 << 20);
    struct Job {
      int kind;
      Chunk c;
    };
    std::vector<Job> jobs;
    for (auto& c : ec) jobs.push_back({0, c});
    for (auto& c : cc) jobs.push_back({1, c});
    std::stable_sort(jobs.begin(), jobs.end(),
                     [](const Job& a, const Job& b) { return (a.c.e - a.c.s) > (b.c.e - b.c.s); });
    std::atomic<size_t> next{0};
    par_run([&](int t) {
      TCtx& W = TC[t];
      for (;;) {
        size_t i = next.fetch_add(1);
        if (i >= jobs.size()) break;
        const Job& j = jobs[i];
        if (j.kind == 0)
          events_chunk(W, evf[j.c.file], j.c.s, j.c.e);
        else
          clicks_chunk(W, ckf[j.c.file], j.c.s, j.c.e);
      }
    });
  }
  logt("parsed events+clicks");
  if (getenv("PIPE_STOP_PARSE")) { fprintf(stderr, "x %llu\n", (unsigned long long)TC[0].n_ev_valid); _exit(0); }

  {
    std::atomic<u32> next{0};
    u32 Q = 1u << QBITS;
    par_run([&](int t) {
      for (;;) {
        u32 q = next.fetch_add(1);
        if (q >= Q) break;
        dedup_partition(TC[t], q);
      }
    });
  }
  logt("dedup");

  {
    u32 P = 1u << PBITS;
    std::vector<std::pair<u64, u32>> order(P);
    for (u32 p = 0; p < P; p++) {
      u64 s = 0;
      for (int t = 0; t < NT; t++) s += TC[t].ps[p].count();
      order[p] = {s, p};
    }
    std::sort(order.begin(), order.end(), [](auto& a, auto& b) { return a.first > b.first; });
    std::atomic<u32> next{0};
    par_run([&](int t) {
      TCtx& W = TC[t];
      W.catrev.assign(CATS.size(), 0);
      W.cst.assign(NC + 1, CampState{0, 0, 0, 0});
      size_t cd = (size_t)(NC + 1) * ND;
      W.c_sess.assign(cd, 0);
      W.c_purch.assign(cd, 0);
      W.c_assist.assign(cd, 0);
      W.c_rev.assign(cd, 0);
      for (;;) {
        u32 i = next.fetch_add(1);
        if (i >= P) break;
        person_partition(W, order[i].second);
      }
    });
  }
  logt("persons");

  // outputs
  std::string ud = OUTD + "/user_daily", cdd = OUTD + "/campaign_daily";
  mkdir(OUTD.c_str(), 0755);
  mkdir(ud.c_str(), 0755);
  mkdir(cdd.c_str(), 0755);
  {
    std::atomic<int> next{0};
    par_run([&](int t) {
      if (t == 0) {
        // campaign_daily
        size_t cd = (size_t)(NC + 1) * ND;
        std::vector<u64> s(cd, 0), pu(cd, 0), as(cd, 0), rv(cd, 0);
        for (int u = 0; u < NT; u++)
          for (size_t i = 0; i < cd; i++) {
            s[i] += TC[u].c_sess[i];
            pu[i] += TC[u].c_purch[i];
            as[i] += TC[u].c_assist[i];
            rv[i] += TC[u].c_rev[i];
          }
        std::vector<ColDef> defs = {{"campaign_id", PT_BYTE_ARRAY, 0, 1, false},
                                    {"channel", PT_BYTE_ARRAY, 0, 1, false},
                                    {"day", PT_INT32, 6, 6, false},
                                    {"sessions", PT_INT64, -1, 0, false},
                                    {"purchasing_sessions", PT_INT64, -1, 0, false},
                                    {"revenue_cents", PT_INT64, -1, 0, false},
                                    {"assisted_sessions", PT_INT64, -1, 0, false}};
        std::vector<ColOut> cols(7);
        u64 nr = 0;
        for (int c = 0; c <= NC; c++)
          for (int d = 0; d < ND; d++) {
            size_t i = (size_t)c * ND + d;
            if (s[i] == 0 && as[i] == 0) continue;
            const std::string& id = c == NC ? std::string("organic") : CAMPS.str[c];
            const std::string& chn = c == NC ? std::string("organic") : CHANNELS[c];
            put_str_col(cols[0].data, id.data(), (u32)id.size());
            put_str_col(cols[1].data, chn.data(), (u32)chn.size());
            i32 day = (i32)(DAY0 + d);
            cols[2].data.put(&day, 4);
            i64 v;
            v = (i64)s[i];
            cols[3].data.put(&v, 8);
            v = (i64)pu[i];
            cols[4].data.put(&v, 8);
            v = (i64)rv[i];
            cols[5].data.put(&v, 8);
            v = c == NC ? 0 : (i64)as[i];
            cols[6].data.put(&v, 8);
            nr++;
          }
        write_parquet(cdd + "/part-0.parquet", defs, cols, nr);
      }
      TCtx& W = TC[t];
      size_t tot = 0;
      for (int u = 0; u < NT; u++) tot += TC[u].o_name.size();
      if (W.o_name.empty() && !(t == 0 && tot == 0)) return;
      write_user_daily(W, ud + "/part-" + std::to_string(t) + ".parquet");
    });
  }
  logt("written");
  if (VERBOSE) {
    u64 a = 0, b = 0, c = 0, r = 0;
    for (int t = 0; t < NT; t++) {
      a += TC[t].n_ev_lines;
      b += TC[t].n_ev_valid;
      c += TC[t].n_ck_valid;
      r += TC[t].o_name.size();
    }
    fprintf(stderr, "event lines %llu valid-copies %llu clicks %llu user_daily rows %llu P=%d Q=%d\n",
            (unsigned long long)a, (unsigned long long)b, (unsigned long long)c, (unsigned long long)r, PBITS,
            QBITS);
  }
  fflush(stderr);
  _exit(0);
}
