#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "big_uint.cuh"

#ifndef PRIMEFAC_BITS
#define PRIMEFAC_BITS 2048
#endif

static_assert(PRIMEFAC_BITS >= 64, "PRIMEFAC_BITS must be >= 64");
static_assert((PRIMEFAC_BITS % 64) == 0, "PRIMEFAC_BITS must be a multiple of 64");

constexpr size_t kBigBits = (size_t)PRIMEFAC_BITS;
constexpr size_t kBigLimbs = (kBigBits + 63u) / 64u;
using Big = bigu::BigUInt<kBigLimbs>;

// Any divisor we test satisfies d <= sqrt(N), so for a fixed-width N this fits in ~half the bits.
constexpr size_t kDivLimbs = (kBigLimbs + 1u) / 2u;
using Div = bigu::BigUInt<kDivLimbs>;

static void print_status_line(const std::string& msg)
{
    static size_t lastLen = 0;
    std::cout << "\r" << msg;
    if (lastLen > msg.size()) {
        std::cout << std::string(lastLen - msg.size(), ' ');
    }
    std::cout << std::flush;
    lastLen = msg.size();
}

enum class TrialStrategy {
    Linear,
    Random,
    MeetInTheMiddle,
};

static const char* strategy_name(TrialStrategy s)
{
    switch (s) {
        case TrialStrategy::Linear:
            return "linear";
        case TrialStrategy::Random:
            return "random";
        case TrialStrategy::MeetInTheMiddle:
            return "mitm";
    }
    return "linear";
}

static bool parse_strategy_arg(const std::string& s, TrialStrategy& out)
{
    if (s == "linear") {
        out = TrialStrategy::Linear;
        return true;
    }
    if (s == "random") {
        out = TrialStrategy::Random;
        return true;
    }
    if (s == "mitm" || s == "meet" || s == "meet-in-the-middle" || s == "meet_in_the_middle") {
        out = TrialStrategy::MeetInTheMiddle;
        return true;
    }
    return false;
}

static bool parse_u64_arg(const char* s, uint64_t& out)
{
    if (s == nullptr || *s == '\0') {
        return false;
    }
    char* endp = nullptr;
    unsigned long long v = std::strtoull(s, &endp, 10);
    if (endp == s || *endp != '\0') {
        return false;
    }
    out = (uint64_t)v;
    return true;
}

static bool opt_is(const char* a, const char* opt)
{
    const size_t n = std::strlen(opt);
    return std::strncmp(a, opt, n) == 0 && (a[n] == '\0' || a[n] == '=');
}

static inline uint64_t splitmix64(uint64_t& x)
{
    uint64_t z = (x += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static long double big_to_long_double_approx(const Div& x)
{
    // Convert using up to 3 highest non-zero limbs. Good enough for progress percentages.
    constexpr long double kTwo64 = 18446744073709551616.0L;
    long double v = 0.0L;
    int used = 0;
    for (size_t i = kDivLimbs; i-- > 0;) {
        uint64_t w = x.limb[i];
        if (w == 0 && used == 0) {
            continue;
        }
        v = v * kTwo64 + (long double)w;
        if (++used >= 3) {
            break;
        }
    }
    return v;
}

static double approx_percent_complete(const Div& done, const Div& total)
{
    if (total.is_zero()) return 0.0;
    Div d = done;
    if (d > total) d = total;

    // Scale both down by the same shift so the approximation stays in-range.
    uint32_t bl = total.bit_length();
    uint32_t shift = (bl > 160u) ? (bl - 160u) : 0u;
    Div ds = d.shr_bits(shift);
    Div ts = total.shr_bits(shift);

    long double dv = big_to_long_double_approx(ds);
    long double tv = big_to_long_double_approx(ts);
    if (tv <= 0.0L) return 0.0;
    long double p = (dv / tv) * 100.0L;
    if (p < 0.0L) p = 0.0L;
    if (p > 100.0L) p = 100.0L;
    return (double)p;
}

static bool parse_big_dec(const char* s, Big& out)
{
    if (s == nullptr || *s == '\0') {
        return false;
    }

    Big value = Big::zero();
    for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        uint32_t digit = (uint32_t)(*p - '0');

        // value = value * 10 + digit, with fixed-width overflow detection.
        if (value.mul_u32_inplace(10)) {
            return false;
        }
        if (value.add_u32_inplace(digit)) {
            return false;
        }
    }
    out = value;
    return true;
}

// Wheel modulus 510510 = 2*3*5*7*11*13*17.
// phi(510510) = 92160.
// The raw gap table is 92160 bytes as uint8_t, which does NOT fit in 64KiB constant memory.
// However, the maximum gap is 26, so each gap fits in 5 bits.
// 92160 * 5 bits = 57600 bytes, which DOES fit in 64KiB constant memory when bit-packed.
constexpr uint32_t kWheelMod = 510510;
constexpr uint32_t kWheelLen510510 = 92160;
constexpr uint32_t kWheelGapBits = 5;
constexpr uint32_t kWheelGapMask = (1u << kWheelGapBits) - 1u;
constexpr uint32_t kWheelPackedWords = (kWheelLen510510 * kWheelGapBits + 31u) / 32u; // 14400 words => 57600 bytes

__device__ __constant__ uint32_t dWheelGaps510510Packed[kWheelPackedWords];

static inline bool isWheelResidue510510(uint32_t x)
{
    return (x % 2u) != 0u && (x % 3u) != 0u && (x % 5u) != 0u && (x % 7u) != 0u && (x % 11u) != 0u && (x % 13u) != 0u && (x % 17u) != 0u;
}

static std::vector<uint8_t> makeWheelGaps510510()
{
    std::vector<uint32_t> residues;
    residues.reserve(kWheelLen510510);
    for (uint32_t x = 1; x <= kWheelMod; ++x) {
        if (isWheelResidue510510(x)) {
            residues.push_back(x);
        }
    }
    if (residues.size() != kWheelLen510510 || residues.empty() || residues[0] != 1u) {
        std::cerr << "Wheel residues generation failed (count=" << residues.size() << ")" << std::endl;
        std::exit(1);
    }

    std::vector<uint8_t> gaps(residues.size());
    uint64_t sum = 0;
    uint32_t maxGap = 0;
    for (size_t i = 0; i < residues.size(); ++i) {
        uint32_t cur = residues[i];
        uint32_t next = (i + 1 < residues.size()) ? residues[i + 1] : (residues[0] + kWheelMod);
        uint32_t gap = next - cur;
        if (gap > 255u) {
            std::cerr << "Gap too large for uint8_t: " << gap << std::endl;
            std::exit(1);
        }
        gaps[i] = (uint8_t)gap;
        sum += gap;
        if (gap > maxGap) {
            maxGap = gap;
        }
    }
    if (sum != kWheelMod) {
        std::cerr << "Wheel gaps sum mismatch: " << sum << " != " << kWheelMod << std::endl;
        std::exit(1);
    }
    (void)maxGap;
    return gaps;
}

static std::vector<uint32_t> packWheelGaps510510_5bit(const std::vector<uint8_t>& gaps)
{
    if (gaps.size() != kWheelLen510510) {
        std::cerr << "Unexpected gap count: " << gaps.size() << std::endl;
        std::exit(1);
    }

    std::vector<uint32_t> packed(kWheelPackedWords, 0u);
    for (uint32_t i = 0; i < (uint32_t)gaps.size(); ++i) {
        uint32_t g = (uint32_t)gaps[i];
        if (g > kWheelGapMask) {
            std::cerr << "Gap too large for 5-bit packing: " << g << std::endl;
            std::exit(1);
        }
        uint32_t bitPos = i * kWheelGapBits;
        uint32_t word = bitPos >> 5;
        uint32_t shift = bitPos & 31u;

        packed[word] |= (g << shift);
        if (shift > (32u - kWheelGapBits)) {
            // Straddles 32-bit boundary.
            uint32_t spill = (g >> (32u - shift));
            packed[word + 1] |= spill;
        }
    }
    return packed;
}

template <size_t LIMBS>
static inline bigu::BigUInt<LIMBS> ceil_div_by_u32(const bigu::BigUInt<LIMBS>& a, uint32_t b)
{
    // Returns ceil(a / b) for 32-bit b > 0.
    bigu::BigUInt<LIMBS> t = a;
    t.add_u32_inplace(b - 1u);
    (void)t.div_mod_u32_inplace(b);
    return t;
}

static inline Big isqrt_big(const Big& n)
{
    return bigu::isqrt_big(n);
}

static void factor_u64(uint64_t n, std::vector<uint64_t>& outPrimeFactors)
{
    if (n < 2) {
        return;
    }
    while ((n % 2ull) == 0ull) {
        outPrimeFactors.push_back(2ull);
        n /= 2ull;
    }
    for (uint64_t p = 3; p <= n / p; p += 2) {
        while ((n % p) == 0ull) {
            outPrimeFactors.push_back(p);
            n /= p;
        }
    }
    if (n > 1) {
        outPrimeFactors.push_back(n);
    }
}

template <size_t LIMBS>
static std::string uint_to_string(bigu::BigUInt<LIMBS> v)
{
    if (v.is_zero()) {
        return "0";
    }
    std::string s;
    while (!v.is_zero()) {
        uint32_t digit = v.div_mod_u32_inplace(10u);
        s.push_back(char('0' + digit));
    }
    std::reverse(s.begin(), s.end());
    return s;
}

static std::string big_to_string(Big v) { return uint_to_string(v); }

static inline Div big_to_div_trunc(const Big& x)
{
    Div d = Div::zero();
    for (size_t i = 0; i < kDivLimbs; ++i) {
        d.limb[i] = x.limb[i];
    }
    return d;
}

static inline Big div_to_big_zext(const Div& x)
{
    Big b = Big::zero();
    for (size_t i = 0; i < kDivLimbs; ++i) {
        b.limb[i] = x.limb[i];
    }
    return b;
}

static inline bool big_is_one_or_zero_divisor(uint64_t p) { return p <= 1ull; }

static inline bool big_divisible_by_u32(const Big& n, uint32_t p)
{
    Big t = n;
    return t.div_mod_u32_inplace(p) == 0u;
}

static inline void big_div_exact_u32_inplace(Big& n, uint32_t p)
{
    (void)n.div_mod_u32_inplace(p);
}

static void checkCuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << what << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

// Kernel searches wheel-block indices [startWheelBlock, endWheelBlock) and writes found divisors.
__device__ __forceinline__ uint32_t wheelGap510510(uint32_t i)
{
    // Extract 5-bit packed value from constant memory.
    uint32_t bitPos = i * kWheelGapBits;
    uint32_t word = bitPos >> 5;
    uint32_t shift = bitPos & 31u;

    uint32_t lo = dWheelGaps510510Packed[word];
    if (shift <= (32u - kWheelGapBits)) {
        return (lo >> shift) & kWheelGapMask;
    }
    uint32_t hi = dWheelGaps510510Packed[word + 1];
    uint32_t v = (lo >> shift) | (hi << (32u - shift));
    return v & kWheelGapMask;
}

__global__
void factorize_chunk(const Big* n,
                     const Div* sqrtn,
                     Div baseWheelBlock,
                     uint64_t startWheelOffset,
                     uint64_t endWheelOffset,
                     uint32_t* outCount,
                     Div* outDivisors,
                     uint32_t outCapacity)
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

    const Div one = Div::from_u64(1ull);

    for (uint64_t wheelOffset = startWheelOffset + idx; wheelOffset < endWheelOffset; wheelOffset += stride) {
        Div wheelBlock = baseWheelBlock;
        wheelBlock.add_u64_inplace(wheelOffset);

        Div value = wheelBlock;
        value.mul_u32_inplace(kWheelMod);
        value.add_u32_inplace(1u);

        // One full wheel traversal covers exactly (wheelBlock*kWheelMod + 1) .. +kWheelMod.
        for (uint32_t i = 0; i < kWheelLen510510; ++i) {
            if (value > *sqrtn) {
                return;
            }

            if (value > one) {
                Div r = bigu::mod_fast<kBigLimbs, kDivLimbs>(*n, value);
                if (r.is_zero()) {
                    uint32_t pos = atomicAdd(outCount, 1u);
                    if (pos < outCapacity) {
                        outDivisors[pos] = value;
                    }
                    return;
                }
            }
            value.add_u32_inplace(wheelGap510510(i));
        }
    }
}
 
int main(int argc, char** argv)
{
    TrialStrategy strategy = TrialStrategy::Linear;
    const char* nArg = nullptr;
    uint64_t warmupBlocksOpt = 4096ull;
    uint64_t strategyChunkBlocksOpt = 8192ull;
    uint64_t seedOpt = 0;
    bool seedProvided = false;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (a == nullptr) continue;
        if (opt_is(a, "--warmup-blocks")) {
            const char* valp = nullptr;
            if (a[15] == '=') {
                valp = a + 16;
            } else {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for --warmup-blocks" << std::endl;
                    return 1;
                }
                valp = argv[++i];
            }
            uint64_t v = 0;
            if (!parse_u64_arg(valp, v)) {
                std::cerr << "Invalid --warmup-blocks value: " << valp << std::endl;
                return 1;
            }
            warmupBlocksOpt = v;
            continue;
        }
        if (opt_is(a, "--strategy-chunk-blocks")) {
            const char* valp = nullptr;
            if (a[22] == '=') {
                valp = a + 23;
            } else {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for --strategy-chunk-blocks" << std::endl;
                    return 1;
                }
                valp = argv[++i];
            }
            uint64_t v = 0;
            if (!parse_u64_arg(valp, v) || v == 0) {
                std::cerr << "Invalid --strategy-chunk-blocks value: " << valp << " (must be >= 1)" << std::endl;
                return 1;
            }
            strategyChunkBlocksOpt = v;
            continue;
        }
        if (opt_is(a, "--seed")) {
            const char* valp = nullptr;
            if (a[6] == '=') {
                valp = a + 7;
            } else {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for --seed" << std::endl;
                    return 1;
                }
                valp = argv[++i];
            }
            uint64_t v = 0;
            if (!parse_u64_arg(valp, v)) {
                std::cerr << "Invalid --seed value: " << valp << std::endl;
                return 1;
            }
            seedOpt = v;
            seedProvided = true;
            continue;
        }
        if (opt_is(a, "--strategy")) {
            std::string val;
            if (a[10] == '=') {
                val = std::string(a + 11);
            } else {
                if (i + 1 >= argc) {
                    std::cerr << "Missing value for --strategy" << std::endl;
                    return 1;
                }
                val = std::string(argv[++i]);
            }
            if (!parse_strategy_arg(val, strategy)) {
                std::cerr << "Invalid strategy: " << val << " (expected linear|random|mitm)" << std::endl;
                return 1;
            }
            continue;
        }
        if (std::strncmp(a, "--", 2) == 0) {
            std::cerr << "Unknown option: " << a << std::endl;
            return 1;
        }
        if (nArg == nullptr) {
            nArg = a;
        } else {
            std::cerr << "Unexpected extra argument: " << a << std::endl;
            return 1;
        }
    }

    if (nArg == nullptr) {
        std::cerr << "Usage: " << argv[0]
                  << " [--strategy linear|random|mitm]"
                  << " [--warmup-blocks <wheelBlocks>]"
                  << " [--strategy-chunk-blocks <wheelBlocks>]"
                  << " [--seed <u64>]"
                  << " <N (decimal, >=2)>" << std::endl;
        return 1;
    }

    Big N;
    if (!parse_big_dec(nArg, N) || (N < Big::from_u64(2ull))) {
        std::cerr << "Invalid input number: " << nArg << std::endl;
        return 2;
    }

    // Build wheel gaps on CPU and upload to device constant memory (5-bit packed).
    const auto wheelGaps = makeWheelGaps510510();
    const auto wheelPacked = packWheelGaps510510_5bit(wheelGaps);
    if (wheelPacked.size() != kWheelPackedWords) {
        std::cerr << "Packed wheel size mismatch: " << wheelPacked.size() << " != " << kWheelPackedWords << std::endl;
        return 3;
    }
    checkCuda(cudaMemcpyToSymbol(dWheelGaps510510Packed,
                                 wheelPacked.data(),
                                 sizeof(uint32_t) * wheelPacked.size(),
                                 0,
                                 cudaMemcpyHostToDevice),
              "cudaMemcpyToSymbol(dWheelGaps510510Packed)");

    // Managed buffers used across factorization iterations.
    const int blockSize = 256;
    const uint32_t outCapacity = 128; // divisors are ~half width of N
    uint32_t* outCount = nullptr;
    Div* outDivisors = nullptr;
    Big* dN = nullptr;
    Div* dSqrtN = nullptr;
    checkCuda(cudaMallocManaged(&outCount, sizeof(uint32_t)), "cudaMallocManaged(outCount)");
    checkCuda(cudaMallocManaged(&outDivisors, sizeof(Div) * outCapacity), "cudaMallocManaged(outDivisors)");
    checkCuda(cudaMallocManaged(&dN, sizeof(Big)), "cudaMallocManaged(dN)");
    checkCuda(cudaMallocManaged(&dSqrtN, sizeof(Div)), "cudaMallocManaged(dSqrtN)");

    auto factor_big = [&](auto&& self, Big n, std::vector<Big>& outFactors) -> void {
        static thread_local int depth = 0;
        struct DepthGuard {
            int& d;
            DepthGuard(int& dd) : d(dd) { ++d; }
            ~DepthGuard() { --d; }
        } guard(depth);

        const bool doLog = (depth == 1);
        TrialStrategy localStrategy = strategy;

        // Fast path: use 64-bit factorization when possible.
        uint64_t n64 = 0;
        if (n.fits_u64(n64)) {
            std::vector<uint64_t> pf;
            factor_u64(n64, pf);
            for (uint64_t p : pf) {
                outFactors.push_back(Big::from_u64(p));
            }
            return;
        }

        Big remaining = n;

        // Fixed field widths for status output so the progress line doesn't visually jitter.
        // Compute once per top-level factoring call.
        size_t statusRemWidth = 0;
        size_t statusBlocksWidth = 0;
        if (doLog) {
            statusRemWidth = big_to_string(n).size();
            Big sq0 = isqrt_big(remaining);
            Div sq0d = big_to_div_trunc(sq0);
            Div twb0 = ceil_div_by_u32(sq0d, kWheelMod);
            statusBlocksWidth = uint_to_string(twb0).size();
        }

        // PRNG state for random strategy (deterministic per factoring call).
        uint64_t rngState = seedProvided ? seedOpt : 0x6a09e667f3bcc909ull;
        // Mix in the current remaining so different inputs still produce different sequences,
        // while remaining reproducible for a given (seed, N).
        rngState ^= remaining.limb[0];
        if constexpr (kBigLimbs > 1) rngState ^= remaining.limb[1] * 0x9e3779b97f4a7c15ull;
        if constexpr (kBigLimbs > 2) rngState ^= remaining.limb[2] * 0xbf58476d1ce4e5b9ull;

        auto divide_out_u32 = [&](uint32_t p) {
            while (!big_is_one_or_zero_divisor(p) && big_divisible_by_u32(remaining, p)) {
                outFactors.push_back(Big::from_u64((uint64_t)p));
                big_div_exact_u32_inplace(remaining, p);
            }
        };

        // Always handle wheel primes on CPU; the wheel never tests them.
        divide_out_u32(2);
        divide_out_u32(3);
        divide_out_u32(5);
        divide_out_u32(7);
        divide_out_u32(11);
        divide_out_u32(13);
        divide_out_u32(17);

        const uint32_t maxGridBlocks = 65535;
        const uint32_t gridBlocksForChunk = 256;
        const uint64_t threadsPerLaunch = (uint64_t)gridBlocksForChunk * (uint64_t)blockSize;
        const uint64_t wheelBlocksPerThread = 8;
        const uint64_t chunkWheelBlocks = threadsPerLaunch * wheelBlocksPerThread;

        const Big one = Big::from_u64(1ull);
        const Div oneDiv = Div::from_u64(1ull);

        while (remaining > one) {
            Big sqrtnBig = isqrt_big(remaining);
            Div sqrtn = big_to_div_trunc(sqrtnBig);
            Div totalWheelBlocks = ceil_div_by_u32(sqrtn, kWheelMod);

            bool foundAny = false;

            // For strategies that spend time near sqrt(N) (large denominators), keep each launch smaller
            // so we alternate more frequently and can bail out earlier.
            const uint64_t chunkBlocksThisRound = (localStrategy == TrialStrategy::Linear)
                                                      ? chunkWheelBlocks
                                                      : std::min<uint64_t>(chunkWheelBlocks, strategyChunkBlocksOpt);
            const Div chunkDiv = Div::from_u64(chunkBlocksThisRound);

            auto run_chunk = [&](const Div& startBlock, uint64_t endOffset, Div& doneBlocks) -> bool {
                Div endBlock = startBlock;
                endBlock.add_u64_inplace(endOffset);

                // Count attempted work regardless of whether progress logging is enabled.
                doneBlocks.add_u64_inplace(endOffset);

                if (doLog) {
                    const double pct = approx_percent_complete(doneBlocks, totalWheelBlocks);
                    std::ostringstream oss;
                    oss << std::setw(6) << std::fixed << std::setprecision(2) << pct << "% "
                        << strategy_name(localStrategy) << " blocks ["
                        << std::setw(statusBlocksWidth) << uint_to_string(startBlock) << ","
                        << std::setw(statusBlocksWidth) << uint_to_string(endBlock) << ")/"
                        << std::setw(statusBlocksWidth) << uint_to_string(totalWheelBlocks)
                        << " rem=" << std::setw(statusRemWidth) << uint_to_string(remaining);
                    print_status_line(oss.str());
                }

                *outCount = 0;
                *dN = remaining;
                *dSqrtN = sqrtn;
                checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(pre-launch)");

                factorize_chunk<<<(uint32_t)std::min<uint64_t>(maxGridBlocks, gridBlocksForChunk), blockSize>>>(
                    dN,
                    dSqrtN,
                    startBlock,
                    0ull,
                    endOffset,
                    outCount,
                    outDivisors,
                    outCapacity);
                checkCuda(cudaGetLastError(), "kernel launch");
                checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(post-kernel)");

                uint32_t found = *outCount;
                if (doLog) {
                    const double pct = approx_percent_complete(doneBlocks, totalWheelBlocks);
                    std::ostringstream oss;
                    oss << std::setw(6) << std::fixed << std::setprecision(2) << pct << "% "
                        << strategy_name(localStrategy) << " blocks ["
                        << std::setw(statusBlocksWidth) << uint_to_string(startBlock) << ","
                        << std::setw(statusBlocksWidth) << uint_to_string(endBlock) << ")/"
                        << std::setw(statusBlocksWidth) << uint_to_string(totalWheelBlocks)
                        << " found " << std::setw(3) << found
                        << " rem=" << std::setw(statusRemWidth) << uint_to_string(remaining);
                    print_status_line(oss.str());
                }

                if (found == 0) {
                    return false;
                }

                uint32_t toProcess = std::min(found, outCapacity);
                for (uint32_t i = 0; i < toProcess; ++i) {
                    Div d = outDivisors[i];
                    Div r = bigu::mod_fast<kBigLimbs, kDivLimbs>(remaining, d);
                    if (d > oneDiv && r.is_zero()) {
                        // Factor the divisor fully (it may be composite). If it divides multiple times,
                        // repeat those prime factors for each division.
                        Big dBig = div_to_big_zext(d);
                        std::vector<Big> divFactors;
                        self(self, dBig, divFactors);

                        bool firstDivision = true;
                        while (true) {
                            if (!firstDivision) {
                                Div rr = bigu::mod_fast<kBigLimbs, kDivLimbs>(remaining, d);
                                if (!rr.is_zero()) {
                                    break;
                                }
                            }
                            firstDivision = false;
                            for (const auto& pf : divFactors) {
                                outFactors.push_back(pf);
                            }
                            Big q;
                            Div rem;
                            bigu::div_mod_fast<kBigLimbs, kDivLimbs>(remaining, d, q, rem);
                            remaining = q;
                        }
                        return true;
                    }
                }
                return false;
            };

            auto chunk_len = [&](const Div& cur, const Div& limit) -> uint64_t {
                Div blocksLeft = limit;
                blocksLeft.sub_inplace(cur);
                uint64_t endOffset = chunkBlocksThisRound;
                if (blocksLeft < chunkDiv) {
                    endOffset = blocksLeft.limb[0];
                }
                return endOffset;
            };

            Div doneBlocks = Div::zero();

            // Warmup prefix scan for non-linear strategies: quickly catch small factors cheaply.
            Div warmupLimit = Div::zero();
            if (localStrategy != TrialStrategy::Linear) {
                Div warmup = Div::from_u64(warmupBlocksOpt);
                warmupLimit = (totalWheelBlocks < warmup) ? totalWheelBlocks : warmup;

                Div cur = Div::zero();
                while (cur < warmupLimit && remaining > one) {
                    uint64_t endOffset = chunk_len(cur, warmupLimit);
                    if (run_chunk(cur, endOffset, doneBlocks)) {
                        foundAny = true;
                        break;
                    }
                    cur.add_u64_inplace(endOffset);
                }
            }

            if (foundAny) {
                continue;
            }

            if (localStrategy == TrialStrategy::Linear) {
                Div cur = Div::zero();
                while (cur < totalWheelBlocks && remaining > one) {
                    uint64_t endOffset = chunk_len(cur, totalWheelBlocks);
                    if (run_chunk(cur, endOffset, doneBlocks)) {
                        foundAny = true;
                        break;
                    }
                    cur.add_u64_inplace(endOffset);
                }
            } else if (localStrategy == TrialStrategy::Random) {
                if (warmupLimit < totalWheelBlocks) {
                    // Pseudo-random permutation of chunk indices, so we bounce around the space
                    // but still cover it completely if we run to completion.
                    Div span = totalWheelBlocks;
                    span.sub_inplace(warmupLimit);

                    Div q = Div::zero();
                    uint64_t r = 0;
                    bigu::div_mod_u64(span, chunkBlocksThisRound, q, r);
                    Div totalChunks = q;
                    if (r != 0) {
                        totalChunks.add_u64_inplace(1);
                    }

                    if (!totalChunks.is_zero()) {
                        const uint32_t k = totalChunks.bit_length();
                        auto mask_to_bits = [&](Div& v, uint32_t bits) {
                            if (bits == 0) {
                                v = Div::zero();
                                return;
                            }
                            const uint32_t top = (bits - 1u) / 64u;
                            const uint32_t bitsInTop = bits - top * 64u;
                            const uint64_t mask = (bitsInTop == 64u) ? ~0ull : ((1ull << bitsInTop) - 1ull);
                            for (size_t i = (size_t)top + 1; i < kDivLimbs; ++i) {
                                v.limb[i] = 0;
                            }
                            v.limb[top] &= mask;
                        };

                        // Build a full-period LCG over 2^k.
                        uint64_t mult = splitmix64(rngState);
                        mult = (mult & ~3ull) | 1ull; // A ≡ 1 (mod 4)
                        uint64_t inc = splitmix64(rngState) | 1ull; // B odd

                        Div state = Div::zero();
                        for (size_t i = 0; i < kDivLimbs; ++i) {
                            state.limb[i] = splitmix64(rngState);
                        }
                        mask_to_bits(state, k);

                        auto step = [&]() {
                            state.mul_u64_inplace(mult);
                            state.add_u64_inplace(inc);
                            mask_to_bits(state, k);
                        };

                        Div chunksDone = Div::zero();
                        while (chunksDone < totalChunks && remaining > one) {
                            Div idx;
                            do {
                                step();
                                idx = state;
                            } while (idx >= totalChunks);

                            Div startBlock = idx;
                            startBlock.mul_u64_inplace(chunkBlocksThisRound);
                            startBlock.add_inplace(warmupLimit);

                            uint64_t endOffset = chunk_len(startBlock, totalWheelBlocks);
                            if (run_chunk(startBlock, endOffset, doneBlocks)) {
                                foundAny = true;
                                break;
                            }
                            chunksDone.add_u64_inplace(1);
                        }
                    }
                }
            } else {
                // Meet in the middle: alternate scanning near sqrt (end) and near 0 (begin).
                Div low = warmupLimit;
                Div high = totalWheelBlocks;
                bool takeHigh = true;
                while (low < high && remaining > one) {
                    Div span = high;
                    span.sub_inplace(low);
                    uint64_t endOffset = chunkBlocksThisRound;
                    if (span < chunkDiv) {
                        endOffset = span.limb[0];
                    }

                    if (takeHigh) {
                        Div startBlock = high;
                        startBlock.sub_inplace(Div::from_u64(endOffset));
                        if (run_chunk(startBlock, endOffset, doneBlocks)) {
                            foundAny = true;
                            break;
                        }
                        high = startBlock;
                    } else {
                        Div startBlock = low;
                        if (run_chunk(startBlock, endOffset, doneBlocks)) {
                            foundAny = true;
                            break;
                        }
                        low.add_u64_inplace(endOffset);
                    }
                    takeHigh = !takeHigh;
                }
            }

            if (!foundAny) {
                // No divisors <= sqrt(remaining) were found, so remaining is prime.
                outFactors.push_back(remaining);
                break;
            }
        }

        if (doLog) {
            std::cout << std::endl;
        }
    };

    std::vector<Big> factors;
    factor_big(factor_big, N, factors);
    for (const auto& f : factors) {
        std::cout << "factor: " << big_to_string(f) << std::endl;
    }

    checkCuda(cudaFree(dSqrtN), "cudaFree(dSqrtN)");
    checkCuda(cudaFree(dN), "cudaFree(dN)");
    checkCuda(cudaFree(outDivisors), "cudaFree(outDivisors)");
    checkCuda(cudaFree(outCount), "cudaFree(outCount)");
    return 0;
}