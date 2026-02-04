#include <algorithm>
#include <array>
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
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <N (decimal, >=2)>" << std::endl;
        return 1;
    }

    Big N;
    if (!parse_big_dec(argv[1], N) || (N < Big::from_u64(2ull))) {
        std::cerr << "Invalid input number: " << argv[1] << std::endl;
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
        const Div chunkDiv = Div::from_u64(chunkWheelBlocks);

        while (remaining > one) {
            Big sqrtnBig = isqrt_big(remaining);
            Div sqrtn = big_to_div_trunc(sqrtnBig);
            Div totalWheelBlocks = ceil_div_by_u32(sqrtn, kWheelMod);

            bool foundAny = false;
            Div start = Div::zero();
            while (start < totalWheelBlocks && remaining > one) {
                Div blocksLeft = totalWheelBlocks;
                blocksLeft.sub_inplace(start);

                uint64_t endOffset = chunkWheelBlocks;
                if (blocksLeft < chunkDiv) {
                    // blocksLeft fits in u64 because it's < chunkWheelBlocks (u64).
                    endOffset = blocksLeft.limb[0];
                }

                Div endBlock = start;
                endBlock.add_u64_inplace(endOffset);

                if (doLog) {
                    const double pct = approx_percent_complete(endBlock, totalWheelBlocks);
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << pct << "% "
                        << "blocks [" << uint_to_string(start) << "," << uint_to_string(endBlock) << ")/" << uint_to_string(totalWheelBlocks)
                        << " rem=" << uint_to_string(remaining);
                    print_status_line(oss.str());
                }

                *outCount = 0;
                *dN = remaining;
                *dSqrtN = sqrtn;
                checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(pre-launch)");

                factorize_chunk<<<(uint32_t)std::min<uint64_t>(maxGridBlocks, gridBlocksForChunk), blockSize>>>(
                    dN,
                    dSqrtN,
                    start,
                    0ull,
                    endOffset,
                    outCount,
                    outDivisors,
                    outCapacity);
                checkCuda(cudaGetLastError(), "kernel launch");
                checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(post-kernel)");

                if (doLog) {
                    const double pct = approx_percent_complete(endBlock, totalWheelBlocks);
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << pct << "% "
                        << "blocks [" << uint_to_string(start) << "," << uint_to_string(endBlock) << ")/" << uint_to_string(totalWheelBlocks)
                        << " found " << *outCount << " rem=" << uint_to_string(remaining);
                    print_status_line(oss.str());
                }

                uint32_t found = *outCount;
                if (found > 0) {
                    uint32_t toProcess = std::min(found, outCapacity);
                    for (uint32_t i = 0; i < toProcess; ++i) {
                        Div d = outDivisors[i];
                        Div r = bigu::mod_fast<kBigLimbs, kDivLimbs>(remaining, d);
                        if (d > oneDiv && r.is_zero()) {
                            foundAny = true;

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

                            // Remaining changed; restart scanning from the beginning.
                            break;
                        }
                    }
                    if (foundAny) {
                        break;
                    }
                }

                start.add_u64_inplace(chunkWheelBlocks);
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