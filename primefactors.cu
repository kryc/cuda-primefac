#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <math.h>
#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

static bool parse_u128_dec(const char* s, __uint128_t& out)
{
    if (s == nullptr || *s == '\0') {
        return false;
    }
    __uint128_t value = 0;
    for (const unsigned char* p = (const unsigned char*)s; *p; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }
        uint32_t digit = (uint32_t)(*p - '0');
        // Check overflow for value = value*10 + digit.
        if (value > (((__uint128_t)(-1) - digit) / 10)) {
            return false;
        }
        value = value * 10 + digit;
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

static inline uint64_t ceil_div_u128_to_u64(__uint128_t a, uint64_t b)
{
    // Returns ceil(a / b) as uint64_t (assumes result fits in uint64_t).
    return (uint64_t)((a + (b - 1)) / b);
}

static uint64_t isqrt_u128(__uint128_t n)
{
    // Exact floor(sqrt(n)) for n up to 2^128-1.
    uint64_t lo = 0;
    uint64_t hi = ~0ull;
    uint64_t ans = 0;
    while (lo <= hi) {
        uint64_t mid = lo + ((hi - lo) >> 1);
        __uint128_t sq = (__uint128_t)mid * (__uint128_t)mid;
        if (sq <= n) {
            ans = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return ans;
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

static std::string u128_to_string(__uint128_t v)
{
    if (v == 0) {
        return "0";
    }
    std::string s;
    while (v > 0) {
        uint32_t digit = (uint32_t)(v % 10);
        s.push_back(char('0' + digit));
        v /= 10;
    }
    std::reverse(s.begin(), s.end());
    return s;
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
void factorize_chunk(__uint128_t n,
                     uint64_t sqrtn,
                     uint64_t startWheelBlock,
                     uint64_t endWheelBlock,
                     uint32_t* outCount,
                     uint64_t* outDivisors,
                     uint32_t outCapacity)
{
    uint64_t idx = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

    for (uint64_t wheelBlock = startWheelBlock + idx; wheelBlock < endWheelBlock; wheelBlock += stride) {
        __uint128_t value = (__uint128_t)wheelBlock * kWheelMod + 1;
        // One full wheel traversal covers exactly (wheelBlock*kWheelMod + 1) .. +kWheelMod.
        for (uint32_t i = 0; i < kWheelLen510510; ++i) {
            if (value > sqrtn) {
                return;
            }
            if (value > 1 && (n % value) == 0) {
                uint32_t pos = atomicAdd(outCount, 1u);
                if (pos < outCapacity) {
                    outDivisors[pos] = (uint64_t)value;
                }
                return;
            }
            value += wheelGap510510(i);
        }
    }
}
 
int main(int argc, char** argv)
{
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <N (decimal, >=2)>" << std::endl;
        return 1;
    }

    __uint128_t N;
    if (!parse_u128_dec(argv[1], N) || N < 2) {
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

    // CPU-side factorization loop.
    std::vector<__uint128_t> factors;
    __uint128_t remaining = N;

    auto divide_out = [&](uint64_t p) {
        while (p > 1 && (remaining % p) == 0) {
            factors.push_back((__uint128_t)p);
            remaining /= p;
        }
    };

    // Always handle wheel primes on CPU; the wheel never tests them.
    divide_out(2);
    divide_out(3);
    divide_out(5);
    divide_out(7);
    divide_out(11);
    divide_out(13);
    divide_out(17);

    const int blockSize = 256;
    const uint32_t outCapacity = 256; // max divisors to report per chunk (demo)
    uint32_t* outCount = nullptr;
    uint64_t* outDivisors = nullptr;
    checkCuda(cudaMallocManaged(&outCount, sizeof(uint32_t)), "cudaMallocManaged(outCount)");
    checkCuda(cudaMallocManaged(&outDivisors, sizeof(uint64_t) * outCapacity), "cudaMallocManaged(outDivisors)");

    // Tune chunk size: make each thread process multiple wheel blocks per launch.
    const uint32_t maxGridBlocks = 65535;
    const uint32_t gridBlocksForChunk = 256; // fixed grid size for consistent occupancy
    const uint64_t threadsPerLaunch = (uint64_t)gridBlocksForChunk * (uint64_t)blockSize;
    const uint64_t wheelBlocksPerThread = 8; // each thread processes ~8 wheel blocks per launch
    const uint64_t chunkWheelBlocks = threadsPerLaunch * wheelBlocksPerThread;

    while (remaining > 1) {
        uint64_t sqrtn = isqrt_u128(remaining);
        uint64_t totalWheelBlocks = ceil_div_u128_to_u64((__uint128_t)sqrtn, (uint64_t)kWheelMod);

        bool foundAny = false;
        for (uint64_t start = 0; start < totalWheelBlocks && remaining > 1; start += chunkWheelBlocks) {
            uint64_t end = std::min<uint64_t>(start + chunkWheelBlocks, totalWheelBlocks);

            {
                __uint128_t candLo = (__uint128_t)start * (__uint128_t)kWheelMod + 1;
                __uint128_t candHi = (__uint128_t)end * (__uint128_t)kWheelMod;
                std::cout << "\rScanning blocks [" << start << "," << end << ")"
                          << " candidates ~[" << u128_to_string(candLo) << "," << u128_to_string(candHi) << "]"
                          << " (rem=" << u128_to_string(remaining) << ")"
                          << std::flush;
            }

            *outCount = 0;
            // Ensure the GPU sees outCount reset before the kernel reads/writes it.
            checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(pre-launch)");

            factorize_chunk<<<(uint32_t)std::min<uint64_t>(maxGridBlocks, gridBlocksForChunk), blockSize>>>(remaining, sqrtn, start, end, outCount, outDivisors, outCapacity);
            checkCuda(cudaGetLastError(), "kernel launch");
            checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(post-kernel)");

            std::cout << "\rScanned blocks [" << start << "," << end << ")"
                      << " found " << *outCount << " divisors"
                      << " (rem=" << u128_to_string(remaining) << ")"
                      << std::endl;
            
            uint32_t found = *outCount;
            if (found > 0) {
                foundAny = true;
                uint32_t toProcess = std::min(found, outCapacity);
                for (uint32_t i = 0; i < toProcess; ++i) {
                    uint64_t d = outDivisors[i];
                    if (d > 1 && (remaining % d) == 0) {
                        std::cout << "GPU found divisor: " << d << std::endl;

                        // The GPU can return a composite divisor. Fully split it on the CPU,
                        // and divide those prime factors out of the remaining value.
                        std::vector<uint64_t> pf;
                        factor_u64(d, pf);
                        for (uint64_t p : pf) {
                            divide_out(p);
                        }
                    }
                }

                // Remaining changed; restart scanning from the beginning.
                break;
            }
        }

        if (!foundAny) {
            // No divisors <= sqrt(remaining) were found, so remaining is prime.
            factors.push_back(remaining);
            remaining = 1;
        }
    }

    for (auto f : factors) {
        std::cout << "factor: " << u128_to_string(f) << std::endl;
    }

    checkCuda(cudaFree(outDivisors), "cudaFree(outDivisors)");
    checkCuda(cudaFree(outCount), "cudaFree(outCount)");
    return 0;
}