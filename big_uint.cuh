#pragma once

#include <cstddef>
#include <cstdint>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

namespace bigu {

__host__ __device__ static inline uint32_t clz_u64(uint64_t x)
{
#if defined(__CUDA_ARCH__)
    return x ? (uint32_t)__clzll((unsigned long long)x) : 64u;
#else
    return x ? (uint32_t)__builtin_clzll((unsigned long long)x) : 64u;
#endif
}

__host__ __device__ static inline void mul_u64_u64(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo)
{
    // Portable 64x64 -> 128 using 32-bit partial products.
    const uint64_t a0 = (uint64_t)(uint32_t)a;
    const uint64_t a1 = a >> 32;
    const uint64_t b0 = (uint64_t)(uint32_t)b;
    const uint64_t b1 = b >> 32;

    const uint64_t p00 = a0 * b0;
    const uint64_t p01 = a0 * b1;
    const uint64_t p10 = a1 * b0;
    const uint64_t p11 = a1 * b1;

    const uint64_t middle = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;

    lo = (p00 & 0xffffffffull) | (middle << 32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (middle >> 32);
}

template <size_t LIMBS>
struct BigUInt {
    static_assert(LIMBS > 0, "LIMBS must be > 0");

    uint64_t limb[LIMBS]; // little-endian (limb[0] is least significant)

    __host__ __device__ constexpr BigUInt() : limb{} {}

    __host__ __device__ static constexpr BigUInt zero() { return BigUInt(); }

    __host__ __device__ static constexpr BigUInt from_u64(uint64_t v)
    {
        BigUInt x;
        x.limb[0] = v;
        return x;
    }

    __host__ __device__ bool is_zero() const
    {
        for (size_t i = 0; i < LIMBS; ++i) {
            if (limb[i] != 0) {
                return false;
            }
        }
        return true;
    }

    __host__ __device__ bool fits_u64(uint64_t& out) const
    {
        for (size_t i = 1; i < LIMBS; ++i) {
            if (limb[i] != 0) {
                return false;
            }
        }
        out = limb[0];
        return true;
    }

    __host__ __device__ int cmp(const BigUInt& other) const
    {
        for (size_t i = LIMBS; i-- > 0;) {
            if (limb[i] < other.limb[i]) {
                return -1;
            }
            if (limb[i] > other.limb[i]) {
                return 1;
            }
        }
        return 0;
    }

    __host__ __device__ bool operator==(const BigUInt& other) const { return cmp(other) == 0; }
    __host__ __device__ bool operator!=(const BigUInt& other) const { return cmp(other) != 0; }
    __host__ __device__ bool operator<(const BigUInt& other) const { return cmp(other) < 0; }
    __host__ __device__ bool operator<=(const BigUInt& other) const { return cmp(other) <= 0; }
    __host__ __device__ bool operator>(const BigUInt& other) const { return cmp(other) > 0; }
    __host__ __device__ bool operator>=(const BigUInt& other) const { return cmp(other) >= 0; }

    __host__ __device__ uint32_t bit_length() const
    {
        for (size_t i = LIMBS; i-- > 0;) {
            uint64_t w = limb[i];
            if (w != 0) {
                const uint32_t lead = clz_u64(w);
                return (uint32_t)(i * 64u + (64u - lead));
            }
        }
        return 0;
    }

    __host__ __device__ bool get_bit(uint32_t bitIndex) const
    {
        const uint32_t word = bitIndex >> 6;
        const uint32_t bit = bitIndex & 63u;
        if (word >= LIMBS) {
            return false;
        }
        return ((limb[word] >> bit) & 1ull) != 0ull;
    }

    __host__ __device__ void set_bit(uint32_t bitIndex)
    {
        const uint32_t word = bitIndex >> 6;
        const uint32_t bit = bitIndex & 63u;
        if (word < LIMBS) {
            limb[word] |= (1ull << bit);
        }
    }

    __host__ __device__ void shl1_inplace()
    {
        uint64_t carry = 0;
        for (size_t i = 0; i < LIMBS; ++i) {
            uint64_t newCarry = limb[i] >> 63;
            limb[i] = (limb[i] << 1) | carry;
            carry = newCarry;
        }
    }

    __host__ __device__ void shr1_inplace()
    {
        uint64_t carry = 0;
        for (size_t i = LIMBS; i-- > 0;) {
            uint64_t newCarry = limb[i] << 63;
            limb[i] = (limb[i] >> 1) | carry;
            carry = newCarry;
        }
    }

    __host__ __device__ BigUInt shl_bits(uint32_t bits) const
    {
        if (bits == 0) {
            return *this;
        }
        BigUInt out;
        const uint32_t wordShift = bits >> 6;
        const uint32_t bitShift = bits & 63u;

        for (size_t i = 0; i < LIMBS; ++i) {
            out.limb[i] = 0;
        }
        for (size_t i = 0; i < LIMBS; ++i) {
            const size_t dst = i + wordShift;
            if (dst >= LIMBS) {
                break;
            }
            uint64_t v = limb[i];
            if (bitShift == 0) {
                out.limb[dst] |= v;
            } else {
                out.limb[dst] |= (v << bitShift);
                if (dst + 1 < LIMBS) {
                    out.limb[dst + 1] |= (v >> (64u - bitShift));
                }
            }
        }
        return out;
    }

    __host__ __device__ BigUInt shr_bits(uint32_t bits) const
    {
        if (bits == 0) {
            return *this;
        }
        BigUInt out;
        const uint32_t wordShift = bits >> 6;
        const uint32_t bitShift = bits & 63u;

        for (size_t i = 0; i < LIMBS; ++i) {
            out.limb[i] = 0;
        }
        for (size_t i = wordShift; i < LIMBS; ++i) {
            const size_t dst = i - wordShift;
            uint64_t v = limb[i];
            if (bitShift == 0) {
                out.limb[dst] |= v;
            } else {
                out.limb[dst] |= (v >> bitShift);
                if (i + 1 < LIMBS) {
                    out.limb[dst] |= (limb[i + 1] << (64u - bitShift));
                }
            }
        }
        return out;
    }

    __host__ __device__ bool add_inplace(const BigUInt& other)
    {
        uint64_t carry = 0;
        for (size_t i = 0; i < LIMBS; ++i) {
            uint64_t a = limb[i];
            uint64_t b = other.limb[i];
            uint64_t sum = a + b;
            uint64_t carry1 = (sum < a) ? 1ull : 0ull;
            uint64_t sum2 = sum + carry;
            uint64_t carry2 = (sum2 < sum) ? 1ull : 0ull;
            limb[i] = sum2;
            carry = (carry1 | carry2);
        }
        return carry != 0;
    }

    __host__ __device__ bool add_u64_inplace(uint64_t v)
    {
        uint64_t carry = v;
        for (size_t i = 0; i < LIMBS && carry != 0; ++i) {
            uint64_t prev = limb[i];
            limb[i] = prev + carry;
            carry = (limb[i] < prev) ? 1ull : 0ull;
        }
        return carry != 0;
    }

    __host__ __device__ bool add_u32_inplace(uint32_t v) { return add_u64_inplace((uint64_t)v); }

    __host__ __device__ void sub_inplace(const BigUInt& other)
    {
        // Requires *this >= other.
        uint64_t borrow = 0;
        for (size_t i = 0; i < LIMBS; ++i) {
            uint64_t a = limb[i];
            uint64_t b = other.limb[i];
            uint64_t t = a - b;
            uint64_t borrow1 = (t > a) ? 1ull : 0ull;
            uint64_t t2 = t - borrow;
            uint64_t borrow2 = (t2 > t) ? 1ull : 0ull;
            limb[i] = t2;
            borrow = (borrow1 | borrow2);
        }
    }

    __host__ __device__ bool mul_u64_inplace(uint64_t m)
    {
        uint64_t carry = 0;
        for (size_t i = 0; i < LIMBS; ++i) {
            uint64_t hi, lo;
            mul_u64_u64(limb[i], m, hi, lo);
            uint64_t lo2 = lo + carry;
            uint64_t carryOut = (lo2 < lo) ? 1ull : 0ull;
            limb[i] = lo2;
            carry = hi + carryOut;
        }
        return carry != 0;
    }

    __host__ __device__ bool mul_u32_inplace(uint32_t m) { return mul_u64_inplace((uint64_t)m); }

    __host__ __device__ uint32_t div_mod_u32_inplace(uint32_t d)
    {
        // Divides by 32-bit divisor. Returns remainder.
        // Uses 32-bit long division per 64-bit limb.
        uint64_t rem = 0;
        for (size_t i = LIMBS; i-- > 0;) {
            const uint64_t w = limb[i];
            const uint64_t hi32 = w >> 32;
            const uint64_t lo32 = (uint32_t)w;

            uint64_t part1 = (rem << 32) | hi32;
            uint64_t q1 = part1 / d;
            uint64_t r1 = part1 - q1 * d;

            uint64_t part2 = (r1 << 32) | lo32;
            uint64_t q0 = part2 / d;
            uint64_t r0 = part2 - q0 * d;

            limb[i] = (q1 << 32) | q0;
            rem = r0;
        }
        return (uint32_t)rem;
    }
};

template <size_t LIMBS>
__host__ __device__ static inline BigUInt<LIMBS> add_u32(const BigUInt<LIMBS>& a, uint32_t v)
{
    BigUInt<LIMBS> out = a;
    out.add_u32_inplace(v);
    return out;
}

template <size_t LIMBS>
__host__ __device__ static inline BigUInt<LIMBS> mul_u32(const BigUInt<LIMBS>& a, uint32_t m)
{
    BigUInt<LIMBS> out = a;
    out.mul_u32_inplace(m);
    return out;
}

template <size_t LIMBS>
__host__ __device__ static inline BigUInt<LIMBS> mod_big(const BigUInt<LIMBS>& a, const BigUInt<LIMBS>& m)
{
    // Binary long division remainder (shift/subtract). Correct but not fast.
    if (m.is_zero()) {
        return BigUInt<LIMBS>::zero();
    }
    if (a < m) {
        return a;
    }

    BigUInt<LIMBS> rem = a;
    const uint32_t aBits = rem.bit_length();
    const uint32_t mBits = m.bit_length();
    if (mBits == 0) {
        return BigUInt<LIMBS>::zero();
    }
    int32_t shift = (int32_t)aBits - (int32_t)mBits;
    BigUInt<LIMBS> denom = m.shl_bits((uint32_t)shift);

    for (int32_t i = shift; i >= 0; --i) {
        if (rem >= denom) {
            rem.sub_inplace(denom);
        }
        denom.shr1_inplace();
    }

    return rem;
}

template <size_t LIMBS>
__host__ __device__ static inline void div_mod_big(const BigUInt<LIMBS>& numerator,
                                                  const BigUInt<LIMBS>& denom,
                                                  BigUInt<LIMBS>& quotient,
                                                  BigUInt<LIMBS>& remainder)
{
    quotient = BigUInt<LIMBS>::zero();
    if (denom.is_zero()) {
        remainder = BigUInt<LIMBS>::zero();
        return;
    }
    if (numerator < denom) {
        remainder = numerator;
        return;
    }

    remainder = numerator;
    const uint32_t nBits = remainder.bit_length();
    const uint32_t dBits = denom.bit_length();
    int32_t shift = (int32_t)nBits - (int32_t)dBits;
    BigUInt<LIMBS> d = denom.shl_bits((uint32_t)shift);

    for (int32_t i = shift; i >= 0; --i) {
        if (remainder >= d) {
            remainder.sub_inplace(d);
            quotient.set_bit((uint32_t)i);
        }
        d.shr1_inplace();
    }
}

template <size_t LIMBS>
__host__ __device__ static inline BigUInt<LIMBS> isqrt_big(const BigUInt<LIMBS>& n)
{
    // Digit-by-digit integer sqrt (binary restoring method), adapted for big integers.
    // Works for all widths; returns floor(sqrt(n)).

    if (n.is_zero()) {
        return BigUInt<LIMBS>::zero();
    }

    BigUInt<LIMBS> x = n;
    BigUInt<LIMBS> res = BigUInt<LIMBS>::zero();

    // bit = highest power of four <= n
    uint32_t bl = x.bit_length();
    uint32_t bitPos = (bl == 0) ? 0 : (bl - 1);
    bitPos &= ~1u; // make even

    BigUInt<LIMBS> bit = BigUInt<LIMBS>::zero();
    bit.set_bit(bitPos);

    while (!bit.is_zero()) {
        BigUInt<LIMBS> t = res;
        t.add_inplace(bit);
        if (x >= t) {
            x.sub_inplace(t);
            res.shr1_inplace();
            res.add_inplace(bit);
        } else {
            res.shr1_inplace();
        }
        bit = bit.shr_bits(2);
    }

    return res;
}

} // namespace bigu
