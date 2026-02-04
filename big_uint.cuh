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
    // 64x64 -> 128.
#if defined(__CUDA_ARCH__)
    lo = (uint64_t)((unsigned long long)a * (unsigned long long)b);
    hi = (uint64_t)__umul64hi((unsigned long long)a, (unsigned long long)b);
#else
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
#endif
}

template <size_t NL>
__host__ __device__ static inline uint64_t shl64(uint64_t x, uint32_t s)
{
    return (s == 0) ? x : (x << s);
}

__host__ __device__ static inline uint64_t shr64(uint64_t x, uint32_t s)
{
    return (s == 0) ? x : (x >> s);
}

__host__ __device__ static inline uint64_t udiv128by64(uint64_t u1, uint64_t u0, uint64_t v, uint64_t& r)
{
    // Divide 128-bit (u1:u0) by 64-bit v. Returns 64-bit quotient and remainder.
    // Uses base 2^32 Knuth division for a 4-digit numerator by 2-digit divisor.
    // Preconditions: v != 0, and (for a 64-bit quotient) u1 < v.

    if (v == 0) {
        r = 0;
        return 0;
    }
    if (u1 == 0) {
        r = u0 % v;
        return u0 / v;
    }

    const uint32_t v1 = (uint32_t)(v >> 32);
    const uint32_t v0 = (uint32_t)v;
    if (v1 == 0) {
        // Divisor fits in 32 bits; do 128/32 division.
        const uint32_t d = v0;
        uint32_t un[4] = {(uint32_t)u0, (uint32_t)(u0 >> 32), (uint32_t)u1, (uint32_t)(u1 >> 32)};
        uint64_t rem = 0;
        uint32_t qd[4] = {};
        for (int i = 3; i >= 0; --i) {
            uint64_t cur = (rem << 32) | un[i];
            uint64_t qi = cur / d;
            rem = cur - qi * d;
            qd[i] = (uint32_t)qi;
        }
        r = rem;
        // For our usage (u1 < v), the quotient fits in 64 bits, i.e. qd[3:2] should be 0.
        return ((uint64_t)qd[1] << 32) | qd[0];
    }

    // Normalize so highest bit of v1 is set.
    uint32_t s = 0;
    {
        uint32_t lead = v1;
#if defined(__CUDA_ARCH__)
        s = lead ? (uint32_t)__clz(lead) : 32u;
#else
        s = lead ? (uint32_t)__builtin_clz(lead) : 32u;
#endif
    }
    const uint64_t vn = (s == 0) ? v : (v << s);
    const uint32_t vn1 = (uint32_t)(vn >> 32);
    const uint32_t vn0 = (uint32_t)vn;

    // Build normalized numerator digits un[0..4] in base 2^32 (little endian).
    const uint32_t u3 = (uint32_t)(u1 >> 32);
    const uint32_t u2 = (uint32_t)u1;
    const uint32_t u1d = (uint32_t)(u0 >> 32);
    const uint32_t u0d = (uint32_t)u0;

    uint32_t un[5] = {};
    if (s == 0) {
        un[0] = u0d;
        un[1] = u1d;
        un[2] = u2;
        un[3] = u3;
        un[4] = 0;
    } else {
        const uint32_t rs = 32u - s;
        un[0] = u0d << s;
        un[1] = (u1d << s) | (u0d >> rs);
        un[2] = (u2 << s) | (u1d >> rs);
        un[3] = (u3 << s) | (u2 >> rs);
        un[4] = (u3 >> rs);
    }

    auto submul2 = [&](int j, uint32_t qhat) -> bool {
        // Subtract qhat * (vn1:vn0) from un[j+2..j] (3 digits). Returns true if borrow.
        uint64_t p0 = (uint64_t)qhat * (uint64_t)vn0;
        uint64_t p1 = (uint64_t)qhat * (uint64_t)vn1;
        uint64_t sub0 = ((uint64_t)un[j] - (p0 & 0xffffffffull));
        uint64_t borrow0 = (sub0 >> 63) & 1ull;
        un[j] = (uint32_t)sub0;

        uint64_t sub1 = (uint64_t)un[j + 1] - (uint32_t)(p0 >> 32) - (uint32_t)p1 - borrow0;
        uint64_t borrow1 = (sub1 >> 63) & 1ull;
        un[j + 1] = (uint32_t)sub1;

        uint64_t sub2 = (uint64_t)un[j + 2] - (uint32_t)(p1 >> 32) - borrow1;
        uint64_t borrow2 = (sub2 >> 63) & 1ull;
        un[j + 2] = (uint32_t)sub2;
        return borrow2 != 0;
    };

    auto addback2 = [&](int j) {
        uint64_t sum = (uint64_t)un[j] + vn0;
        un[j] = (uint32_t)sum;
        uint64_t carry = sum >> 32;
        sum = (uint64_t)un[j + 1] + vn1 + carry;
        un[j + 1] = (uint32_t)sum;
        carry = sum >> 32;
        un[j + 2] = (uint32_t)((uint64_t)un[j + 2] + carry);
    };

    uint32_t qd[3] = {};
    for (int j = 2; j >= 0; --j) {
        uint64_t numerator2 = ((uint64_t)un[j + 2] << 32) | (uint64_t)un[j + 1];
        uint64_t qhat = numerator2 / vn1;
        uint64_t rhat = numerator2 - qhat * vn1;

        while (qhat >= 0x100000000ull || (qhat * vn0) > ((rhat << 32) | un[j])) {
            qhat -= 1;
            rhat += vn1;
            if (rhat >= 0x100000000ull) {
                break;
            }
        }

        bool borrow = submul2(j, (uint32_t)qhat);
        if (borrow) {
            qhat -= 1;
            addback2(j);
        }
        qd[j] = (uint32_t)qhat;
    }

    // Denormalize remainder (two digits un[1:0]).
    uint64_t remn = ((uint64_t)un[1] << 32) | un[0];
    r = (s == 0) ? remn : (remn >> s);

    // Quotient is qd[2:0] in base 2^32. For our intended use, qd[2] should be 0.
    return ((uint64_t)qd[1] << 32) | (uint64_t)qd[0];
}

template <size_t N>
__host__ __device__ static inline void shl_array(uint64_t (&out)[N], const uint64_t (&in)[N], uint32_t s)
{
    if (s == 0) {
        for (size_t i = 0; i < N; ++i) out[i] = in[i];
        return;
    }
    uint64_t carry = 0;
    for (size_t i = 0; i < N; ++i) {
        uint64_t w = in[i];
        out[i] = (w << s) | carry;
        carry = (w >> (64u - s));
    }
}

template <size_t N>
__host__ __device__ static inline void shr_array(uint64_t (&out)[N], const uint64_t (&in)[N], uint32_t s)
{
    if (s == 0) {
        for (size_t i = 0; i < N; ++i) out[i] = in[i];
        return;
    }
    uint64_t carry = 0;
    for (size_t i = N; i-- > 0;) {
        uint64_t w = in[i];
        out[i] = (w >> s) | carry;
        carry = (w << (64u - s));
    }
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

template <size_t NL>
__host__ __device__ static inline uint64_t mod_u64(const BigUInt<NL>& numerator, uint64_t d)
{
    if (d == 0) return 0;
    uint64_t rem = 0;
    for (size_t i = NL; i-- > 0;) {
        uint64_t rr;
        (void)udiv128by64(rem, numerator.limb[i], d, rr);
        rem = rr;
    }
    return rem;
}

template <size_t NL>
__host__ __device__ static inline void div_mod_u64(const BigUInt<NL>& numerator,
                                                   uint64_t d,
                                                   BigUInt<NL>& quotient,
                                                   uint64_t& remainder)
{
    quotient = BigUInt<NL>::zero();
    remainder = 0;
    if (d == 0) return;

    uint64_t rem = 0;
    for (size_t i = NL; i-- > 0;) {
        uint64_t rr;
        uint64_t q = udiv128by64(rem, numerator.limb[i], d, rr);
        quotient.limb[i] = q;
        rem = rr;
    }
    remainder = rem;
}

template <size_t NL, size_t DL>
__host__ __device__ static inline BigUInt<DL> mod_knuth(const BigUInt<NL>& numerator, const BigUInt<DL>& denom)
{
    static_assert(NL >= 1 && DL >= 1, "invalid limb sizes");

    // denom==0 => 0 remainder
    bool denomZero = true;
    for (size_t i = 0; i < DL; ++i) {
        if (denom.limb[i] != 0) {
            denomZero = false;
            break;
        }
    }
    if (denomZero) {
        return BigUInt<DL>::zero();
    }

    // If numerator < denom, remainder is numerator truncated.
    // (This is safe for the uses in this project where denom is <= numerator width.)
    bool less = true;
    {
        // Compare using the overlapping highest limbs.
        const size_t top = (NL < DL) ? NL : DL;
        // If NL > DL, numerator cannot be < denom unless higher limbs are zero and overlap is <.
        if constexpr (NL > DL) {
            for (size_t i = NL; i-- > DL;) {
                if (numerator.limb[i] != 0) {
                    less = false;
                    break;
                }
            }
        }
        if (less) {
            for (size_t i = top; i-- > 0;) {
                uint64_t a = numerator.limb[i];
                uint64_t b = denom.limb[i];
                if (a < b) {
                    less = true;
                    break;
                }
                if (a > b) {
                    less = false;
                    break;
                }
            }
        }
    }
    if (less) {
        BigUInt<DL> r = BigUInt<DL>::zero();
        for (size_t i = 0; i < NL && i < DL; ++i) {
            r.limb[i] = numerator.limb[i];
        }
        return r;
    }

    // Strip leading zeros.
    size_t m = DL;
    while (m > 1 && denom.limb[m - 1] == 0) {
        --m;
    }
    size_t n = NL;
    while (n > 1 && numerator.limb[n - 1] == 0) {
        --n;
    }
    if (n < m) {
        BigUInt<DL> r = BigUInt<DL>::zero();
        for (size_t i = 0; i < n && i < DL; ++i) r.limb[i] = numerator.limb[i];
        return r;
    }

    // Fast path: 1-limb divisor.
    if (m == 1) {
        uint64_t d = denom.limb[0];
        uint64_t rem = 0;
        for (size_t i = n; i-- > 0;) {
            uint64_t rr;
            (void)udiv128by64(rem, numerator.limb[i], d, rr);
            rem = rr;
        }
        return BigUInt<DL>::from_u64(rem);
    }

    uint64_t v[DL] = {};
    uint64_t u[NL + 1] = {};

    const uint32_t s = clz_u64(denom.limb[m - 1]);
    {
        uint64_t vin[DL] = {};
        for (size_t i = 0; i < m; ++i) vin[i] = denom.limb[i];
        shl_array<DL>(v, vin, s);
    }
    {
        uint64_t uin[NL + 1] = {};
        for (size_t i = 0; i < n; ++i) uin[i] = numerator.limb[i];
        uin[n] = 0;
        shl_array<NL + 1>(u, uin, s);
    }

    const size_t jMax = n - m;
    for (size_t jj = 0; jj <= jMax; ++jj) {
        const size_t j = jMax - jj;

        uint64_t rhat = 0;
        uint64_t qhat = udiv128by64(u[j + m], u[j + m - 1], v[m - 1], rhat);

        // Refine qhat.
        if (qhat != 0) {
            while (true) {
                uint64_t pHi, pLo;
                mul_u64_u64(qhat, v[m - 2], pHi, pLo);
                if (pHi < rhat) {
                    break;
                }
                if (pHi > rhat || pLo > u[j + m - 2]) {
                    qhat -= 1;
                    uint64_t r2 = rhat + v[m - 1];
                    if (r2 < rhat) {
                        rhat = r2;
                        break;
                    }
                    rhat = r2;
                    continue;
                }
                break;
            }
        }

        // u[j..j+m] -= qhat * v[0..m-1]
        uint64_t borrow = 0;
        uint64_t carry = 0;
        for (size_t i = 0; i < m; ++i) {
            uint64_t pHi, pLo;
            mul_u64_u64(v[i], qhat, pHi, pLo);

            uint64_t t = pLo + carry;
            carry = pHi + ((t < pLo) ? 1ull : 0ull);

            uint64_t uu = u[j + i];
            uint64_t sub = uu - t;
            uint64_t b1 = (sub > uu) ? 1ull : 0ull;
            uint64_t sub2 = sub - borrow;
            uint64_t b2 = (sub2 > sub) ? 1ull : 0ull;
            u[j + i] = sub2;
            borrow = (b1 | b2);
        }
        {
            uint64_t uu = u[j + m];
            uint64_t sub = uu - carry;
            uint64_t b1 = (sub > uu) ? 1ull : 0ull;
            uint64_t sub2 = sub - borrow;
            uint64_t b2 = (sub2 > sub) ? 1ull : 0ull;
            u[j + m] = sub2;
            if ((b1 | b2) != 0) {
                // Add v back.
                uint64_t c = 0;
                for (size_t i = 0; i < m; ++i) {
                    uint64_t prev = u[j + i];
                    uint64_t sum = prev + v[i] + c;
                    c = (sum < prev) ? 1ull : 0ull;
                    // also carry from +v[i]
                    if (c == 0 && sum < v[i]) {
                        c = 1ull;
                    }
                    u[j + i] = sum;
                }
                u[j + m] += c;
            }
        }
    }

    BigUInt<DL> rem = BigUInt<DL>::zero();
    {
        uint64_t rarr[DL] = {};
        for (size_t i = 0; i < m; ++i) rarr[i] = u[i];
        uint64_t outArr[DL] = {};
        shr_array<DL>(outArr, rarr, s);
        for (size_t i = 0; i < m; ++i) rem.limb[i] = outArr[i];
    }
    return rem;
}

template <size_t NL, size_t DL>
__host__ __device__ static inline void div_mod_knuth(const BigUInt<NL>& numerator,
                                                     const BigUInt<DL>& denom,
                                                     BigUInt<NL>& quotient,
                                                     BigUInt<DL>& remainder)
{
    quotient = BigUInt<NL>::zero();
    remainder = BigUInt<DL>::zero();

    // denom==0 => {0,0}
    bool denomZero = true;
    for (size_t i = 0; i < DL; ++i) {
        if (denom.limb[i] != 0) {
            denomZero = false;
            break;
        }
    }
    if (denomZero) {
        return;
    }

    // Strip leading zeros.
    size_t m = DL;
    while (m > 1 && denom.limb[m - 1] == 0) --m;
    size_t n = NL;
    while (n > 1 && numerator.limb[n - 1] == 0) --n;
    if (n < m) {
        for (size_t i = 0; i < n && i < DL; ++i) remainder.limb[i] = numerator.limb[i];
        return;
    }

    if (m == 1) {
        uint64_t d = denom.limb[0];
        uint64_t rem = 0;
        for (size_t i = n; i-- > 0;) {
            uint64_t rr;
            uint64_t q = udiv128by64(rem, numerator.limb[i], d, rr);
            quotient.limb[i] = q;
            rem = rr;
        }
        remainder.limb[0] = rem;
        return;
    }

    uint64_t v[DL] = {};
    uint64_t u[NL + 1] = {};
    const uint32_t s = clz_u64(denom.limb[m - 1]);

    {
        uint64_t vin[DL] = {};
        for (size_t i = 0; i < m; ++i) vin[i] = denom.limb[i];
        shl_array<DL>(v, vin, s);
    }
    {
        uint64_t uin[NL + 1] = {};
        for (size_t i = 0; i < n; ++i) uin[i] = numerator.limb[i];
        uin[n] = 0;
        shl_array<NL + 1>(u, uin, s);
    }

    const size_t jMax = n - m;
    for (size_t jj = 0; jj <= jMax; ++jj) {
        const size_t j = jMax - jj;

        uint64_t rhat = 0;
        uint64_t qhat = udiv128by64(u[j + m], u[j + m - 1], v[m - 1], rhat);

        if (qhat != 0) {
            while (true) {
                uint64_t pHi, pLo;
                mul_u64_u64(qhat, v[m - 2], pHi, pLo);
                if (pHi < rhat) break;
                if (pHi > rhat || pLo > u[j + m - 2]) {
                    qhat -= 1;
                    uint64_t r2 = rhat + v[m - 1];
                    if (r2 < rhat) {
                        rhat = r2;
                        break;
                    }
                    rhat = r2;
                    continue;
                }
                break;
            }
        }

        uint64_t borrow = 0;
        uint64_t carry = 0;
        for (size_t i = 0; i < m; ++i) {
            uint64_t pHi, pLo;
            mul_u64_u64(v[i], qhat, pHi, pLo);
            uint64_t t = pLo + carry;
            carry = pHi + ((t < pLo) ? 1ull : 0ull);

            uint64_t uu = u[j + i];
            uint64_t sub = uu - t;
            uint64_t b1 = (sub > uu) ? 1ull : 0ull;
            uint64_t sub2 = sub - borrow;
            uint64_t b2 = (sub2 > sub) ? 1ull : 0ull;
            u[j + i] = sub2;
            borrow = (b1 | b2);
        }
        uint64_t uu = u[j + m];
        uint64_t sub = uu - carry;
        uint64_t b1 = (sub > uu) ? 1ull : 0ull;
        uint64_t sub2 = sub - borrow;
        uint64_t b2 = (sub2 > sub) ? 1ull : 0ull;
        u[j + m] = sub2;

        if ((b1 | b2) != 0) {
            qhat -= 1;
            uint64_t c = 0;
            for (size_t i = 0; i < m; ++i) {
                uint64_t prev = u[j + i];
                uint64_t sum = prev + v[i] + c;
                c = (sum < prev) ? 1ull : 0ull;
                if (c == 0 && sum < v[i]) c = 1ull;
                u[j + i] = sum;
            }
            u[j + m] += c;
        }

        if (j < NL) {
            quotient.limb[j] = qhat;
        }
    }

    {
        uint64_t rarr[DL] = {};
        for (size_t i = 0; i < m; ++i) rarr[i] = u[i];
        uint64_t outArr[DL] = {};
        shr_array<DL>(outArr, rarr, s);
        for (size_t i = 0; i < m; ++i) remainder.limb[i] = outArr[i];
    }
}

template <size_t NL, size_t DL>
__host__ __device__ static inline BigUInt<DL> mod_knuth_fixed_full(const BigUInt<NL>& numerator, const BigUInt<DL>& denom)
{
    // Fast path for common fixed-size cases where numerator and denom both use their full limb widths
    // (i.e. top limbs are non-zero). Uses Knuth Algorithm D with fixed m=DL, n=NL.
    static_assert(NL >= 2 && DL >= 2, "fixed division expects at least 2 limbs");

    const size_t m = DL;
    const size_t n = NL;

    uint64_t v[DL];
    uint64_t u[NL + 1];

    const uint32_t s = clz_u64(denom.limb[m - 1]);

    // Normalize v directly from denom.
    if (s == 0) {
#pragma unroll
        for (size_t i = 0; i < DL; ++i) {
            v[i] = denom.limb[i];
        }
    } else {
        uint64_t carry = 0;
#pragma unroll
        for (size_t i = 0; i < DL; ++i) {
            uint64_t w = denom.limb[i];
            v[i] = (w << s) | carry;
            carry = (w >> (64u - s));
        }
    }

    // Normalize u directly from numerator, with u[n]=0.
    if (s == 0) {
#pragma unroll
        for (size_t i = 0; i < NL; ++i) {
            u[i] = numerator.limb[i];
        }
        u[NL] = 0;
    } else {
        uint64_t carry = 0;
#pragma unroll
        for (size_t i = 0; i < NL; ++i) {
            uint64_t w = numerator.limb[i];
            u[i] = (w << s) | carry;
            carry = (w >> (64u - s));
        }
        u[NL] = carry;
    }

    const size_t jMax = n - m;
    for (size_t jj = 0; jj <= jMax; ++jj) {
        const size_t j = jMax - jj;

        uint64_t rhat = 0;
        uint64_t qhat = udiv128by64(u[j + m], u[j + m - 1], v[m - 1], rhat);

        if (qhat != 0) {
            while (true) {
                uint64_t pHi, pLo;
                mul_u64_u64(qhat, v[m - 2], pHi, pLo);
                if (pHi < rhat) {
                    break;
                }
                if (pHi > rhat || pLo > u[j + m - 2]) {
                    qhat -= 1;
                    uint64_t r2 = rhat + v[m - 1];
                    if (r2 < rhat) {
                        rhat = r2;
                        break;
                    }
                    rhat = r2;
                    continue;
                }
                break;
            }
        }

        uint64_t borrow = 0;
        uint64_t carryMul = 0;
#pragma unroll
        for (size_t i = 0; i < DL; ++i) {
            uint64_t pHi, pLo;
            mul_u64_u64(v[i], qhat, pHi, pLo);

            uint64_t t = pLo + carryMul;
            carryMul = pHi + ((t < pLo) ? 1ull : 0ull);

            uint64_t uu = u[j + i];
            uint64_t sub = uu - t;
            uint64_t b1 = (sub > uu) ? 1ull : 0ull;
            uint64_t sub2 = sub - borrow;
            uint64_t b2 = (sub2 > sub) ? 1ull : 0ull;
            u[j + i] = sub2;
            borrow = (b1 | b2);
        }

        uint64_t uu = u[j + m];
        uint64_t sub = uu - carryMul;
        uint64_t b1 = (sub > uu) ? 1ull : 0ull;
        uint64_t sub2 = sub - borrow;
        uint64_t b2 = (sub2 > sub) ? 1ull : 0ull;
        u[j + m] = sub2;

        if ((b1 | b2) != 0) {
            uint64_t c = 0;
#pragma unroll
            for (size_t i = 0; i < DL; ++i) {
                uint64_t prev = u[j + i];
                uint64_t sum = prev + v[i] + c;
                c = (sum < prev) ? 1ull : 0ull;
                if (c == 0 && sum < v[i]) c = 1ull;
                u[j + i] = sum;
            }
            u[j + m] += c;
        }
    }

    BigUInt<DL> rem = BigUInt<DL>::zero();
    // Denormalize the low m limbs of u into rem.
    if (s == 0) {
#pragma unroll
        for (size_t i = 0; i < DL; ++i) {
            rem.limb[i] = u[i];
        }
    } else {
        uint64_t carry = 0;
        for (size_t ii = 0; ii < DL; ++ii) {
            size_t i = DL - 1 - ii;
            uint64_t w = u[i];
            rem.limb[i] = (w >> s) | carry;
            carry = (w << (64u - s));
        }
    }
    return rem;
}

template <size_t NL, size_t DL>
__host__ __device__ static inline BigUInt<DL> mod_knuth_fixed_full_streaming(const BigUInt<NL>& numerator,
                                                                            const BigUInt<DL>& denom)
{
    // Fixed-size remainder using a sliding (m+1)-limb window over u, to reduce scratch.
    // Preconditions: numerator.limb[NL-1] != 0 and denom.limb[DL-1] != 0.
    static_assert(NL >= 2 && DL >= 2, "fixed division expects at least 2 limbs");

    const size_t m = DL;
    const size_t n = NL;

    uint64_t v[DL];
    uint64_t uwin[DL + 1];

    const uint32_t s = clz_u64(denom.limb[m - 1]);
    const uint32_t rs = (s == 0) ? 0u : (64u - s);

    auto den_norm = [&](size_t i) -> uint64_t {
        if (s == 0) return denom.limb[i];
        if (i == 0) return denom.limb[0] << s;
        return (denom.limb[i] << s) | (denom.limb[i - 1] >> rs);
    };
    auto num_norm = [&](size_t i) -> uint64_t {
        if (s == 0) {
            if (i == n) return 0ull;
            return numerator.limb[i];
        }
        if (i == 0) return numerator.limb[0] << s;
        if (i < n) return (numerator.limb[i] << s) | (numerator.limb[i - 1] >> rs);
        // i==n
        return numerator.limb[n - 1] >> rs;
    };

#pragma unroll
    for (size_t i = 0; i < DL; ++i) {
        v[i] = den_norm(i);
    }

    const size_t jMax = n - m;

    // Initialize window to u[jMax..jMax+m] == u[n-m..n].
#pragma unroll
    for (size_t i = 0; i < DL + 1; ++i) {
        uwin[i] = num_norm(jMax + i);
    }

    for (size_t jj = 0; jj <= jMax; ++jj) {
        const size_t j = jMax - jj;

        uint64_t rhat = 0;
        uint64_t qhat = udiv128by64(uwin[m], uwin[m - 1], v[m - 1], rhat);

        if (qhat != 0) {
            while (true) {
                uint64_t pHi, pLo;
                mul_u64_u64(qhat, v[m - 2], pHi, pLo);
                if (pHi < rhat) break;
                if (pHi > rhat || pLo > uwin[m - 2]) {
                    qhat -= 1;
                    uint64_t r2 = rhat + v[m - 1];
                    if (r2 < rhat) {
                        rhat = r2;
                        break;
                    }
                    rhat = r2;
                    continue;
                }
                break;
            }
        }

        uint64_t borrow = 0;
        uint64_t carryMul = 0;
#pragma unroll
        for (size_t i = 0; i < DL; ++i) {
            uint64_t pHi, pLo;
            mul_u64_u64(v[i], qhat, pHi, pLo);

            uint64_t t = pLo + carryMul;
            carryMul = pHi + ((t < pLo) ? 1ull : 0ull);

            uint64_t uu = uwin[i];
            uint64_t sub = uu - t;
            uint64_t b1 = (sub > uu) ? 1ull : 0ull;
            uint64_t sub2 = sub - borrow;
            uint64_t b2 = (sub2 > sub) ? 1ull : 0ull;
            uwin[i] = sub2;
            borrow = (b1 | b2);
        }

        {
            uint64_t uu = uwin[m];
            uint64_t sub = uu - carryMul;
            uint64_t b1 = (sub > uu) ? 1ull : 0ull;
            uint64_t sub2 = sub - borrow;
            uint64_t b2 = (sub2 > sub) ? 1ull : 0ull;
            uwin[m] = sub2;

            if ((b1 | b2) != 0) {
                uint64_t c = 0;
#pragma unroll
                for (size_t i = 0; i < DL; ++i) {
                    uint64_t prev = uwin[i];
                    uint64_t sum = prev + v[i] + c;
                    c = (sum < prev) ? 1ull : 0ull;
                    if (c == 0 && sum < v[i]) c = 1ull;
                    uwin[i] = sum;
                }
                uwin[m] += c;
            }
        }

        if (j > 0) {
#pragma unroll
            for (size_t i = DL; i-- > 0;) {
                uwin[i + 1] = uwin[i];
            }
            uwin[0] = num_norm(j - 1);
        }
    }

    BigUInt<DL> rem = BigUInt<DL>::zero();
    if (s == 0) {
#pragma unroll
        for (size_t i = 0; i < DL; ++i) {
            rem.limb[i] = uwin[i];
        }
        return rem;
    }

    // Denormalize: shift right by s within the m-limb remainder (do NOT use uwin[m]).
#pragma unroll
    for (size_t i = 0; i < DL; ++i) {
        uint64_t hi = (i + 1 < DL) ? uwin[i + 1] : 0ull;
        rem.limb[i] = (uwin[i] >> s) | (hi << rs);
    }
    return rem;
}

template <size_t NL, size_t DL>
__host__ __device__ static inline void div_mod_knuth_fixed_full(const BigUInt<NL>& numerator,
                                                                const BigUInt<DL>& denom,
                                                                BigUInt<NL>& quotient,
                                                                BigUInt<DL>& remainder)
{
    // Fixed-size version mirroring mod_knuth_fixed_full, recording qhat into quotient.
    quotient = BigUInt<NL>::zero();
    remainder = BigUInt<DL>::zero();

    const size_t m = DL;
    const size_t n = NL;

    uint64_t v[DL];
    uint64_t u[NL + 1];

    const uint32_t s = clz_u64(denom.limb[m - 1]);

    if (s == 0) {
#pragma unroll
        for (size_t i = 0; i < DL; ++i) v[i] = denom.limb[i];
    } else {
        uint64_t carry = 0;
#pragma unroll
        for (size_t i = 0; i < DL; ++i) {
            uint64_t w = denom.limb[i];
            v[i] = (w << s) | carry;
            carry = (w >> (64u - s));
        }
    }

    if (s == 0) {
#pragma unroll
        for (size_t i = 0; i < NL; ++i) u[i] = numerator.limb[i];
        u[NL] = 0;
    } else {
        uint64_t carry = 0;
#pragma unroll
        for (size_t i = 0; i < NL; ++i) {
            uint64_t w = numerator.limb[i];
            u[i] = (w << s) | carry;
            carry = (w >> (64u - s));
        }
        u[NL] = carry;
    }

    const size_t jMax = n - m;
    for (size_t jj = 0; jj <= jMax; ++jj) {
        const size_t j = jMax - jj;

        uint64_t rhat = 0;
        uint64_t qhat = udiv128by64(u[j + m], u[j + m - 1], v[m - 1], rhat);

        if (qhat != 0) {
            while (true) {
                uint64_t pHi, pLo;
                mul_u64_u64(qhat, v[m - 2], pHi, pLo);
                if (pHi < rhat) break;
                if (pHi > rhat || pLo > u[j + m - 2]) {
                    qhat -= 1;
                    uint64_t r2 = rhat + v[m - 1];
                    if (r2 < rhat) {
                        rhat = r2;
                        break;
                    }
                    rhat = r2;
                    continue;
                }
                break;
            }
        }

        uint64_t borrow = 0;
        uint64_t carryMul = 0;
#pragma unroll
        for (size_t i = 0; i < DL; ++i) {
            uint64_t pHi, pLo;
            mul_u64_u64(v[i], qhat, pHi, pLo);
            uint64_t t = pLo + carryMul;
            carryMul = pHi + ((t < pLo) ? 1ull : 0ull);

            uint64_t uu = u[j + i];
            uint64_t sub = uu - t;
            uint64_t b1 = (sub > uu) ? 1ull : 0ull;
            uint64_t sub2 = sub - borrow;
            uint64_t b2 = (sub2 > sub) ? 1ull : 0ull;
            u[j + i] = sub2;
            borrow = (b1 | b2);
        }

        uint64_t uu = u[j + m];
        uint64_t sub = uu - carryMul;
        uint64_t b1 = (sub > uu) ? 1ull : 0ull;
        uint64_t sub2 = sub - borrow;
        uint64_t b2 = (sub2 > sub) ? 1ull : 0ull;
        u[j + m] = sub2;

        if ((b1 | b2) != 0) {
            qhat -= 1;
            uint64_t c = 0;
#pragma unroll
            for (size_t i = 0; i < DL; ++i) {
                uint64_t prev = u[j + i];
                uint64_t sum = prev + v[i] + c;
                c = (sum < prev) ? 1ull : 0ull;
                if (c == 0 && sum < v[i]) c = 1ull;
                u[j + i] = sum;
            }
            u[j + m] += c;
        }

        quotient.limb[j] = qhat;
    }

    if (s == 0) {
#pragma unroll
        for (size_t i = 0; i < DL; ++i) remainder.limb[i] = u[i];
    } else {
        uint64_t carry = 0;
        for (size_t ii = 0; ii < DL; ++ii) {
            size_t i = DL - 1 - ii;
            uint64_t w = u[i];
            remainder.limb[i] = (w >> s) | carry;
            carry = (w << (64u - s));
        }
    }
}

template <size_t NL, size_t DL>
__host__ __device__ static inline BigUInt<DL> mod_fast(const BigUInt<NL>& numerator, const BigUInt<DL>& denom)
{
    // Dispatch to a fixed-size fast path for common widths, otherwise fall back to generic.
    // The fixed-size path requires full-width top limbs non-zero.
    // Extra fast path: denom fits in u64 (very common early in trial division).

    bool denomFitsU64 = true;
    for (size_t i = 1; i < DL; ++i) {
        if (denom.limb[i] != 0ull) {
            denomFitsU64 = false;
            break;
        }
    }
    if (denomFitsU64) {
        const uint64_t d = denom.limb[0];
        const uint64_t r = mod_u64(numerator, d);
        BigUInt<DL> out = BigUInt<DL>::zero();
        out.limb[0] = r;
        return out;
    }

    const bool full = (numerator.limb[NL - 1] != 0ull) && (denom.limb[DL - 1] != 0ull);

    if constexpr ((NL == 32 && DL == 16) || (NL == 16 && DL == 8) || (NL == 8 && DL == 4) || (NL == 4 && DL == 2)) {
        if (full) {
            return mod_knuth_fixed_full_streaming<NL, DL>(numerator, denom);
        }
    }
    return mod_knuth<NL, DL>(numerator, denom);
}

template <size_t NL, size_t DL>
__host__ __device__ static inline void div_mod_fast(const BigUInt<NL>& numerator,
                                                    const BigUInt<DL>& denom,
                                                    BigUInt<NL>& quotient,
                                                    BigUInt<DL>& remainder)
{
    bool denomFitsU64 = true;
    for (size_t i = 1; i < DL; ++i) {
        if (denom.limb[i] != 0ull) {
            denomFitsU64 = false;
            break;
        }
    }
    if (denomFitsU64) {
        const uint64_t d = denom.limb[0];
        uint64_t rem = 0;
        div_mod_u64(numerator, d, quotient, rem);
        remainder = BigUInt<DL>::zero();
        remainder.limb[0] = rem;
        return;
    }

    const bool full = (numerator.limb[NL - 1] != 0ull) && (denom.limb[DL - 1] != 0ull);

    if constexpr ((NL == 32 && DL == 16) || (NL == 16 && DL == 8) || (NL == 8 && DL == 4) || (NL == 4 && DL == 2)) {
        if (full) {
            div_mod_knuth_fixed_full<NL, DL>(numerator, denom, quotient, remainder);
            return;
        }
    }
    div_mod_knuth<NL, DL>(numerator, denom, quotient, remainder);
}

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
    // Use faster limb-based remainder when possible.
    return mod_knuth<LIMBS, LIMBS>(a, m);
}

template <size_t LIMBS>
__host__ __device__ static inline void div_mod_big(const BigUInt<LIMBS>& numerator,
                                                  const BigUInt<LIMBS>& denom,
                                                  BigUInt<LIMBS>& quotient,
                                                  BigUInt<LIMBS>& remainder)
{
    div_mod_knuth<LIMBS, LIMBS>(numerator, denom, quotient, remainder);
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
