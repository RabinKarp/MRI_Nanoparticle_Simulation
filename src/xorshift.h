#ifndef XORSHIFT_H
#define XORSHIFT_H

/*  The code for the xoroshiro128+ (as indicated below in comment blocks) was
    written in 2016 by David Blackman and Sebastiano Vigna (vigna@acm.org).
    This code was modified in 2016 by Aadyot Bhatnagar to satisfy the conditions
    placed on a UnivornRandomNumberGenerator for use with C++ STL random number
    generation functions. */

#include <stdint.h>

/*
 * This class defines a UniformRandomNumberGenerator based on the xoroshiro128+
 * PRNG. Though the class is templated, it is only intended to be used with the
 * uint64_t type. The only reason the class is templated is to satisfy the
 * standard conditions placed on UniformRandomNumberGenerator classes.
 */
template <typename T = uint64_t>
class XORShift
{
    public:
    typedef T result_type;
    XORShift(uint64_t x) { seed_rand(x); }

    /* Returns the smallest uint64_t that can be generated */
    static constexpr result_type min(void) { return 0; }

    /* Returns the largest uint64_t that can be generated */
    static constexpr result_type max(void) { return -1; }

    /* Returns the next random uint64_t */
     result_type operator()(void) { return next_rand(); }

    /*
     * The code below was written by Aadyot Bhatnagar (2016). It uses the
     * output of the xoroshiro128+ PRNG to generate a randomized 64-bit
     * unsigned integer and uses IEEE-754 rules for double-precision floating
     * point numbers to generate numbers in the ranges [0, 1) and (-1, 1).
     */

    /*
     * Returns a double in the range [0, 1).
     */
    double rand_pos_double(void)
    {
        uint64_t x = next_rand();
        union {uint64_t i; double d; } u;
        /* generate double in the range [1,2) */
        u.i = UINT64_C(0x3FF) << 52 | x >> 12;
        return u.d - 1.0;
    }

    /*
     * Returns a double in the range (-1, 1).
     */
    double rand_double(void)
    {
        uint64_t x = next_rand();
        union {uint64_t i; double d; } u;
        /* generate double in range (-2, -1] U [1,2) */
        u.i = (UINT64_C(0x3FF) << 52 | x >> 12) | (UINT64_C(1) << 63 & x << 52);
        if (u.d > 0)
            return u.d - 1.0;
        return u.d + 1.0;
    }

    /* This is the jump function for the generator. It is equivalent
       to 2^64 calls to next_rand(); it can be used to generate 2^64
       non-overlapping subsequences for parallel computations. Written
       by the authors of the original xoroshiro128+ code.*/

    void jump(void) {
        static const uint64_t JUMP[] = {0xbeac0467eba5facb, 0xd86b048b86aa9922};

        uint64_t s0 = 0;
        uint64_t s1 = 0;
        for(unsigned i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
            for(int b = 0; b < 64; b++) {
                if (JUMP[i] & 1ULL << b) {
                    s0 ^= s[0];
                    s1 ^= s[1];
                }
                next_rand();
            }

        s[0] = s0;
        s[1] = s1;
    }


private:

/* Below is the successor to xorshift128+. It is the fastest full-period
   generator passing BigCrush without systematic failures, but due to the
   relatively short period it is acceptable only for applications with a
   mild amount of parallelism; otherwise, use a xorshift1024* generator.

   Beside passing BigCrush, this generator passes the PractRand test suite
   up to (and included) 16TB, with the exception of binary rank tests,
   which fail due to the lowest bit being an LFSR; all other bits pass all
   tests. We suggest to use a sign test to extract a random Boolean value.

   Note that the generator uses a simulated rotate operation, which most C
   compilers will turn into a single instruction. In Java, you can use
   Long.rotateLeft(). In languages that do not make low-level rotation
   instructions accessible xorshift128+ could be faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

   uint64_t s[2];

     static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

     uint64_t next_rand(void) {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = s0 + s1;

        s1 ^= s0;
        s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
        s[1] = rotl(s1, 36); // c

        return result;
    }

    /* This is a fixed-increment version of Java 8's SplittableRandom generator
       See http://dx.doi.org/10.1145/2714064.2660195 and
       http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

       It is a very fast generator passing BigCrush, and it can be useful if
       for some reason you absolutely want 64 bits of state; otherwise, we
       rather suggest to use a xorshift128+ (for moderately parallel
       computations) or xorshift1024* (for massively parallel computations)
       generator.

       The function has been modified from the original code used in the
       splitmix64 PRNG to seed an array of two uint64_t's for use in the
       xoroshiro128+ PRNG.
    */

    void seed_rand(uint64_t x) {
        uint64_t z;
        int i;
        for (i = 0; i < 2; i++)
        {
            z = (x += UINT64_C(0x9E3779B97F4A7C15));
            z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
            s[i] = z ^ (z >> 31);
        }
    }
};
#endif /* XORSHIFT_H */
