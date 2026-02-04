# CUDA Prime Factorisor

## About

This is a CUDA implementation of trial division prime factorisation. It is capable of finding prime factors for numbers up to 2^128. It works by first building and packing a factorisation wheel with a 510510 modulus (2⋅3⋅5⋅7⋅11⋅13⋅17). It then calculates the search space (0 - √N) and breaks it up into blocks. It dispatches each block to the GPU for trial division.

The integer type used for N is a fixed-width multi-limb big integer (base 2^64) that is compile-time configurable. By default it is built as a 256-bit unsigned integer.

## Build

```bash
sudo apt install nvidia-cuda-toolkit
nvcc -std=c++20 -O2 primefactors.cu -o primefactors

# Optional: set the bit-width (must be a multiple of 64).
# Examples:
#   -DPRIMEFAC_BITS=128  -> 2 limbs
#   -DPRIMEFAC_BITS=256  -> 4 limbs (default)
#   -DPRIMEFAC_BITS=512  -> 8 limbs
nvcc -std=c++20 -O2 -DPRIMEFAC_BITS=256 primefactors.cu -o primefactors
```

### Usage

```bash
./primefactors 213679575440397248358775931752856
Scanned blocks [0,524288) found 14 divisors (rem=26709946930049656044846991469107)46991469107)
GPU found divisor: 53
GPU found divisor: 2660113
GPU found divisor: 1159663
GPU found divisor: 605152979
GPU found divisor: 269960419
factor: 2
factor: 2
factor: 2
factor: 53
factor: 887
factor: 2999
factor: 1159663
factor: 605152979
factor: 269960419
```