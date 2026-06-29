#ifndef RANDOM_H
#define RANDOM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * This module wraps a single global PRNG instance. It is NOT thread-safe;
 * all functions below operate on shared global state with no internal
 * locking. If used from multiple threads, the caller must synchronize
 * access externally.
 *                            VERIFY!!!!
 */

/**
 * random_init -- create (if needed) and seed the global generator.
 *
 * Safe to call multiple times; each call reseeds the existing generator
 * rather than leaking and reallocating. If random_init() is never called,
 * random_index() will lazily initialize the generator with a default
 * seed on first use.
 *
 * @param seed  Seed value for the generator. Using the same seed produces
 *              the same sequence of outputs.
 */

void random_init(uint64_t seed);

/**
 * random_index -- return a random integer in the range [0, n).
 *
 * @param n  Exclusive upper bound. Must be > 0.
 * @return   A pseudo-random value uniformly distributed in [0, n).
 */
uint64_t random_index(uint64_t n);

/**
 * random_free -- release the generator and reset the pointer to NULL.
 *
 * Safe to call even if random_init() was never called, or if called
 * more than once in a row (idempotent). After calling, a subsequent
 * call to random_index() will lazily reinitialize the generator with
 * its default seed.
 */
void random_free(void);

#ifdef __cplusplus
}
#endif

#endif
