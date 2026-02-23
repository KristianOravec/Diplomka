#include <stdlib.h>
#include "random.h"
#include <gsl/gsl_rng.h>
static gsl_rng* global_rng = NULL;
void random_init(uint64_t seed) {
    if (!global_rng) global_rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(global_rng, seed);
}
uint64_t random_index(uint64_t n) {
    if (!global_rng) random_init(0);
    return gsl_rng_get(global_rng) % n;
}
void random_free(void) {
    if (global_rng) { gsl_rng_free(global_rng); global_rng = NULL; }
}
