#ifndef RANDOM_H
#define RANDOM_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
void random_init(uint64_t seed);
uint64_t random_index(uint64_t n);
void random_free(void);
#ifdef __cplusplus
}
#endif
#endif
