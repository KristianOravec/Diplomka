#ifndef CSV_H
#define CSV_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    double *data;
    uint64_t size;
} TuningData;
TuningData csv_load(const char *filename, const char *column_name);
void csv_free(TuningData *data);
#ifdef __cplusplus
}
#endif
#endif
