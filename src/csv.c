#include "csv.h"
#include "minicsv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *path;
    double *data;
    uint64_t size;
} CsvCacheEntry;

#define CSV_CACHE_SIZE 16
static CsvCacheEntry csv_cache[CSV_CACHE_SIZE];
static uint64_t csv_cache_count = 0;

static char *csv_strdup(const char *s) {
    size_t len = strlen(s) + 1;
    char *out = malloc(len);
    if (!out)
        return NULL;
    memcpy(out, s, len);
    return out;
}

static void csv_cache_clear(void) {
    for (uint64_t i = 0; i < csv_cache_count; i++) {
        free(csv_cache[i].path);
        free(csv_cache[i].data);
        csv_cache[i].path = NULL;
        csv_cache[i].data = NULL;
        csv_cache[i].size = 0;
    }
    csv_cache_count = 0;
}

static void csv_cache_register_cleanup(void) {
    static int registered = 0;
    if (!registered) {
        atexit(csv_cache_clear);
        registered = 1;
    }
}

static int csv_cache_lookup(const char *filename, TuningData *result) {
    for (uint64_t i = 0; i < csv_cache_count; i++) {
        if (strcmp(csv_cache[i].path, filename) == 0) {
            result->size = csv_cache[i].size;
            result->data = malloc(result->size * sizeof(double));
            if (!result->data) {
                result->size = 0;
                return 0;
            }

            memcpy(result->data, csv_cache[i].data,
                   result->size * sizeof(double));
            return 1;
        }
    }

    return 0;
}

static void csv_cache_store(const char *filename, const double *data,
                            uint64_t size) {
    if (csv_cache_count >= CSV_CACHE_SIZE || !data || size == 0)
        return;

    char *path_copy = csv_strdup(filename);
    double *data_copy = malloc(size * sizeof(double));
    if (!path_copy || !data_copy) {
        free(path_copy);
        free(data_copy);
        return;
    }

    memcpy(data_copy, data, size * sizeof(double));
    csv_cache[csv_cache_count].path = path_copy;
    csv_cache[csv_cache_count].data = data_copy;
    csv_cache[csv_cache_count].size = size;
    csv_cache_count++;
}

TuningData csv_load(const char *filename, const char *column_name) {
    TuningData result = {NULL, 0};
    csv_cache_register_cleanup();

    if (csv_cache_lookup(filename, &result))
        return result;

    FILE *fp = fopen(filename, "r");
    if (!fp)
        return result;
    char line[8192];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return result;
    }

    char *cols[256];
    size_t cols_count;
    minicsv_parse_line(line, cols, &cols_count, 256);
    uint64_t target_col = 0, found = 0;

    for (size_t i = 0; i < cols_count; i++) {
        if (strcmp(cols[i], column_name) == 0 ||
            strcmp(cols[i], "Kernel duration (us)") == 0) {
            target_col = i;
            found = 1;
            break;
        }
    }

    if (!found) {
        fclose(fp);
        return result;
    }

    uint64_t capacity = 1024;
    double *data = malloc(capacity * sizeof(double));
    if (!data) {
        fclose(fp);
        return result;
    }

    uint64_t idx = 0;
    while (fgets(line, sizeof(line), fp)) {
        minicsv_parse_line(line, cols, &cols_count, 256);
        if (target_col >= cols_count)
            continue;
        if (idx == capacity) {
            uint64_t new_capacity = capacity * 2;
            double *grown = realloc(data, new_capacity * sizeof(double));

            if (!grown) {
                free(data);
                fclose(fp);
                return result;
            }

            data = grown;
            capacity = new_capacity;
        }

        data[idx++] = atof(cols[target_col]);
    }
    fclose(fp);

    if (idx == 0) {
        free(data);
        return result;
    }

    if (idx < capacity) {
        double *shrunk = realloc(data, idx * sizeof(double));
        if (shrunk)
            data = shrunk;
    }

    result.data = data;
    result.size = idx;
    csv_cache_store(filename, result.data, result.size);
    return result;
}

void csv_free(TuningData *data) {
    if (data && data->data) {
        free(data->data);
        data->data = NULL;
        data->size = 0;
    }
}
