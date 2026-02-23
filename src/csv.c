#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "csv.h"
#include "minicsv.h"

TuningData csv_load(const char* filename, const char* column_name) {
    TuningData result = {NULL, 0};
    FILE* fp = fopen(filename, "r");
    if (!fp) return result;
    char line[8192];
    if (!fgets(line, sizeof(line), fp)) { fclose(fp); return result; }
    
    char* cols[256];
    size_t cols_count;
    minicsv_parse_line(line, cols, &cols_count, 256);
    uint64_t target_col = 0, found = 0;
    for (size_t i = 0; i < cols_count; i++) {
        if (strcmp(cols[i], column_name) == 0 || strcmp(cols[i], "Kernel duration (us)") == 0) {
            target_col = i; found = 1; break;
        }
    }
    fclose(fp);
    if (!found) return result;
    
    fp = fopen(filename, "r");
    if (!fp) return result;
    fgets(line, sizeof(line), fp);
    
    uint64_t row_count = 0;
    while (fgets(line, sizeof(line), fp)) row_count++;
    
    double* data = (double*)malloc(row_count * sizeof(double));
    if (!data) { fclose(fp); return result; }
    
    rewind(fp);
    fgets(line, sizeof(line), fp);
    
    uint64_t idx = 0;
    while (fgets(line, sizeof(line), fp) && idx < row_count) {
        minicsv_parse_line(line, cols, &cols_count, 256);
        if (target_col < cols_count) data[idx++] = atof(cols[target_col]);
    }
    fclose(fp);
    result.data = data;
    result.size = idx;
    return result;
}

void csv_free(TuningData* data) {
    if (data && data->data) { free(data->data); data->data = NULL; data->size = 0; }
}
