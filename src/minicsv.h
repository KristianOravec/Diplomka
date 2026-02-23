#ifndef MINICSV_H
#define MINICSV_H

#include <stddef.h>

#define MINICSV_DELIM ','
#define MINICSV_QUOTE '"'

char *minicsv_parse_line(char *const buf, char **const cols,
                         size_t *const cols_count_p, const size_t cols_max);

void minicsv_trim_cols(char **const cols, const size_t cols_count);

#endif
