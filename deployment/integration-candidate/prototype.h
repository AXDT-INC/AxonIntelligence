#ifndef AXON_PROTO_H
#define AXON_PROTO_H
#include <stdint.h>
#include <stddef.h>
#define BLOCK_LIMIT 65536u
/* Return codes are versioned prototype API, not a V3 artifact format. */
enum result { OK=0, NONFINITE=1, OVERFLOW=2, BAD_LENGTH=3, MISMATCH=4,
              BAD_ROUNDING=5, BAD_STATE=6 };
typedef struct {
    uint64_t elements, changed, zeroed_nonzero, output_subnormal;
    uint64_t input_nonfinite, overflow, output_nonfinite;
    double max_absolute_error, max_relative_error, sum_absolute_error;
} Statistics;
int prototype_convert(const uint8_t *, size_t, uint8_t *, size_t, Statistics *);
int prototype_verify(const uint8_t *, size_t, const uint8_t *, size_t, Statistics *);
#endif
