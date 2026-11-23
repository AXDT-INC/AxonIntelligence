/* Integer conversion only; no _Float16, float cast, SIMD half conversion or I/O.
   Statistics use ordered binary64 operations matching V3 on FE_TONEAREST. */
#include "prototype.h"
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <string.h>
_Static_assert(DBL_MANT_DIG == 53 && sizeof(double) == 8, "binary64 required");

static uint16_t read_word(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}
static uint16_t convert_word(uint16_t bits) {
    unsigned e = (bits >> 7) & 255u;
    unsigned f = bits & 127u;
    unsigned sign = bits & 32768u;
    if (e < 102u) return (uint16_t)sign;
    if (e >= 113u) return (uint16_t)(sign | ((e-112u)<<10) | (f<<3));
    unsigned mant = 128u + f;
    if (e >= 110u) return (uint16_t)(sign | (mant << (e-110u)));
    unsigned shift = 110u-e;
    unsigned rounded = mant >> shift;
    unsigned tail = mant & ((1u<<shift)-1u);
    unsigned half = 1u<<(shift-1u);
    if (tail > half || (tail == half && (rounded & 1u))) rounded++;
    return (uint16_t)(sign | rounded);
}
static double bf_value(uint16_t bits) {
    unsigned e=(bits>>7)&255u, f=bits&127u;
    double magnitude = e ? ldexp((double)(128u+f),(int)e-134) : ldexp((double)f,-133);
    return bits & 32768u ? -magnitude : magnitude;
}
static double half_value(uint16_t bits) {
    unsigned e=(bits>>10)&31u, f=bits&1023u;
    double magnitude = e ? ldexp((double)(1024u+f),(int)e-25) : ldexp((double)f,-24);
    return bits & 32768u ? -magnitude : magnitude;
}
int prototype_convert(const uint8_t *src,size_t bytes,uint8_t *dst,size_t capacity,Statistics *stats) {
    if (bytes>BLOCK_LIMIT || (bytes&1u) || capacity<bytes) return BAD_LENGTH;
    if (!src || !dst || !stats || stats->elements>UINT64_MAX-bytes/2u) return BAD_STATE;
    if (fegetround()!=FE_TONEAREST) return BAD_ROUNDING;
    /* A rejected block changes neither output nor statistics. */
    for(size_t i=0;i<bytes;i+=2u) {
        uint16_t b=read_word(src+i);
        if (((b>>7)&255u)==255u) return NONFINITE;
        if ((b&32767u)>0x477fu) return OVERFLOW;
    }
    Statistics s=*stats;
    for(size_t i=0;i<bytes;i+=2u) {
        uint16_t b=read_word(src+i), h=convert_word(b);
        dst[i]=(uint8_t)(h&255u);dst[i+1]=(uint8_t)(h>>8);
        double x=bf_value(b),y=half_value(h);
        double error=fabs(y-x), relative=x!=0.0 ? error/fabs(x) : 0.0;
        s.elements++;
        s.changed+=(uint64_t)(x!=y);
        s.zeroed_nonzero+=(uint64_t)(x!=0.0 && y==0.0);
        s.output_subnormal+=(uint64_t)((h&32767u)>0u && (h&32767u)<1024u);
        if(error>s.max_absolute_error)s.max_absolute_error=error;
        if(relative>s.max_relative_error)s.max_relative_error=relative;
        s.sum_absolute_error+=error;
    }
    *stats=s;
    return OK;
}
