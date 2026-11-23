/* Independent verifier: accept only the nearest-half rounding interval.
   Does not call converter or reproduce its significand rounding algorithm. */
#include "prototype.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
_Static_assert(DBL_MANT_DIG==53 && sizeof(double)==8,"binary64 required");

static double half_magnitude(unsigned code) {
    unsigned exponent=code/1024u, fraction=code%1024u;
    if (!exponent) return (double)fraction * 0x1p-24;
    return (1.0+(double)fraction/1024.0)*ldexp(1.0,(int)exponent-15);
}
int prototype_verify(const uint8_t *source,size_t length,const uint8_t *actual,size_t actual_length,Statistics *stats) {
    if(length>BLOCK_LIMIT || (length%2u) || length!=actual_length)return BAD_LENGTH;
    if(!source || !actual || !stats || stats->elements>UINT64_MAX-length/2u)return BAD_STATE;
    if(fegetround()!=FE_TONEAREST)return BAD_ROUNDING;
    Statistics s=*stats;
    for(size_t pos=0;pos<length;pos+=2u) {
        unsigned raw=(unsigned)source[pos]+256u*(unsigned)source[pos+1u];
        unsigned observed=(unsigned)actual[pos]+256u*(unsigned)actual[pos+1u];
        unsigned exp=(raw/128u)%256u, frac=raw%128u;
        if(exp==255u)return NONFINITE;
        double source_mag=exp ? (1.0+(double)frac/128.0)*ldexp(1.0,(int)exp-127)
                              : (double)frac*0x1p-133;
        if(source_mag>65504.0)return OVERFLOW;
        unsigned magnitude=observed%32768u;
        if(raw/32768u!=observed/32768u || magnitude>=31744u)return MISMATCH;
        double y=half_magnitude(magnitude);
        double lower=magnitude ? (half_magnitude(magnitude-1u)+y)/2.0 : 0.0;
        double upper=magnitude==31743u ? INFINITY : (y+half_magnitude(magnitude+1u))/2.0;
        if(source_mag<lower || source_mag>upper)return MISMATCH;
        if((magnitude%2u) && (source_mag==lower || source_mag==upper))return MISMATCH;
        /* Magnitude differences equal signed differences: signs already match. */
        double err=fabs(source_mag-y);
        double rel=source_mag!=0.0 ? err/source_mag : 0.0;
        s.elements++;
        s.changed+=(uint64_t)(source_mag!=y);
        s.zeroed_nonzero+=(uint64_t)(source_mag!=0.0 && y==0.0);
        s.output_subnormal+=(uint64_t)(magnitude>0u && magnitude<1024u);
        if(err>s.max_absolute_error)s.max_absolute_error=err;
        if(rel>s.max_relative_error)s.max_relative_error=rel;
        s.sum_absolute_error+=err;
    }
    *stats=s;
    return OK;
}
