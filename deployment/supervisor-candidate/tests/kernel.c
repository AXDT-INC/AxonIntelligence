#define _POSIX_C_SOURCE 200809L
#include "prototype.h"
#include <stdio.h>
#include <time.h>
static unsigned char a[65536],b[65536];
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
int main(void){for(unsigned i=0;i<32768;i++){unsigned v=i%2?0x3301:0x3f80;a[2*i]=v;a[2*i+1]=v>>8;}printf("{\"status\":\"PASS\",\"trials\":[");
 for(unsigned trial=0;trial<3;trial++){double times[2];for(unsigned phase=0;phase<2;phase++){Statistics s={0};double t=now();for(unsigned block=0;block<16;block++){int rc=phase?prototype_verify(a,sizeof a,b,sizeof b,&s):prototype_convert(a,sizeof a,b,sizeof b,&s);if(rc)return 2;}times[phase]=now()-t;}
 printf("%s{\"convert_seconds\":%.9f,\"verify_seconds\":%.9f}",trial?",":"",times[0],times[1]);}puts("]}");return 0;}
