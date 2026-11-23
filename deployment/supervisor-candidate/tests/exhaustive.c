#include "prototype.h"
#include <stdio.h>
#include <string.h>
int main(void){unsigned corrupted=0;
 for(unsigned b=0;b<65536;b++){unsigned char in[2]={b,b>>8},out[2]={0};Statistics s={0};unsigned char rc=prototype_convert(in,2,out,2,&s);
  if(!rc){Statistics v={0};if(prototype_verify(in,2,out,2,&v)||memcmp(&s,&v,80))return 2;
   for(unsigned bit=0;bit<16;bit++){unsigned char bad[2]={out[0],out[1]};bad[bit/8]^=(1u<<(bit%8));Statistics reject={0};if(!prototype_verify(in,2,bad,2,&reject))return 3;corrupted++;}}
  if(fwrite(&rc,1,1,stdout)!=1||fwrite(out,1,2,stdout)!=2||fwrite(&s,1,80,stdout)!=80)return 4;
 }
 fprintf(stderr,"{\"corrupted_outputs_rejected\":%u}\n",corrupted);return 0;
}
