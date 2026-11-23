#define _GNU_SOURCE
#include "prototype.h"
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <sys/syscall.h>
#include <stddef.h>
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__ || !defined(__x86_64__)
#error reviewed ABI requires little-endian x86_64
#endif
_Static_assert(sizeof(Statistics)==80,"statistics ABI");
#define MAGIC 0x41584e31u
static unsigned char src[65536],out[65536];
static int io(int fd,void *p,size_t n,int writing){
 unsigned char*b=p;while(n){ssize_t z=writing?write(fd,b,n):read(fd,b,n);if(z<0&&errno==EINTR)continue;if(z<=0)return -1;b+=z;n-=(size_t)z;}return 0;
}
static void sandbox(void){
 /* Only stdin reads, stdout/stderr writes, and termination/signal return. */
 struct sock_filter f[]={
 BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,arch)),
 BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,AUDIT_ARCH_X86_64,1,0),BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
 BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,nr)),
 BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_exit_group,10,0),
 BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_exit,9,0),
 BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_rt_sigreturn,8,0),
 BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_read,2,0),
 BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,__NR_write,3,0),
 BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS),
 BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,args[0])),
 BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,0,3,4),
 BPF_STMT(BPF_LD|BPF_W|BPF_ABS,offsetof(struct seccomp_data,args[0])),
 BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,1,1,0),
 BPF_JUMP(BPF_JMP|BPF_JEQ|BPF_K,2,0,1),
 BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_ALLOW),
 BPF_STMT(BPF_RET|BPF_K,SECCOMP_RET_KILL_PROCESS)};
 struct sock_fprog prog={(unsigned short)(sizeof f/sizeof f[0]),f};
 struct rlimit mem={128*1024*1024,128*1024*1024},cpu={120,120},core={0,0};
 if(setrlimit(RLIMIT_AS,&mem)||setrlimit(RLIMIT_CPU,&cpu)||setrlimit(RLIMIT_CORE,&core)||prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0)||prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,&prog))_exit(90);
}
int main(void){
 sandbox();uint32_t hello[6]={MAGIC,1,0,0,0,80};if(io(1,hello,sizeof hello,1))return 91;
 uint32_t seq=1;
 for(;;){uint32_t h[6];Statistics s;
 if(io(0,h,sizeof h,0))return 92;
 if(h[0]!=MAGIC||h[1]!=1||h[2]!=seq++||h[4]>65536||(h[4]&1)||h[5]!=80)return 93;
 if(h[3]==3){if(h[4])return 93;h[3]=0;if(io(1,h,sizeof h,1))return 94;return 0;}
 if(h[3]!=1&&h[3]!=2)return 93;
 if(io(0,&s,80,0)||io(0,src,h[4],0))return 94;
 if(s.elements>UINT64_MAX-h[4]/2||s.changed>s.elements||s.zeroed_nonzero>s.elements||s.output_subnormal>s.elements||s.input_nonfinite||s.overflow||s.output_nonfinite||!isfinite(s.sum_absolute_error)||s.sum_absolute_error<0||!isfinite(s.max_absolute_error)||s.max_absolute_error<0||!isfinite(s.max_relative_error)||s.max_relative_error<0)return 95;
 unsigned op=h[3];if(op==2&&io(0,out,h[4],0))return 94;
 int rc=op==1?prototype_convert(src,h[4],out,h[4],&s):prototype_verify(src,h[4],out,h[4],&s);
 h[3]=(uint32_t)rc;if(op==2||rc)h[4]=0;
 if(io(1,h,sizeof h,1)||io(1,&s,80,1)||(h[4]&&io(1,out,h[4],1)))return 94;
 if(rc)return 96;
 }
}
