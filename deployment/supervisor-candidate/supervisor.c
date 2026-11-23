#define _GNU_SOURCE
#include "protocol.h"
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <poll.h>
#include <time.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#if !defined(__x86_64__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error Linux x86_64 little endian required
#endif
_Static_assert(sizeof(Frame)==24,"control ABI");
static volatile sig_atomic_t cancelled;
static void flag(int sig){cancelled=sig;}
static double now(void){struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))_exit(110);return t.tv_sec+t.tv_nsec/1e9;}
/* C handlers only set flags. No allocation callbacks or asynchronous unwinding. */
typedef struct {pid_t pid;int pidfd,in,out,reaped,status;uint32_t token;const char*state;unsigned errors;int codes[64];double deadline;struct rusage usage;} Child;
static Child w={.pid=-1,.pidfd=-1,.in=-1,.out=-1,.state="not_launched"},p={.pid=-1,.pidfd=-1,.in=-1,.out=-1,.state="not_launched"};
static int image=-1;static const char*fault="none";static unsigned killfault,waitfault,closefault;static uint32_t numerical_seq;
/* Per worker-token publication history, owned solely by this event loop.
   ACTIVE is a grant, not evidence of filesystem publication. */
enum Commit { COMMIT_IDLE, COMMIT_ACTIVE, COMMIT_DENIED, COMMIT_NOT_PUBLISHED, COMMIT_PUBLISHED, COMMIT_UNKNOWN };
static enum Commit commit_state;
static unsigned cancel_requested,finish_completed;
static unsigned commit_durability=3;
static const char *commit_decision="none";
static const char *commit_names[]={"idle","in_progress","denied","not_published","published","UNKNOWN"};
static int cancel_complete(void){return cancel_requested&&commit_state!=COMMIT_ACTIVE&&commit_state!=COMMIT_UNKNOWN&&w.reaped&&!w.errors&&w.in<0&&w.out<0&&w.pidfd<0;}
static void error(Child*c,int code){if(c->errors<64)c->codes[c->errors]=code;c->errors++;}
static void closefd(Child*c,int*fd){if(*fd>=0){int old=*fd;*fd=-1;if(close(old))error(c,1000+errno);if(c==&w&&!strcmp(fault,"close_failure")&&closefault++<2)error(c,1000+EIO);}}
static void event(const char*name,Child*c){dprintf(2,"{\"event\":\"%s\",\"pid\":%d,\"token\":%u,\"state\":\"%s\",\"reaped\":%s,\"errors\":%u}\n",name,c->pid,c->token,c->state,c->reaped?"true":"false",c->errors);}
static int transfer(int fd,void*buf,size_t n,int writing,double deadline){size_t initial=n;unsigned char*b=buf;while(n){if(cancelled)return -1;double left=deadline-now();if(left<=0)return -1;struct pollfd q={fd,writing?POLLOUT:POLLIN,0};int r=poll(&q,1,(int)(left*1000)+1);if(r<0&&errno==EINTR)continue;if(r<=0)return -1;ssize_t z=writing?write(fd,b,n):read(fd,b,n);if(z<0&&(errno==EINTR||errno==EAGAIN))continue;if(z==0)return n==initial?-2:-1;if(z<0)return -1;b+=z;n-=(size_t)z;}return 0;}
static int reap(Child*c){if(c->pid<=0||c->reaped)return 1;
 if(c==&w&&((!strcmp(fault,"wait_failure")&&waitfault++<2)||(!strcmp(fault,"wait_unresolved")&&waitfault++<250))){error(c,2000+EIO);return 0;}
 struct rusage use;int status;pid_t r=wait4(c->pid,&status,WNOHANG,&use);
 if(r==c->pid){c->usage=use;c->reaped=1;c->status=status;c->state="exited_reaped";event("wait_confirmed",c);dprintf(2,"{\"event\":\"usage\",\"pid\":%d,\"user_seconds\":%.6f,\"system_seconds\":%.6f,\"peak_rss_kib\":%ld}\n",c->pid,use.ru_utime.tv_sec+use.ru_utime.tv_usec/1e6,use.ru_stime.tv_sec+use.ru_stime.tv_usec/1e6,use.ru_maxrss);return 1;}
 if(r<0)error(c,2000+errno);return 0;
}
static int terminate(Child*c){if(c->pid<=0||c->reaped)return 0;c->state="termination_requested";
 if(c==&w&&((!strcmp(fault,"kill_failure")&&killfault++<2)||(!strcmp(fault,"kill_unresolved")&&killfault++<25))){error(c,3000+EPERM);c->state="termination_unconfirmed";return -1;}
 /* pid fallback only for our exclusive, unreaped direct child when pidfd_open failed. */
 int r=c->pidfd>=0?(int)syscall(SYS_pidfd_send_signal,c->pidfd,SIGKILL,NULL,0):kill(c->pid,SIGKILL);
 if(r&&errno!=ESRCH)error(c,3000+errno);c->state="termination_unconfirmed";return r;
}
static int cleanup(Child*c,int killfirst){if(c->pid>0&&!c->reaped){if(killfirst)terminate(c);double end=now()+2;unsigned attempts=0;while(!reap(c)&&now()<end){if(killfirst&&attempts++%20==0)terminate(c);struct timespec t={0,10000000};nanosleep(&t,NULL);}if(!c->reaped){c->state="cleanup_incomplete";error(c,4000);event("unresolved",c);return -1;}}
 closefd(c,&c->in);closefd(c,&c->out);closefd(c,&c->pidfd);return c->errors?1:0;
}
static int resource_snapshot(char*b,size_t size){struct rusage own;getrusage(RUSAGE_SELF,&own);size_t off=(size_t)snprintf(b,size,"{\"pid\":%d,\"token\":%u,\"process_state\":\"%s\",\"reaped\":%s,\"cleanup_complete\":%s,\"wait_status\":%d,\"cleanup_errors\":[",w.pid,w.token,w.state,w.reaped?"true":"false",w.reaped&&w.in<0&&w.out<0&&w.pidfd<0?"true":"false",w.status);for(unsigned i=0;i<w.errors&&i<64&&off<size;i++)off+=(size_t)snprintf(b+off,size-off,"%s%d",i?",":"",w.codes[i]);if(off<size)off+=(size_t)snprintf(b+off,size-off,"],\"error_count\":%u,\"errors_truncated_count\":%u,\"worker_cpu_seconds\":%.9f,\"worker_peak_rss_kib\":%ld,\"supervisor_cpu_seconds\":%.9f,\"supervisor_peak_rss_kib\":%ld}",w.errors,w.errors>64?w.errors-64:0,w.usage.ru_utime.tv_sec+w.usage.ru_utime.tv_usec/1e6+w.usage.ru_stime.tv_sec+w.usage.ru_stime.tv_usec/1e6,w.usage.ru_maxrss,own.ru_utime.tv_sec+own.ru_utime.tv_usec/1e6+own.ru_stime.tv_sec+own.ru_stime.tv_usec/1e6,own.ru_maxrss);return off<size?(int)off:-1;}
static int snapshot(char*b,size_t size){
 int n=resource_snapshot(b,size);if(n<1)return -1;
 int extra=snprintf(b+n-1,size-(size_t)n+1,",\"commit_protocol\":%u,\"cancellation_requested\":%s,\"cancellation_complete\":%s,\"finish_completed\":%s,\"commit_state\":\"%s\",\"commit_decision\":\"%s\",\"commit_durability\":%u}",COMMIT_PROTOCOL,cancel_requested?"true":"false",cancel_complete()?"true":"false",finish_completed?"true":"false",commit_names[commit_state],commit_decision,commit_durability);
 return extra<0||(size_t)extra>=size-(size_t)n+1?-1:n-1+extra;
}
static int reply(Frame h,const void*b,unsigned n){h.op|=RESPONSE;h.length=n;double end=now()+5;return transfer(p.out,&h,24,1,end)||transfer(p.out,(void*)b,n,1,end);}
static int status_reply(Frame h){char b[2048];int n=snapshot(b,sizeof b);return n<0?-1:reply(h,b,(unsigned)n);}
static void child_signals(void){struct sigaction d={.sa_handler=SIG_DFL};sigemptyset(&d.sa_mask);sigaction(SIGINT,&d,NULL);sigaction(SIGTERM,&d,NULL);sigaction(SIGHUP,&d,NULL);signal(SIGPIPE,SIG_DFL);sigset_t empty;sigemptyset(&empty);sigprocmask(SIG_SETMASK,&empty,NULL);}
static void child_fds(int input,int output,int keep){if(dup2(input,0)<0||dup2(output,1)<0)_exit(111);for(int i=3;i<256;i++)if(i!=keep)close(i);}
static int launch_worker(uint32_t token){
 if(w.pid>0&&!w.reaped)return -1;
 w=(Child){.pid=-1,.pidfd=-1,.in=-1,.out=-1,.token=token,.state="launch_pending",.deadline=now()+2};
 int a[2]={-1,-1},b[2]={-1,-1};
 /* pipe2 writes descriptor numbers into preallocated native storage. */
 if(pipe2(a,O_CLOEXEC|O_NONBLOCK))return -1;
 if(!strcmp(fault,"descriptor_checkpoint"))raise(SIGTERM);
 if(pipe2(b,O_CLOEXEC|O_NONBLOCK)||cancelled){close(a[0]);close(a[1]);if(b[0]>=0){close(b[0]);close(b[1]);}return -1;}
 pid_t parent=getpid();w.pid=fork();
 if(w.pid==0){child_signals();if(prctl(PR_SET_PDEATHSIG,SIGKILL)||getppid()!=parent)_exit(112);child_fds(a[0],b[1],image);int f=fcntl(0,F_GETFL);fcntl(0,F_SETFL,f&~O_NONBLOCK);f=fcntl(1,F_GETFL);fcntl(1,F_SETFL,f&~O_NONBLOCK);char*args[]={"axon-worker",NULL};char*env[]={"LC_ALL=C",NULL};fexecve(image,args,env);_exit(113);}
 w.in=a[1];w.out=b[0];close(a[0]);close(b[1]);
 if(w.pid<0){closefd(&w,&w.in);closefd(&w,&w.out);w.state="not_launched";return -1;}
 /* No Python return/assignment boundary exists. Flags cannot unwind this record. */
 w.pidfd=(int)syscall(SYS_pidfd_open,w.pid,0);event("created",&w);
 if(w.pidfd<0){error(&w,5000+errno);cleanup(&w,1);return -1;}
 uint32_t hello[6];if(transfer(w.out,hello,24,0,w.deadline)||memcmp(hello,(uint32_t[]){0x41584e31,1,0,0,0,80},24)){cleanup(&w,1);return -1;}
 w.state="ready_unacknowledged";numerical_seq=1;
 if(!strcmp(fault,"child_before_delivery"))kill(p.pid,SIGINT);
 if(!strcmp(fault,"stop_before_ack")||!strcmp(fault,"kill_unresolved"))kill(w.pid,SIGSTOP);
 return 0;
}
static int data(Frame h,unsigned char*buf){
 if(strcmp(w.state,"running_supervised")||h.length<104)return -1;
 uint32_t n[6];memcpy(n,buf,24);unsigned len=n[4],op=n[3];
 if(n[0]!=0x41584e31||n[1]!=1||n[2]!=numerical_seq++||(op!=1&&op!=2)||len>65536||(len&1)||n[5]!=80||h.length!=104+len*(op==2?2:1))return -1;
 double end=now()+5;if(transfer(w.in,buf,h.length,1,end)||transfer(w.out,buf,24,0,end))return -1;
 memcpy(n,buf,24);if(n[0]!=0x41584e31||n[1]!=1||n[2]!=numerical_seq-1||n[3]>6||n[5]!=80||n[4]!=(op==1&&n[3]==0?len:0))return -1;
 unsigned response=104+n[4];if(transfer(w.out,buf+24,response-24,0,end))return -1;
 if(!strcmp(fault,"unexpected_response")){buf[response]=0;response++;}
 return reply(h,buf,response);
}
static int loop(void){double session_end=now()+180;uint32_t sequence=1,last_token=0;unsigned char buf[MAX_FRAME] __attribute__((aligned(8)));
 for(;;){
  if(cancelled||now()>session_end)return -1;
  if(w.pid>0&&!w.reaped&&!strcmp(w.state,"ready_unacknowledged")&&now()>w.deadline)cleanup(&w,1);
  struct pollfd q={p.in,POLLIN,0};int r=poll(&q,1,50);if(r<0&&errno==EINTR)continue;if(r<0)return -1;if(!r){if(reap(&p))return 0;continue;}
  Frame h;int got=transfer(p.in,&h,24,0,now()+5);if(got)return got==-2&&(w.reaped||w.pid<0)?0:-1;
  if(h.magic!=CONTROL_MAGIC||h.abi!=CONTROL_ABI||h.seq!=sequence++||h.length>MAX_FRAME||h.op<START||h.op>COMMIT_RESULT)return -1;
  if(h.length&&transfer(p.in,buf,h.length,0,now()+5))return -1;
  if(h.op!=DATA&&h.op!=COMMIT_RESULT&&h.length)return -1;
  if(h.op==COMMIT_RESULT&&h.length!=8)return -1;
  commit_decision="none";
  if(h.op==START){if(h.token<=last_token||commit_state==COMMIT_ACTIVE||launch_worker(h.token))return -1;last_token=h.token;cancel_requested=finish_completed=0;commit_state=COMMIT_IDLE;commit_durability=3;if(!strcmp(fault,"lost_ready"))continue;if(status_reply(h))return -1;continue;}
  if(h.token!=w.token)return -1;
  if(h.op==ACK){if(strcmp(w.state,"ready_unacknowledged")||now()>w.deadline)return -1;uint32_t go=0x474f0001;if(transfer(w.in,&go,4,1,w.deadline))return -1;w.state="running_supervised";if(!strcmp(fault,"lost_ack"))continue;if(status_reply(h))return -1;}
  else if(h.op==STATUS){if(status_reply(h))return -1;}
  else if(h.op==CANCEL){
   cancel_requested=1; /* Never erased by FINISH, STATUS or cleanup. */
   if(commit_state!=COMMIT_ACTIVE&&commit_state!=COMMIT_UNKNOWN)cleanup(&w,1);
   /* During commit this is a PENDING acknowledgment, never completion. */
   if(status_reply(h))return -1;
  }
  else if(h.op==COMMIT_BEGIN){
   if(commit_state==COMMIT_IDLE&&!cancel_requested&&finish_completed&&w.reaped&&WIFEXITED(w.status)&&WEXITSTATUS(w.status)==0&&!w.errors&&w.in<0&&w.out<0&&w.pidfd<0){commit_state=COMMIT_ACTIVE;commit_decision="granted";}
   else {if(commit_state==COMMIT_IDLE)commit_state=COMMIT_DENIED;commit_decision="denied";}
   if(status_reply(h))return -1;
  }
  else if(h.op==COMMIT_RESULT){
   uint32_t result[2];memcpy(result,buf,8);
   if(commit_state!=COMMIT_ACTIVE||!((result[0]==0&&result[1]==0)||(result[0]==1&&(result[1]==1||result[1]==2))||(result[0]==2&&result[1]==3)))return -1;
   commit_state=result[0]==0?COMMIT_NOT_PUBLISHED:result[0]==1?COMMIT_PUBLISHED:COMMIT_UNKNOWN;
   commit_durability=result[1];if(status_reply(h))return -1;
  }
  else if(h.op==DATA){if(data(h,buf))return -1;}
  else if(h.op==FINISH){
   if(cancel_requested||finish_completed||strcmp(w.state,"running_supervised"))return -1;
   uint32_t hwork[6]={0x41584e31,1,numerical_seq++,3,0,80},ack[6];double end=now()+5;
   if(transfer(w.in,hwork,24,1,end)||transfer(w.out,ack,24,0,end))return -1;hwork[3]=0;if(memcmp(ack,hwork,24))return -1;
   closefd(&w,&w.in);unsigned char extra;struct pollfd tail={w.out,POLLIN,0};if(poll(&tail,1,5000)<=0||read(w.out,&extra,1)!=0)return -1;if(cleanup(&w,0)!=0||!WIFEXITED(w.status)||WEXITSTATUS(w.status)!=0)return -1;
   finish_completed=1;if(status_reply(h))return -1;
  }
 }
}
int main(int argc,char**argv){
 if(argc!=6||geteuid()==0||geteuid()!=getuid())return 120;image=atoi(argv[1]);const char*root=argv[2],*mode=argv[3],*scenario=argv[4];fault=argv[5];
 if(image<3||fcntl(image,F_GET_SEALS)!=(F_SEAL_WRITE|F_SEAL_GROW|F_SEAL_SHRINK|F_SEAL_SEAL))return 121;
 struct rlimit core={0,0},nofile={256,256},cpu={120,120},mem={512*1024*1024,512*1024*1024};if(setrlimit(RLIMIT_CPU,&cpu)||setrlimit(RLIMIT_AS,&mem)||setrlimit(RLIMIT_CORE,&core)||setrlimit(RLIMIT_NOFILE,&nofile)||prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0))return 122;
 for(int fd=3;fd<256;fd++)if(fd!=image)close(fd);
 struct sigaction sa={.sa_handler=flag},def={.sa_handler=SIG_DFL};sigemptyset(&sa.sa_mask);sigemptyset(&def.sa_mask);
 if(sigaction(SIGINT,&sa,NULL)||sigaction(SIGTERM,&sa,NULL)||sigaction(SIGHUP,&sa,NULL)||sigaction(SIGCHLD,&def,NULL)||signal(SIGPIPE,SIG_IGN)==SIG_ERR)return 122;
 sigset_t empty;sigemptyset(&empty);if(sigprocmask(SIG_SETMASK,&empty,NULL))return 122;
 /* Supplemental stderr evidence must not block process ownership/cleanup on a full pipe. */
 int flags=fcntl(2,F_GETFL);if(flags<0||fcntl(2,F_SETFL,flags|O_NONBLOCK)<0)return 122;
 int a[2]={-1,-1},b[2]={-1,-1};if(pipe2(a,O_CLOEXEC|O_NONBLOCK)||pipe2(b,O_CLOEXEC|O_NONBLOCK))return 123;
 p.pid=fork();if(p.pid==0){child_signals();/* Dedicated fixed inherited channel; no Python acquisition. */
  int i=fcntl(a[0],F_DUPFD_CLOEXEC,10),o=fcntl(b[1],F_DUPFD_CLOEXEC,10);if(i<0||o<0||dup2(i,3)<0||dup2(o,4)<0)_exit(124);
  for(int fd=5;fd<256;fd++)close(fd);close(0);char path[4096];if(snprintf(path,sizeof path,"%s/coordinator.py",root)>=(int)sizeof path)_exit(125);
  char*args[]={"/usr/bin/python3","-I","-B",path,(char*)mode,(char*)scenario,NULL};char*env[]={"LC_ALL=C","PATH=/usr/bin:/bin",NULL};execve(args[0],args,env);_exit(126);
 }
 p.in=b[0];p.out=a[1];close(a[0]);close(b[1]);if(p.pid<0)return 127;p.state="running_supervised";p.pidfd=(int)syscall(SYS_pidfd_open,p.pid,0);event("coordinator_created",&p);
 int rc=p.pidfd<0?-1:loop();
 if(commit_state==COMMIT_ACTIVE)commit_state=COMMIT_UNKNOWN;
 if(cleanup(&w,1)<0)rc=-1;
 {char evidence[4096];int n=snapshot(evidence,sizeof evidence);if(n>0)dprintf(2,"{\"event\":\"commit_session_end\",\"publication\":%s}\n",evidence);}

 /* EOF/invalid protocol: give coordinator bounded time to report original error. */
 closefd(&p,&p.in);closefd(&p,&p.out);if(cleanup(&p,0)<0)cleanup(&p,1);
 event("session_end",&w);close(image);
 if(!p.reaped||(!w.reaped&&w.pid>0))return 128;
 if(!WIFEXITED(p.status)||WEXITSTATUS(p.status)!=0)return 129;
 return rc?130:0;
}
