#define _GNU_SOURCE
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <time.h>
#include <sched.h>
#include <sys/resource.h>
/* Dedicated subreaper: waitpid(-1) is confined to children of this test process.
   The only group signalled is a group this harness created itself. */
static pid_t seen[8192];static unsigned count;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec/1e9;}
static void scan(pid_t pid,unsigned depth){if(depth>8)return;char name[128];snprintf(name,sizeof name,"/proc/%d/task/%d/children",pid,pid);FILE*f=fopen(name,"r");if(!f)return;int child;while(fscanf(f,"%d",&child)==1){unsigned i;for(i=0;i<count;i++)if(seen[i]==child)break;if(i==count&&count<8192)seen[count++]=child;scan(child,depth+1);}fclose(f);}
int main(int argc,char**argv){if(argc<3||argc>4)return 2;
 cpu_set_t available,one;CPU_ZERO(&available);CPU_ZERO(&one);if(sched_getaffinity(0,sizeof available,&available))return 3;for(int i=0;i<CPU_SETSIZE;i++)if(CPU_ISSET(i,&available)){CPU_SET(i,&one);break;}
 struct rlimit cpu={120,120},core={0,0};if(sched_setaffinity(0,sizeof one,&one)||setpriority(PRIO_PROCESS,0,19)||setrlimit(RLIMIT_CPU,&cpu)||setrlimit(RLIMIT_CORE,&core))return 3;
 if(prctl(PR_SET_CHILD_SUBREAPER,1))return 3;
 const char*root=argv[1],*mode=argv[2],*scenario=argc==4?argv[3]:"none";
 pid_t top=fork();if(top==0){setpgid(0,0);char path[4096];if((!strcmp(mode,"exhaustive")||!strcmp(mode,"kernel"))){snprintf(path,sizeof path,"%s/%s",root,mode);char*a[]={path,NULL};execv(path,a);}else{snprintf(path,sizeof path,"%s/bootstrap.py",root);char*a[]={"/usr/bin/python3","-I","-B",path,(char*)mode,(char*)scenario,NULL};char*e[]={"PATH=/usr/bin:/bin","LC_ALL=C",NULL};execve(a[0],a,e);}_exit(120);}if(top<0)return 4;setpgid(top,top);
 double end=now()+600;unsigned reaped=0;int topstatus=-1,unresolved=0,timedout=0;
 for(;;){scan(getpid(),0);int status;pid_t r=waitpid(-1,&status,WNOHANG);if(r>0){reaped++;if(r==top)topstatus=status;dprintf(2,"{\"harness_event\":\"reaped\",\"pid\":%d,\"status\":%d}\n",r,status);continue;}if(r<0&&errno==ECHILD)break;if(r<0&&errno!=EINTR){unresolved=1;break;}
  if(now()>end){if(timedout){unresolved=1;break;}kill(-top,SIGKILL);timedout=1;end=now()+5;}
  struct timespec pause={0,10000000};nanosleep(&pause,NULL);
 }
 dprintf(2,"{\"harness_event\":\"complete\",\"os_descendants_observed\":%u,\"adopted_or_direct_reaped\":%u,\"ECHILD_confirmed\":%s,\"timed_out\":%s,\"top_status\":%d}\n",count,reaped,unresolved?"false":"true",timedout?"true":"false",topstatus);
 return unresolved||timedout?5:!WIFEXITED(topstatus)||WEXITSTATUS(topstatus)!=0?6:0;
}
