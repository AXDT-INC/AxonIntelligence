#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>
#include <linux/fs.h>
#include <sys/syscall.h>
#include <unistd.h>
/* Opaque process-lifetime records: bounded to 4096, no recycled identities.
   All record reads/writes use the same mutex. No Python calls in this file. */
typedef struct Gate {struct Gate *next; pthread_mutex_t lock; uint64_t session,generation; unsigned eligible,invalid,attempted,confirmed; int failure; } Gate;
static pthread_mutex_t allocation_lock=PTHREAD_MUTEX_INITIALIZER;
static unsigned allocations;
static Gate *records;
void *gate_create(uint64_t session,uint64_t generation){
 if(!session||!generation){errno=EINVAL;return NULL;}
 pthread_mutex_lock(&allocation_lock);
 for(Gate *old=records;old;old=old->next){if(old->session==session&&old->generation==generation){pthread_mutex_unlock(&allocation_lock);errno=EEXIST;return NULL;}}
 if(allocations>=4096){pthread_mutex_unlock(&allocation_lock);errno=ENOSPC;return NULL;}
 Gate *g=calloc(1,sizeof *g);
 if(!g){pthread_mutex_unlock(&allocation_lock);return NULL;}
 int err=pthread_mutex_init(&g->lock,NULL);
 if(err){free(g);pthread_mutex_unlock(&allocation_lock);errno=err;return NULL;}
 g->session=session;g->generation=generation;g->next=records;records=g;allocations++;pthread_mutex_unlock(&allocation_lock);return g;
}
static int enter(Gate*g,uint64_t s,uint64_t n){
 if(!g){errno=EINVAL;return -1;}
 int e=pthread_mutex_lock(&g->lock);if(e){errno=e;return -1;}
 if(g->session!=s||g->generation!=n){pthread_mutex_unlock(&g->lock);errno=ESTALE;return -1;}return 0;
}
int gate_arm(Gate*g,uint64_t s,uint64_t n){
 if(enter(g,s,n))return -1;
 int e=g->invalid||g->attempted||g->eligible?EPERM:0;
 if(!e)g->eligible=1;pthread_mutex_unlock(&g->lock);errno=e;return e?-1:0;
}
int gate_invalidate(Gate*g,uint64_t s,uint64_t n){
 if(enter(g,s,n))return -1;
 g->invalid=1;g->eligible=0;pthread_mutex_unlock(&g->lock);return 0;
}
int gate_status(Gate*g,uint64_t s,uint64_t n,uint64_t*out){
 if(!out){errno=EINVAL;return -1;}if(enter(g,s,n))return -1;
 out[0]=g->session;out[1]=g->generation;out[2]=g->eligible;out[3]=g->invalid;
 out[4]=g->attempted;out[5]=g->confirmed;out[6]=(unsigned)g->failure;
 pthread_mutex_unlock(&g->lock);return 0;
}
int publish_once(int dir,Gate*g,uint64_t s,uint64_t n,const char*stage,const char*final){
 if(enter(g,s,n))return -1;
 if(!g->eligible||g->invalid||g->attempted){pthread_mutex_unlock(&g->lock);errno=EPERM;return -1;}
 /* Decision and filesystem syscall in ONE native critical section. PyDLL also
    retains the GIL: Python signals/hooks run only outside this operation. */
 g->eligible=0;g->attempted=1;
 int rc=(int)syscall(SYS_renameat2,dir,stage,dir,final,RENAME_NOREPLACE);
 int saved=rc?errno:0;g->failure=saved;g->confirmed=rc==0;
 pthread_mutex_unlock(&g->lock);errno=saved;return rc;
}
