#ifndef AXON_SUPERVISOR_PROTOCOL_H
#define AXON_SUPERVISOR_PROTOCOL_H
#include <stdint.h>
#define CONTROL_MAGIC 0x41585332u
#define CONTROL_ABI 2u
#define MAX_FRAME 131200u
#define START 1u
#define ACK 2u
#define STATUS 3u
#define CANCEL 4u
#define DATA 5u
#define FINISH 6u
#define COMMIT_BEGIN 7u
#define COMMIT_RESULT 8u
#define COMMIT_PROTOCOL 3u
#define RESPONSE 0x80000000u
/* All six fields little endian, x86_64 only. Token strictly increases on START. */
typedef struct {uint32_t magic,abi,seq,op,token,length;} Frame;
#endif
