/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CHECKS_H_
#define NCCL_CHECKS_H_

#include "debug.h"
// 把“错误检查 + 日志 + 返回/跳转”封装成统一宏，减少重复样板代码，并统一 NCCL 的错误传播语义。
// Check CUDA RT calls

// 1. CUDA 调用检查
// CUDACHECK(cmd)：执行 CUDA Runtime API；失败时打印 WARN，并直接 return ncclUnhandledCudaError。
// CUDACHECKGOTO(cmd, RES, label)：失败时把 RES 设为 ncclUnhandledCudaError，然后 goto label，适合需要集中清理资源的函数。
// CUDACHECKIGNORE(cmd)：失败只记录 INFO 并清掉最后错误，不中断流程，适合“可容忍失败”的路径。 
#define CUDACHECK(cmd) do {                                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
        return ncclUnhandledCudaError;                      \
    }                                                       \
} while(false)

#define CUDACHECKGOTO(cmd, RES, label) do {                 \
    cudaError_t err = cmd;                                  \
    if( err != cudaSuccess ) {                              \
        WARN("Cuda failure '%s'", cudaGetErrorString(err)); \
        RES = ncclUnhandledCudaError;                       \
        goto label;                                         \
    }                                                       \
} while(false)

// Report failure but clear error and continue
#define CUDACHECKIGNORE(cmd) do {  \
    cudaError_t err = cmd;         \
    if( err != cudaSuccess ) {     \
        INFO(NCCL_ALL,"%s:%d Cuda failure '%s'", __FILE__, __LINE__, cudaGetErrorString(err)); \
        (void) cudaGetLastError(); \
    }                              \
} while(false)
// 2. 系统调用检查（errno 语义）
// SYSCHECK(statement, name)：内部会重试 EINTR/EWOULDBLOCK/EAGAIN；最终失败 return ncclSystemError。
// SYSCHECKGOTO(..., RES, label)：同上，但失败走 goto 清理路径。
// SYSCHECKSYNC(...)：底层重试循环实现，供上层宏复用。
#include <errno.h>
// Check system calls
#define SYSCHECK(statement, name) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed: %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(statement, name, retval) do { \
  retval = (statement); \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

#define SYSCHECKGOTO(statement, name, RES, label) do { \
  int retval; \
  SYSCHECKSYNC((statement), name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed: %s", strerror(errno)); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)
// 3. pthread 调用检查
// Pthread calls don't set errno and never return EINTR.
#define PTHREADCHECK(statement, name) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN("Call to " name " failed: %s", strerror(retval)); \
    return ncclSystemError; \
  } \
} while (0)

#define PTHREADCHECKGOTO(statement, name, RES, label) do { \
  int retval = (statement); \
  if (retval != 0) { \
    WARN("Call to " name " failed: %s", strerror(retval)); \
    RES = ncclSystemError; \
    goto label; \
  } \
} while (0)
// 4. 通用条件断言式检查
// NEQCHECK / NEQCHECKGOTO：当表达式 不等于 期望值时，报错并返回/跳转。
// EQCHECK / EQCHECKGOTO：当表达式 等于 禁止值时，报错并返回/跳转。
// 这类宏用于快速守卫条件并附带定位日志（文件+行号）。
#define NEQCHECK(statement, value) do {   \
  if ((statement) != value) {             \
    /* Print the back trace*/             \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, ncclSystemError, strerror(errno));    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define NEQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) != value) { \
    /* Print the back trace*/ \
    RES = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, RES, strerror(errno));    \
    goto label; \
  } \
} while (0)

#define EQCHECK(statement, value) do {    \
  if ((statement) == value) {             \
    /* Print the back trace*/             \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, ncclSystemError, strerror(errno));    \
    return ncclSystemError;     \
  }                             \
} while (0)

#define EQCHECKGOTO(statement, value, RES, label) do { \
  if ((statement) == value) { \
    /* Print the back trace*/ \
    RES = ncclSystemError;    \
    INFO(NCCL_ALL,"%s:%d -> %d (%s)", __FILE__, __LINE__, RES, strerror(errno));    \
    goto label; \
  } \
} while (0)
// 5. NCCL 函数结果传播
// NCCLCHECK(call)：调用返回值若不是 ncclSuccess 或 ncclInProgress，则直接向上传播该错误。
// NCCLCHECKGOTO(call, RES, label)：同逻辑，但走统一清理出口。
// 这是该文件最关键的一组：保证错误“原样上抛”，不丢语义。
// Propagate errors up
#define NCCLCHECK(call) do { \
  ncclResult_t RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, RES);    \
    return RES; \
  } \
} while (0)

// 执行 call → 结果存入 RES
//         ↓
// RES 是 ncclSuccess 或 ncclInProgress？
//         ↓ 是                ↓ 否（出错）
//    继续向下执行         打印日志 → goto label
#define NCCLCHECKGOTO(call, RES, label) do { \
  RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    /* Print the back trace*/ \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, RES);    \
    goto label; \
  } \
} while (0)
// 6. 等待循环辅助
// NCCLWAIT / NCCLWAITGOTO：反复执行 call 直到 cond 成立；中间若 call 失败或 abortFlag 被置位则退出。
// 用于异步状态轮询，兼顾错误与中止信号。
#define NCCLWAIT(call, cond, abortFlagPtr) do {         \
  uint32_t* tmpAbortFlag = (abortFlagPtr);     \
  ncclResult_t RES = call;                \
  if (RES != ncclSuccess && RES != ncclInProgress) {               \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, RES);    \
    return ncclInternalError;             \
  }                                       \
  if (__atomic_load(tmpAbortFlag, __ATOMIC_ACQUIRE)) NEQCHECK(*tmpAbortFlag, 0); \
} while (!(cond))

#define NCCLWAITGOTO(call, cond, abortFlagPtr, RES, label) do { \
  uint32_t* tmpAbortFlag = (abortFlagPtr);             \
  RES = call;                             \
  if (RES != ncclSuccess && RES != ncclInProgress) {               \
    if (ncclDebugNoWarn == 0) INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, RES);    \
    goto label;                           \
  }                                       \
  if (__atomic_load(tmpAbortFlag, __ATOMIC_ACQUIRE)) NEQCHECKGOTO(*tmpAbortFlag, 0, RES, label); \
} while (!(cond))

// 7. 异步线程场景
// NCCLCHECKTHREAD(a, args)：线程函数里写 args->ret，失败时返回线程参数指针。
// CUDACHECKTHREAD(a)：CUDA 失败时设置 args->ret=ncclUnhandledCudaError 并返回。
// 目的是把线程内错误安全传回主流程。
#define NCCLCHECKTHREAD(a, args) do { \
  if (((args)->ret = (a)) != ncclSuccess && (args)->ret != ncclInProgress) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, (args)->ret); \
    return args; \
  } \
} while(0)

#define CUDACHECKTHREAD(a) do { \
  if ((a) != cudaSuccess) { \
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, args->ret); \
    args->ret = ncclUnhandledCudaError; \
    return args; \
  } \
} while(0)

#endif
