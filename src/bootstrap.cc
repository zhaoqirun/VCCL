/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "utils.h"
#include "bootstrap.h"
#include "net.h"
#include <unistd.h>
#include <sys/types.h>
#include "proxy.h"
#include "param.h"
#include "ras.h"

#define BOOTSTRAP_N_CHECK_ABORT           10000
#define BOOTSTRAP_TAG_CONNECT             (0x1 << 31)
#define BOOTSTRAP_TAG_ALLGATHER           (0x1 << 30)
#define BOOTSTRAP_TAG_COMMSPLIT           (0x1 << 29)
#define BOOTSTRAP_TAG_INTRANODE_ALLGATHER (0x1 << 28)

#define BOOTSTRAP_INIT_TIME_CREATE 0
#define BOOTSTRAP_INIT_TIME_SEND   1
#define BOOTSTRAP_INIT_TIME_RECV   2
#define BOOTSTRAP_INIT_TIME_RING   3
#define BOOTSTRAP_INIT_TIME_TOTAL  4
#define BOOTSTRAP_INIT_TIME_DELAY  5
#define BOOTSTRAP_INIT_TIME_N      6
#define BOOTSTRAP_INIT_ROOT_WAIT   0
#define BOOTSTRAP_INIT_ROOT_SEND   1
#define BOOTSTRAP_INIT_ROOT_RECV   2
#define BOOTSTRAP_INIT_ROOT_N      3
#define BOOTSTRAP_PROF_OPEN(time) \
  do {                            \
    time = clockNano();           \
  } while (0)
#define BOOTSTRAP_PROF_CLOSE(time) \
  do {                             \
    time = clockNano() - time;     \
  } while (0)

#define BOOTSTRAP_PID(i, n) (((i) + (n)) % (n))
// returns the first rank associated to the root. must have root >=0
// if root >= n_roots, it does NOT assume periodicity
static int firstRankFromRoot(int root, int n_ranks, int nRoots) {
  return root * (n_ranks / nRoots) + std::min(root, n_ranks % nRoots);
}
// returns the root of a rank, must have rank >=0
// if rank >= n_ranks, it does NOT assume periodicity
static int rootIdFromRank(int rank, int nRanks, int nRoots) {
  int rmr = nRanks % nRoots; // rank mod root
  int rpr = nRanks / nRoots; // rank per root
  int D = rmr * (rpr + 1);
  if (rank < D)
    return rank / (rpr + 1);
  else
    return (rank - D) / rpr + rmr;
}
// return the number of child for a root, root will be periodized
static int nRankFromRoot(int root, int nRanks, int nRoots) {
  int ir = BOOTSTRAP_PID(root, nRoots);
  int rmr = nRanks % nRoots; // rank mod root
  int rpr = nRanks / nRoots; // rank per root
  return rpr + ((ir < rmr) ? 1 : 0);
}
// return the local id of a given rank for a given root
// root will be periodize, rank will not
static int localIdFromRoot(int rank, int root, int nRanks, int nRoots) {
  int ir = BOOTSTRAP_PID(root, nRoots);
  return rank - firstRankFromRoot(ir, nRanks, nRoots);
}
// Check if the given rank is the first rank from the root
static int isFirstFromRoot(int rank, int root, int nRanks, int nRoots) {
  return (rank == firstRankFromRoot(root, nRanks, nRoots));
}

struct bootstrapRootArgs {
  struct ncclSocket* listenSock;
  uint64_t magic;
};

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE+1];
static union ncclSocketAddress bootstrapNetIfAddr;
static int bootstrapNetInitDone = 0;
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

NCCL_PARAM(BootstrapNetEnable,"OOB_NET_ENABLE", 0);

ncclResult_t bootstrapNetInit() {
  // 负责为进程间的引导通信（OOB, Out-Of-Band）选择并缓存本地的网络接口信息
  if (bootstrapNetInitDone == 0) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetInitDone == 0) {
      // 本质上是一个 IP:端口 字符串，用来让多进程/多机的 bootstrap 引导阶段能够找到彼此
      const char* env = ncclGetEnv("NCCL_COMM_ID");
      //函数根据环境变量 NCCL_COMM_ID 是否存在，走不同的接口选择策略：
      if (env) {
        union ncclSocketAddress remoteAddr;
        // 从环境变量 NCCL_COMM_ID 解析出远程地址
        if (ncclSocketGetAddrFromString(&remoteAddr, env) != ncclSuccess) {
          WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
          pthread_mutex_unlock(&bootstrapNetLock);
          return ncclInvalidArgument;
        }
        // *******找到与远端地址同子网的本地接口 用户指定了一个远端通信地址（NCCL_COMM_ID），函数会自动选择一个与该远端地址在同一子网的本地网卡。这确保了后续 bootstrap 通信走的是正确的网络路径。
        if (ncclFindInterfaceMatchSubnet(bootstrapNetIfName, &bootstrapNetIfAddr, &remoteAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
          WARN("NET/Socket : No usable listening interface found"); //// 没有可用接口
          pthread_mutex_unlock(&bootstrapNetLock);
          return ncclSystemError;
        }
      } else {
        int nIfs = ncclFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr, MAX_IF_NAME_SIZE, 1);
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          pthread_mutex_unlock(&bootstrapNetLock);
          return ncclInvalidUsage;
        }
      }
      // 将选中的接口名和 IP 地址格式化为日志，例如：Bootstrap: Using eth0:192.168.1.100，方便调试和运维确认
      char line[SOCKET_NAME_MAXLEN+MAX_IF_NAME_SIZE+2];
      snprintf(line, sizeof(line), " %s:", bootstrapNetIfName);
      ncclSocketToString(&bootstrapNetIfAddr, line+strlen(line));
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using%s", line);
      bootstrapNetInitDone = 1;
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return ncclSuccess;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

// check abort function
static ncclResult_t checkAbort(volatile uint32_t* flag, int* cntr) {
  if ((*cntr % BOOTSTRAP_N_CHECK_ABORT) == 0) {
    if (flag && __atomic_load_n(flag, __ATOMIC_ACQUIRE)) {
      TRACE(NCCL_BOOTSTRAP, "bootstrap: abort called");
      return ncclInternalError;
    }
  }
  *cntr = (*cntr + 1) % BOOTSTRAP_N_CHECK_ABORT;
  return ncclSuccess;
}
// send/recv functions
static ncclResult_t netReg(ncclNet_t* net, void* comm, void* data, int size, void** handle) {
  NCCLCHECK(net->regMr(comm, data, size, NCCL_PTR_HOST, handle));
  return ncclSuccess;
}
static ncclResult_t netDereg(ncclNet_t* net, void* comm, void** handle) {
  NCCLCHECK(net->deregMr(comm, *handle));
  *handle = NULL;
  return ncclSuccess;
}
static ncclResult_t netIsend(ncclNet_t* net, void* sendComm, void* data, int size, void* dataHandle, int tag, void** sendReq,
                             int* done) {
  if (*done) return ncclSuccess;
  if (!*sendReq) {
    NCCLCHECK(net->isend(sendComm, data, (size_t)size, tag, dataHandle, NULL, sendReq));
  }
  if (*sendReq) {
    NCCLCHECK(net->test(*sendReq, done, NULL));
    if (*done) {
      *sendReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netIrecv(ncclNet_t* net, void* recvComm, void* data, int size, void* dataHandle, int tag, void** recvReq,
                             int* done) {
  if (*done) return ncclSuccess;
  if (!*recvReq) {
    size_t size64 = size;
    NCCLCHECK(net->irecv(recvComm, 1, &data, &size64, &tag, &dataHandle, NULL, recvReq));
  }
  if (*recvReq) {
    NCCLCHECK(net->test(*recvReq, done, NULL));
    if (*done) {
      *recvReq = NULL;
    }
  }
  return ncclSuccess;
}
static ncclResult_t netSendRecv(ncclNet_t* net, void* sendComm, void* sendData, int sendSize, void* sendDataHandle, void* recvComm,
                                void* recvData, int recvSize, void* recvDataHandle, int tag, volatile uint32_t* abortFlag) {
  int abortCounter = 0;
  int doneSend = 0, doneRecv = 0;
  void *sendReq = NULL, *recvReq = NULL;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!doneRecv) {
      NCCLCHECK(netIrecv(net, recvComm, recvData, recvSize, recvDataHandle, tag, &recvReq, &doneRecv));
    }
    if (!doneSend) {
      NCCLCHECK(netIsend(net, sendComm, sendData, sendSize, sendDataHandle, tag, &sendReq, &doneSend));
    }
  } while (!doneSend || !doneRecv);
  return ncclSuccess;
}

// Additional socket based functions, first send the size, then send the message
static ncclResult_t socketSend(struct ncclSocket* sock, void* data, int size) {
  NCCLCHECK(ncclSocketSend(sock, &size, sizeof(int)));
  if (size > 0)
    NCCLCHECK(ncclSocketSend(sock, data, size));
  return ncclSuccess;
}
static ncclResult_t socketRecv(struct ncclSocket* sock, void* data, int size) {
  int recvSize;
  NCCLCHECK(ncclSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return ncclInternalError;
  }
  int actualSize = std::min(recvSize, size);
  if (actualSize > 0)
    NCCLCHECK(ncclSocketRecv(sock, data, actualSize));
  return ncclSuccess;
}
static ncclResult_t socketSendRecv(struct ncclSocket* sendSock, void* sendData, int sendSize, struct ncclSocket* recvSock,
                                   void* recvData, int recvSize) {
  int senderRecvSize;
  NCCLCHECK(ncclSocketSendRecv(sendSock, &sendSize, sizeof(int), recvSock, &senderRecvSize, sizeof(int)));
  if (senderRecvSize > recvSize) {
    WARN("Message truncated : received %d bytes instead of %d", senderRecvSize, recvSize);
    return ncclInternalError;
  }
  NCCLCHECK(ncclSocketSendRecv(sendSock, sendData, sendSize, recvSock, recvData, std::min(recvSize, senderRecvSize)));
  return ncclSuccess;
}

union ringConnectInfo {
  union ncclSocketAddress addr;
  char handle[NCCL_NET_HANDLE_MAXSIZE];
};

struct extInfo {
  int rank;                                  // rank of the process reaching out
  int nranks;                                // total number of ranks
  int iroot;                                 // current root index
  int nroots;                                // total number of roots
  union ncclSocketAddress listenRootAddress; // address of my listenSocket for the root
  union ringConnectInfo connectInfo;
};
#define NET_HANDLE(h, rank)    ((h) + (rank * NCCL_NET_HANDLE_MAXSIZE))
#define BOOTSTRAP_HANDLE(h, i) ((struct ncclBootstrapHandle*)((char*)h + i * NCCL_UNIQUE_ID_BYTES))

#include <sys/resource.h>

static ncclResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return ncclSuccess;
}

static ncclResult_t rootSend(union ncclSocketAddress* addr, uint64_t magic, union ringConnectInfo* info) {
  ncclResult_t res = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECKGOTO(ncclSocketInit(&sock, addr, magic, ncclSocketTypeBootstrap), res, fail);
  NCCLCHECKGOTO(ncclSocketConnect(&sock), res, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(union ringConnectInfo)), res, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return res;
fail:
  (void)ncclSocketClose(&sock);
  return res;
}
// bootstrapRoot 位于 bootstrap.cc，它的核心逻辑是一个两阶段协调协议：
// 阶段一：收集 (Collect)
// ┌──────────────────────────────────────────────────────┐
// │ 循环 accept 所有 Rank 的连接                          │
// │ 每个 Rank 发来: {rank, 自己的监听地址, 网络 handle}    │
// │ 收集到 rankInfo[] 和 rankAddressesRoot[] 数组中       │
// └──────────────────────────────────────────────────────┘
//                           │
//                           ▼
// 阶段二：分发 (Distribute)  —— 构建 Ring 拓扑
// ┌──────────────────────────────────────────────────────┐
// │ 对每个 Rank，告诉它：                                 │
// │   "你的下一个 Rank 的连接信息是 XXX"                   │
// │                                                     │
// │  例如 Rank 0 收到 Rank 1 的地址，Rank 1 收到 Rank 2 的 │
// │  以此类推，最后一个 Rank 收到 Rank 0 的（环形拓扑）     │
static void* bootstrapRoot(void* rargs) {
  uint64_t timers[BOOTSTRAP_INIT_ROOT_N] = {0};
  struct bootstrapRootArgs* args = (struct bootstrapRootArgs*)rargs;
  struct ncclSocket* listenSock = args->listenSock;
  uint64_t magic = args->magic;
  ncclResult_t res = ncclSuccess;
  int nranks = 0, c = 0;
  int iroot = 0, nroots = 0, localId = 0;
  int nrecv = 0, n2send = 0;
//   struct extInfo {                              // 每个 Rank 发给 Root 的信息
//   int rank;                                   // 我的 Rank 编号
//   int nranks;                                 // 总 Rank 数
//   int iroot;                                  // 当前 Root 索引
//   int nroots;                                 // 总 Root 数
//   union ncclSocketAddress listenRootAddress;  // 我的监听地址（供 Root 回连我）
//   union ringConnectInfo connectInfo;          // 我的 Ring 连接信息（网络 handle 等）
// };
  struct extInfo info;
  union ringConnectInfo* rankInfo = NULL;
  union ncclSocketAddress* rankAddressesRoot = NULL; // for initial rank <-> root information exchange
  // get zeros for comparison
  char zeroHandle[NCCL_NET_HANDLE_MAXSIZE];
  union ncclSocketAddress zeroAddress;
  union ringConnectInfo zeroInfo;
  memset(&zeroAddress, 0, sizeof(union ncclSocketAddress));
  memset(&zeroHandle, 0, NCCL_NET_HANDLE_MAXSIZE);
  memset(&zeroInfo, 0, sizeof(union ringConnectInfo));
  setFilesLimit();

  TRACE(NCCL_BOOTSTRAP, "BEGIN");
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
  /* Receive addresses from all ranks */
  // 收集（边收边发）  第一个 Rank 连接上来时，从 info 中解析出 nranks、nroots、iroot 等全局参数，据此分配 rankInfo[] 和 rankAddressesRoot[] 数组。
  // Root 循环 accept 每个 Rank 的连接，收到 extInfo                 │                                                             │
  // │  Rank 0 ──→ extInfo ──→ Root 存入 rankInfo[0], rankAddressesRoot[0] │  
  do {
    struct ncclSocket sock;
    NCCLCHECKGOTO(ncclSocketInit(&sock), res, out);
    // 阻塞等待一个 Rank 连接
    NCCLCHECKGOTO(ncclSocketAccept(&sock, listenSock), res, out);
    NCCLCHECKGOTO(socketRecv(&sock, &info, sizeof(info)), res, out);
    NCCLCHECKGOTO(ncclSocketClose(&sock), res, out);

    if (c == 0) {
      BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_WAIT]);
      BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_RECV]);
      nranks = info.nranks;
      iroot = info.iroot;
      nroots = info.nroots;
      // if the number of root > 1, we will receive one extra info from the first local_id of the next root
      n2send = nRankFromRoot(iroot, nranks, nroots);
      nrecv = n2send + ((nroots > 1) ? 1 : 0);
      NCCLCHECKGOTO(ncclCalloc(&rankInfo, nrecv), res, out);
      NCCLCHECKGOTO(ncclCalloc(&rankAddressesRoot, nrecv), res, out);
    }

    if (nranks != info.nranks || nroots != info.nroots || iroot != info.iroot) {
      WARN("Bootstrap Root : mismatch in info from procs, nranks %d vs %d, nroots %d vs %d, iroot %d vs %d", nranks, info.nranks, nroots, info.nroots, iroot, info.iroot);
      goto out;
    }

    localId = localIdFromRoot(info.rank, iroot, nranks, nroots);
    if (memcmp(&zeroAddress, &rankAddressesRoot[localId], sizeof(union ncclSocketAddress)) != 0 ||
        memcmp(&zeroInfo, &rankInfo[localId], sizeof(union ringConnectInfo)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in", info.rank, nranks);
      goto out;
    }
    // if the previous has already checked in, send the newly received handle, if not save the handle for later
    // if we have more than 1 root, I do not own the previous of local_id = 0
    // if we have prev > n2send, we do not send anything
    // 前驱转发：如果 Rank[prev] 已经到了，把当前 Rank 的信息发给它
    int prev = (nroots > 1) ? (localId - 1) : BOOTSTRAP_PID(localId - 1, nrecv);
    if (prev >= 0 && prev < n2send && memcmp(&zeroAddress, &rankAddressesRoot[prev], sizeof(union ncclSocketAddress)) != 0) {
      // 含义：告诉 Rank[prev]："你的下一个是 Rank[curr]，这是它的连接信息"
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[prev], magic, &info.connectInfo), res, out);
    } else {
      memcpy(&rankInfo[localId], &info.connectInfo, sizeof(union ringConnectInfo));
    }
    // if the next rank has checked in, send the newly received info, if not save the addr for later
    // for nroots >=1, I will always own the information of the next connection
    // if the local_id id must be [0 ; n2send[ otherwise we do not answer
    // 后继转发：如果 Rank[next] 已经到了，把它的信息发给当前 Rank
    int next = BOOTSTRAP_PID(localId + 1, nrecv);
    if (localId >= 0 && localId < n2send && memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&info.listenRootAddress, magic, &rankInfo[next]), res, out);
    } else {
      memcpy(rankAddressesRoot + localId, &info.listenRootAddress, sizeof(union ncclSocketAddress));
    }
    ++c;
    TRACE(NCCL_BOOTSTRAP, "Received connect from rank %d total %d/%d", info.rank, c, nrecv);
  } while (c < nrecv);
  TRACE(NCCL_BOOTSTRAP, "COLLECTED ALL %d HANDLES", nrecv);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_RECV]);

  // send the remaining info to the ranks who haven't received anything
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  // here we need to send info only to my own local process
  // 补发剩余 处理极端情况：比如 Rank 0 最后到达，阶段一中 prev=-1 不触发，需要阶段二补发。
  for (int r = 0; r < n2send; ++r) {
    // use nrecv to periodize: if 1 root, we will send the first one to the last one, if >1 roots we will send the additional one we have received
    int next = BOOTSTRAP_PID(r + 1, nrecv);
    if (memcmp(&zeroAddress, &rankAddressesRoot[r], sizeof(union ncclSocketAddress)) != 0 &&
        memcmp(&zeroInfo, &rankInfo[next], sizeof(union ringConnectInfo)) != 0) {
      NCCLCHECKGOTO(rootSend(&rankAddressesRoot[r], magic, &rankInfo[next]), res, out);
    }
  }
  //         Rank 0 ──→ Rank 1 ──→ Rank 2 ──→ Rank 3
  //           ↑                                  │
  //           └──────────────────────────────────┘
          
  // 每个 Rank 收到的信息: "你的下一个是 Rank[(i+1)%n]，这是它的连接信息"
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_ROOT_SEND]);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "Root timings (wait %f, recv %f, send %f)", timers[BOOTSTRAP_INIT_ROOT_WAIT] / 1e9, timers[BOOTSTRAP_INIT_ROOT_RECV] / 1e9, timers[BOOTSTRAP_INIT_ROOT_SEND] / 1e9);
out:
  if (listenSock != NULL) {
    (void)ncclSocketClose(listenSock);
    free(listenSock);
  }
  if (rankInfo)
    free(rankInfo);
  if (rankAddressesRoot)
    free(rankAddressesRoot);
  free(rargs);

  TRACE(NCCL_BOOTSTRAP, "DONE");
  return NULL;
}

ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) {
  // NCCL bootstrap 协议中创建集合点（Rendezvous）服务器的核心函数。它的职责简单明了：启动一个监听 Socket，
  // 并派生一个后台线程来协调所有 Rank 的初次握手
  // 它在本机开一个监听端口（大门），启动一个后台线程（前台服务员），然后返回实际地址（旅馆地址 + 房间号）。之后所有 Rank 拿着这个地址来入住，服务员（bootstrapRoot 线程）协调大家完成 Ring 拓扑的握手，握手完成后服务员下班（线程退出）。
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket* listenSock = NULL;
  struct bootstrapRootArgs* args = NULL;
  pthread_t thread;

  NCCLCHECK(ncclCalloc(&listenSock, 1));
  // 用 handle 中的地址和 magic 初始化 Socket    魔数，用于验证连接合法性，防止误连
  NCCLCHECKGOTO(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, NULL, 0), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(listenSock), ret, fail);
  // 回填实际绑定的地址到 handle    ：ncclSocketListen 之后立即调用 ncclSocketGetAddr 回填 handle->addr。这是因为当端口号指定为 0 时，操作系统会分配一个随机可用端口——此时必须把实际端口号写回 handle，后续其他 Rank 才能用正确的地址连接过来。
  NCCLCHECKGOTO(ncclSocketGetAddr(listenSock, &handle->addr), ret, fail);

  NCCLCHECKGOTO(ncclCalloc(&args, 1), ret, fail);
  args->listenSock = listenSock;
  args->magic = handle->magic;
  // 这里把 listenSock 和 magic 打包传给 bootstrapRoot 线程，然后 detach 掉——这意味着调用者不需要等待线程结束，线程会在所有 Rank 握手完成后自行退出。
  PTHREADCHECKGOTO(pthread_create(&thread, NULL, bootstrapRoot, (void*)args), "pthread_create", ret, fail);
  ncclSetThreadName(thread, "NCCL BootstrapR");
  PTHREADCHECKGOTO(pthread_detach(thread), "pthread_detach", ret, fail); // will not be pthread_join()'d
exit:
  return ret;
fail:
  if (listenSock) free(listenSock);
  if (args) free(args);
  goto exit;
}

ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle) {
  memset(handle, 0, sizeof(ncclBootstrapHandle));

  const char* env = ncclGetEnv("NCCL_COMM_ID");
  if (env) {
    // 情况A：环境变量已设置 → 只填 handle，不创建 Root
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    if (ncclSocketGetAddrFromString(&handle->addr, env) != ncclSuccess) {
      WARN("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>");
      return ncclInvalidArgument;
    }
    handle->magic = NCCL_MAGIC;
  } else {
    // 情况B：环境变量未设置 → 生成随机 magic，使用本地网络接口地址，并创建 Root
    NCCLCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));
    memcpy(&handle->addr, &bootstrapNetIfAddr, sizeof(union ncclSocketAddress));
    NCCLCHECK(bootstrapCreateRoot(handle, false));
  }

  return ncclSuccess;
}

struct unexConn {
  int peer;
  int tag;
  struct ncclSocket sock;
  struct unexConn* next;
};

struct bootstrapRing_t {
  union {
    struct {
      void *sendComm, *recvComm;
      ncclNetDeviceHandle_t *sendDevHandle, *recvDevHandle;
    } net;
    struct {
      struct ncclSocket recv;
      struct ncclSocket send;
    } socket;
  };
};
struct bootstrapListen_t {
  struct ncclSocket peerSocket; // socket for peers to contact me in P2P
  union {
    struct {
      int dev;
      void* comm;
      char handle[NCCL_NET_HANDLE_MAXSIZE];
    } net;
    struct ncclSocket socket; // socket to be used for the ring
  };
};

struct bootstrapState {
  struct bootstrapRing_t ring;
  struct bootstrapListen_t listen;
  ncclNet_t* net;
  uint64_t* peerProxyAddressesUDS;
  union ncclSocketAddress* peerProxyAddresses;
  union ncclSocketAddress* peerP2pAddresses;
  struct unexConn* unexpectedConnections;
  int cudaDev;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t* abortFlag;
};
#define STATE_RING(s, f) (s->ring.f)
#define STATE_LISTEN(s, f) (s->listen.f)

// helper functions
static ncclResult_t createListenSocket(struct ncclComm* comm, uint64_t magic, struct ncclSocket* socket, union ncclSocketAddress* addr,
                                       ncclSocketType type) {
  NCCLCHECK(ncclSocketInit(socket, &bootstrapNetIfAddr, magic, type, comm->abortFlag));
  NCCLCHECK(ncclSocketListen(socket));
  NCCLCHECK(ncclSocketGetAddr(socket, addr));
  return ncclSuccess;
}
static ncclResult_t getUDS(uint64_t* peerUDS) {
  uint64_t randId;
  NCCLCHECK(getRandomData(&randId, sizeof(randId)));
  *peerUDS = getPidHash() + randId;
  return ncclSuccess;
}
#define MAX_OOB_DEVS 16
static ncclResult_t netGetDevice(int rank, struct ncclComm* comm, int* dev) {
  static int devOOB = -1;
  if (devOOB < 0) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (devOOB < 0) {
      const char* userIfEnv = ncclGetEnv("NCCL_OOB_NET_IFNAME");
      if (userIfEnv && strlen(userIfEnv) > 0) {
        INFO(NCCL_BOOTSTRAP | NCCL_ENV, "NCCL_OOB_NET_IFNAME set to %s", userIfEnv);
        bool searchNot = userIfEnv && userIfEnv[0] == '^';
        if (searchNot) userIfEnv++;
        bool searchExact = userIfEnv && userIfEnv[0] == '=';
        if (searchExact) userIfEnv++;
        struct netIf userIfs[MAX_OOB_DEVS];
        int nUserIfs = parseStringList(userIfEnv, userIfs, MAX_OOB_DEVS);
        // loop over the device and return the first one matching
        int nDev = 0;
        NCCLCHECK(comm->ncclNet->devices(&nDev));
        int devId = 0;
        while (devId < nDev) {
          ncclNetProperties_t props;
          comm->ncclNet->getProperties(devId, &props);
          // check against user specified HCAs/ports
          if (matchIfList(props.name, props.port, userIfs, nUserIfs, searchExact) ^ searchNot) {
            // All plain physical devices have been initialized at this point
            devOOB = devId;
            break;
          }
          devId++;
        }
        if (devOOB == -1) {
          if (!searchNot)
            WARN("no device found matching %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          else
            WARN("no device found after excluding %s%s, verify NCCL_OOB_NET_IFNAME", searchExact ? "exactly " : "", userIfEnv);
          pthread_mutex_unlock(&bootstrapNetLock);
          return ncclInvalidArgument;
        }
      } else {
        // default choice is device 0
        devOOB = 0;
      }
      // display info on the chosen device
      ncclNetProperties_t props;
      ncclResult_t res = comm->ncclNet->getProperties(devOOB, &props);
      bool hasProp = res == ncclSuccess;
      INFO(NCCL_BOOTSTRAP, "Bootstrap: Using %s:%d", (hasProp) ? props.name : "N/A", (hasProp) ? props.port : -1);
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  *dev = devOOB;
  return ncclSuccess;
}

static ncclResult_t netRingConnect(ncclNet_t* net, struct bootstrapListen_t* listen, char peerHandle[NCCL_NET_HANDLE_MAXSIZE],
                                   void** sendComm, ncclNetDeviceHandle_t** sendDevHandle,
                                   void** recvComm, ncclNetDeviceHandle_t** recvDevHandle, volatile uint32_t* abortFlag) {

  int abortCounter = 0;
  do {
    NCCLCHECK(checkAbort(abortFlag, &abortCounter));
    if (!*sendComm)
      NCCLCHECK(net->connect(listen->net.dev, NULL, peerHandle, sendComm, sendDevHandle));
    if (!*recvComm)
      NCCLCHECK(net->accept(listen->net.comm, recvComm, recvDevHandle));
  } while (!*sendComm || !*recvComm);
  return ncclSuccess;
}
static ncclResult_t socketRingConnect(ncclSocketAddress* addr, struct ncclSocket* sendSocket, struct ncclSocket* listenSock, struct ncclSocket* recvSocket, uint64_t magic, volatile uint32_t* abortFlag) {
  NCCLCHECK(ncclSocketInit(sendSocket, addr, magic, ncclSocketTypeBootstrap, abortFlag));
  NCCLCHECK(ncclSocketConnect(sendSocket));
  NCCLCHECK(ncclSocketInit(recvSocket));
  NCCLCHECK(ncclSocketAccept(recvSocket, listenSock));
  return ncclSuccess;
}
static ncclResult_t ringAllInfo(struct ncclComm* comm, struct bootstrapState* state,
                                union ncclSocketAddress* peerAddresss,
                                union ncclSocketAddress* peerProxy, uint64_t* peerUDS,
                                struct rasRankInit* rasRanks) {
  ncclResult_t res = ncclSuccess;
  int rank = comm->rank;
  int nRanks = comm->nRanks;
  struct bootstrapRingData {
    union ncclSocketAddress peerAddress;
    union ncclSocketAddress peerProxy;
    uint64_t peerUDS;
    struct rasRankInit rasRank;
  }* ringData = NULL;

  NCCLCHECK(ncclCalloc(&ringData, nRanks));
  // pack
  if (peerAddresss)
    memcpy(&(ringData[rank].peerAddress), peerAddresss + rank, sizeof(union ncclSocketAddress));
  if (peerProxy)
    memcpy(&(ringData[rank].peerProxy), peerProxy + rank, sizeof(union ncclSocketAddress));
  if (peerUDS)
    memcpy(&(ringData[rank].peerUDS), peerUDS + rank, sizeof(uint64_t));
  if (rasRanks)
    memcpy(&(ringData[rank].rasRank), rasRanks + rank, sizeof(*rasRanks));

  // allgather
  NCCLCHECKGOTO(bootstrapAllGather(state, ringData, sizeof(struct bootstrapRingData)), res, exit);

  // unpack
  for (int irank = 0; irank < nRanks; ++irank) {
    if (peerAddresss)
      memcpy(peerAddresss + irank, &(ringData[irank].peerAddress), sizeof(union ncclSocketAddress));
    if (peerProxy)
      memcpy(peerProxy + irank, &(ringData[irank].peerProxy), sizeof(union ncclSocketAddress));
    if (peerUDS)
      memcpy(peerUDS + irank, &(ringData[irank].peerUDS), sizeof(uint64_t));
    if (rasRanks)
      memcpy(rasRanks + irank, &(ringData[irank].rasRank), sizeof(*rasRanks));
  }

exit:
  free(ringData);
  return ncclSuccess;
}

static ncclResult_t sendToRoot(struct ncclBootstrapHandle* handle, struct ncclComm* comm, struct extInfo* info) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  NCCLCHECK(ncclSocketInit(&sock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, comm->abortFlag));
  NCCLCHECKGOTO(ncclSocketConnect(&sock), ret, fail);
  NCCLCHECKGOTO(socketSend(&sock, info, sizeof(struct extInfo)), ret, fail);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

NCCL_PARAM(StaggerRate, "UID_STAGGER_RATE", 7000);
NCCL_PARAM(StaggerThreshold, "UID_STAGGER_THRESHOLD", 256);

NCCL_PARAM(RasEnable, "RAS_ENABLE", 1);

// bootstrapInit() 函数的流程图，描述了各个阶段的主要操作和顺序。
// ┌─ bootstrapInit() ─────────────────────────────────────────────────┐
// │                                                                  │
// │  阶段1: CREATE  ─ 创建监听资源                                    │
// │    ├─ 创建 Ring 监听 Socket（或 Net 监听）                        │
// │    └─ 创建 Root 回连监听 Socket                                   │
// │                                                                  │
// │  阶段2: DELAY   ─ 错峰连接（避免 Root 过载）                      │
// │    └─ 大规模场景下，每个 Rank 按 localId 递增延迟连接             │
// │                                                                  │
// │  阶段3: SEND    ─ 向 Root 发送自己的连接信息                      │
// │    ├─ 发送 extInfo 给自己的 Root                                  │
// │    └─ 如果自己是 Root 的第一个 Rank，额外发给前一个 Root           │
// │                                                                  │
// │  阶段4: RECV    ─ 从 Root 接收下一个 Rank 的连接信息               │
// │    └─ 阻塞等待 Root 回连，收到 nextPeer 信息                      │
// │                                                                  │
// │  阶段5: RING_CONNECT ─ 建立 Ring 连接                             │
// │    ├─ 主动连接 nextPeer（发送端）                                  │
// │    └─ 接受上一 Rank 的连接（接收端）                               │
// │                                                                  │
// │  阶段6: AllGather ─ 通过 Ring 交换所有 Rank 的信息                 │
// │    ├─ 交换 Proxy 地址（TCP）                                      │
// │    ├─ 交换 Proxy 地址（UDS）                                      │
// │    ├─ 交换 P2P 地址                                               │
// │    └─ 交换 RAS 信息                                               │
// │                                                                  │
// │  阶段7: FINALIZE ─ 收尾工作                                       │
// │    ├─ ncclProxyInit() 初始化 Proxy 服务                           │
// │    └─ ncclRasAddRanks() 注册 RAS                                 │
// └──────────────────────────────────────────────────────────────────┘
// bootstrapInit() 负责在一个 ncclComm 内完成 Bootstrap 通信层初始化。它不处理真正的 GPU collective 数据，而是先让所有 rank：
// 1. 通过 Root 协调得到环中相邻 rank 的连接信息。
// 2. 建立用于 bootstrap AllGather 的环形连接。
// 3. 通过该环交换后续控制面所需的地址和元数据。
// 4. 初始化 Proxy 与可选的 RAS。
// 最终结果是 comm->bootstrap 指向已完成初始化的 bootstrapState，其中保存了 ring 通信端点、P2P 地址表、Proxy 地址表等。
ncclResult_t bootstrapInit(int nHandles, void* handles, struct ncclComm* comm) {
  ncclResult_t result = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  // char nextPeerHandle[NCCL_NET_HANDLE_MAXSIZE];
  struct bootstrapState* state;
  struct ncclSocket* proxySocket;
  // sock：临时 socket（用于从 Root 接收 next rank 信息） listenSockRoot：给 Root 回连我的监听 socket
  struct ncclSocket sock, listenSockRoot;
  // info：本 rank 上报给 Root 的 extInfo
  struct extInfo info = {0};
  // nextPeer：从 Root 收到的、环后继 rank 的连接信息
  union ringConnectInfo nextPeer;
  bool performRasAddRanks = true;
  struct rasRankInit* rasRanks = nullptr;

  uint64_t timers[BOOTSTRAP_INIT_TIME_N] = {0};

  // 把上下文写入 state，并挂到 comm->bootstrap，使后续 bootstrapSend/Recv/AllGather 能访问。
  // magic 取第一个 handle 的值，作为 socket 握手校验，避免误连
  NCCLCHECK(ncclCalloc(&state, 1));
  // 建立状态对象 state：  本 communicator 的 bootstrap 私有状态，贯穿整个生命周期
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;
  // magic 是 socket 握手的校验值，用来避免不同 communicator 或误连进程之间串线。
  comm->magic = state->magic = BOOTSTRAP_HANDLE(handles, 0)->magic; // state and comm magic set to the first magic ID

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d", rank, nranks);

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  // fill up the info
  info.nranks = nranks;
  info.nroots = nHandles;
  // get the ring connection info
  memset(&nextPeer, 0, sizeof(union ringConnectInfo));
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_CREATE]);

  // 端点一：ring 监听端点（别的 rank 用它连我）。根据 OOB_NET_ENABLE 走网络插件或普通 socket，并把对外连接信息写入 info.connectInfo（union，两种模式分别用 handle 或 addr）。
  if (ncclParamBootstrapNetEnable()) {
    // Create net interface for other ranks to contact me (all gather)
    NCCLCHECK(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)));
    NCCLCHECK(state->net->listen(STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle), &STATE_LISTEN(state, net.comm)));
    memcpy(info.connectInfo.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // create socket for ring neightbor to contact mee
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.connectInfo.addr, ncclSocketTypeBootstrap));
  }
  // 端点二：Root 回连端点。curr_root 是本 rank 所属 Root。用该 Root 的 magic 创建 listenSockRoot，地址写入 info.listenRootAddress，供 Root 稍后把 next rank 信息发回来
  int curr_root = rootIdFromRank(rank, nranks, nHandles);
  NCCLCHECK(createListenSocket(comm, BOOTSTRAP_HANDLE(handles, curr_root)->magic, &listenSockRoot, &info.listenRootAddress, ncclSocketTypeBootstrap));
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_CREATE]);

  // stagger connection times to avoid an overload of the root 
  // 当一个 Root 管理的 rank 超过 256 时，按 rank 在该 Root 内的局部编号 localId 递增延迟，
  // 避免大量 rank 同时冲击 Root 的 accept 队列。localId 越大，睡得越久。
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_DELAY]);
  int nRankRoot = nRankFromRoot(curr_root, nranks, nHandles);
  // 错峰连接（Stagger）
  if (nRankRoot > ncclParamStaggerThreshold()) { // 默认阈值 256
    // for socket the message rate in microsec
    double msg_rate = ncclParamStaggerRate() / 1.0e6;
    // // Rank 0 等 0ms，Rank 1 等 10ms，Rank 2 等 20ms...
    long musec = localIdFromRoot(rank, curr_root, nranks, nHandles) / msg_rate;
    struct timespec tv;
    long c_1e6 = 1e6;
    tv.tv_sec = musec / c_1e6;
    tv.tv_nsec = 1e3 * (musec % c_1e6);
    TRACE(NCCL_BOOTSTRAP, "rank %d delaying connection to root by %ld microsec", rank, musec);
    (void)nanosleep(&tv, NULL);
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_DELAY]);

  // send info on my listening socket to root  SEND 阶段：向 Root 上报
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_SEND]);
  // send contact info to my own root
  info.rank = rank;
  info.iroot = curr_root;
  // 连接自己的 Root，发送 extInfo（含 rank、监听地址、ring 连接信息）。Root 收集后会把“下一个 rank 的连接信息”回传给你。
  NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, curr_root), comm, &info));
  // 跨 Root 闭环：多 Root 时，每组第一个 rank 额外向前一个 Root 上报，让前一个 Root 能告诉它的最后一个 rank“你的后继在下一组”。这样整个环才能跨越 Root 边界连起来。
  // if needed, send the connection info to the previous root
  if (nHandles > 1 && isFirstFromRoot(rank, curr_root, nranks, nHandles)) {
    int prev_rank = BOOTSTRAP_PID(rank - 1, nranks);
    int prev_root = rootIdFromRank(prev_rank, nranks, nHandles);
    info.rank = prev_rank + 1; // my rank as seen by the previous root
    info.iroot = prev_root;
    NCCLCHECK(sendToRoot(BOOTSTRAP_HANDLE(handles, prev_root), comm, &info));
  }
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_SEND]);

  // RECV 阶段：接收 next rank 信息
  // get info on my "next" rank in the bootstrap ring from root
  // 被动等待 Root 回连，收到 nextPeer（环后继的连接信息）。listenSockRoot 一次性使命完成，随即关闭。
  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RECV]);
  NCCLCHECK(ncclSocketInit(&sock));
  NCCLCHECK(ncclSocketAccept(&sock, &listenSockRoot));
  NCCLCHECK(socketRecv(&sock, &nextPeer, sizeof(nextPeer)));
  NCCLCHECK(ncclSocketClose(&sock));
  NCCLCHECK(ncclSocketClose(&listenSockRoot));
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RECV]);

  // RING_CONNECT 阶段：建立环连接 
  // socket 版本内部逻辑：主动 connect 到 nextPeer（send 端），同时 accept 前驱 rank 的连接（recv 端）。每个 rank 同时握着 send/recv 两个方向，形成：
  // accept and connect the ring network
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECK(netRingConnect(state->net, &state->listen, nextPeer.handle,
                             &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                             &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle), state->abortFlag));
  } else {
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket), &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  // 准备 AllGather 用的地址表
  // 为三张表各自填入本 rank 的槽位（+ rank）：
  // 表	                     内容
  // peerProxyAddresses	     Proxy 的 TCP 监听地址
  // peerProxyAddressesUDS	 Proxy 的 Unix Domain Socket 唯一标识
  // peerP2pAddresses	       P2P 通信监听地址
  // 其余槽位留空，等 AllGather 补齐。
  // AllGather all listen handlers
  // in case of failure, those resources will be free'd when calling bootstrapDestroy, so we can return immediatly
  NCCLCHECK(ncclCalloc(&state->peerProxyAddresses, nranks));
  NCCLCHECK(ncclCalloc(&proxySocket, 1));
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank, ncclSocketTypeProxy), result, fail);

  NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), result, fail);
  NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), result, fail);

  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECKGOTO(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress, ncclSocketTypeBootstrap), result, fail);
  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), result, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));

  // 填入本 rank 的 RAS 信息。关键容错：即便 ncclRasCommInit 失败，也不能直接退出，因为环上其它 rank 在等你参与 AllGather；因此只把自己置零并跳过后续注册，保证主协议不被拖死。
  // Initialize RAS
  if (ncclParamRasEnable() == 1) {
    // The RAS thread will take care of freeing the memory allocated below.
    NCCLCHECK(ncclCalloc(&rasRanks, nranks));
    memcpy(&rasRanks[rank].addr, &bootstrapNetIfAddr, sizeof(rasRanks[rank].addr));
    rasRanks[rank].pid = getpid();
    // bootstrapInit 记录的是 GPU 级信息，而非纯进程级：
    rasRanks[rank].cudaDev = comm->cudaDev;
    rasRanks[rank].nvmlDev = comm->nvmlDev;
    rasRanks[rank].hostHash = getHostHash();
    rasRanks[rank].pidHash = getPidHash();
    if (ncclRasCommInit(comm, rasRanks+rank) != ncclSuccess) {
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
      // We should still participate in the ringAllInfo below as the peers will be waiting for us.
      // Just make sure that the address is clearly invalid...
      memset(rasRanks+rank, '\0', sizeof(*rasRanks));
      performRasAddRanks = false;
    }
  }

  BOOTSTRAP_PROF_OPEN(timers[BOOTSTRAP_INIT_TIME_RING]);
  // RING 阶段：一次 AllGather 交换所有信息
  // 把四类数据打包成一个结构，沿环做 nranks-1 步的 ring AllGather。完成后每个 rank 都持有全局完整的 P2P / Proxy / UDS / RAS 表。
  NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses, state->peerProxyAddressesUDS, rasRanks), result, fail);
  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_RING]);

  // FINALIZE 阶段：Proxy 与 RAS 收尾
  // ncclProxyInit：此刻已知全部 peer 的 Proxy 地址，创建服务端 Proxy。
  // ncclRasAddRanks：注册所有 rank 的 RAS 信息（前提是本 rank RAS 初始化成功）。 
  // 最后记录各阶段耗时并返回。
  // Create the service proxy and get the UDS
  NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), result, fail);

  if (ncclParamRasEnable() == 1 && performRasAddRanks) {
    if (ncclRasAddRanks(rasRanks, nranks) != ncclSuccess)
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
  }

  BOOTSTRAP_PROF_CLOSE(timers[BOOTSTRAP_INIT_TIME_TOTAL]);
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d - DONE", rank, nranks);
  INFO(NCCL_BOOTSTRAP | NCCL_PROFILE, "Bootstrap timings total %f (create %f, send %f, recv %f, ring %f, delay %f)", timers[BOOTSTRAP_INIT_TIME_TOTAL] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_CREATE] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_SEND] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RECV] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_RING] / 1e9,
       timers[BOOTSTRAP_INIT_TIME_DELAY] / 1e9);
exit:
  return result;
fail:
  free(proxySocket);
  goto exit;
}

ncclResult_t bootstrapSplit(uint64_t magic, struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* parentRanks) {
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int prev, next;
  union ringConnectInfo info;
  union ringConnectInfo nextPeer;
  struct ncclSocket* proxySocket = NULL;
  struct bootstrapState* state;

  NCCLCHECKGOTO(ncclCalloc(&state, 1), ret, fail);
  state->rank = rank;
  state->nranks = nranks;
  state->cudaDev = comm->cudaDev;
  state->abortFlag = comm->abortFlag;
  state->net = comm->ncclNet;
  comm->bootstrap = state;
  comm->magic = state->magic = magic;

  prev = parentRanks[(rank - 1 + nranks) % nranks];
  next = parentRanks[(rank + 1) % nranks];

  // create a handle for the others to reach out to me
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netGetDevice(rank, comm, &STATE_LISTEN(state, net.dev)), ret, fail);
    NCCLCHECKGOTO(state->net->listen(STATE_LISTEN(state, net.dev), STATE_LISTEN(state, net.handle), &STATE_LISTEN(state, net.comm)), ret, fail);
    memcpy(info.handle, STATE_LISTEN(state, net.handle), NCCL_NET_HANDLE_MAXSIZE);
  } else {
    // create socket for ring neightbor to contact mee
    NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, socket), &info.addr, ncclSocketTypeBootstrap));
  }
  // create a socket for others to reach out (P2P)
  union ncclSocketAddress peerSocketAddress;
  NCCLCHECK(createListenSocket(comm, comm->magic, &STATE_LISTEN(state, peerSocket), &peerSocketAddress, ncclSocketTypeBootstrap));

  if (ncclParamRasEnable() == 1) {
    if (ncclRasCommInit(comm, nullptr) != ncclSuccess)
      INFO(NCCL_INIT|NCCL_RAS, "Continuing in spite of a RAS initialization error");
  }

  // Get addr from next rank using the parent's connections
  NCCLCHECKGOTO(bootstrapSend(parent->bootstrap, prev, BOOTSTRAP_TAG_COMMSPLIT, &info, sizeof(union ringConnectInfo)), ret, fail);
  NCCLCHECKGOTO(bootstrapRecv(parent->bootstrap, next, BOOTSTRAP_TAG_COMMSPLIT, &nextPeer, sizeof(union ringConnectInfo)), ret, fail);
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netRingConnect(state->net, &state->listen, nextPeer.handle,
                                 &STATE_RING(state, net.sendComm), &STATE_RING(state, net.sendDevHandle),
                                 &STATE_RING(state, net.recvComm), &STATE_RING(state, net.recvDevHandle), state->abortFlag),
                  ret, fail);
  } else {
    NCCLCHECK(socketRingConnect(&nextPeer.addr, &STATE_RING(state, socket.send), &STATE_LISTEN(state, socket), &STATE_RING(state, socket.recv), comm->magic, state->abortFlag));
  }

  NCCLCHECKGOTO(ncclCalloc(&state->peerP2pAddresses, nranks), ret, fail);
  memcpy(state->peerP2pAddresses + rank, &peerSocketAddress, sizeof(union ncclSocketAddress));
  if (parent->config.splitShare) {
    /* map local rank to top parent local rank. */
    for (int i = 0; i < nranks; ++i) {
      comm->topParentRanks[i] = parent->topParentRanks[parentRanks[i]];
    }
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, NULL, NULL, NULL), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddresses, nranks), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&state->peerProxyAddressesUDS, nranks), ret, fail);
    // Create the service proxy and get the UDS
    NCCLCHECKGOTO(ncclCalloc(&proxySocket, 1), ret, fail);
    NCCLCHECKGOTO(getUDS(state->peerProxyAddressesUDS + rank), ret, fail);
    NCCLCHECKGOTO(createListenSocket(comm, comm->magic, proxySocket, state->peerProxyAddresses + rank, ncclSocketTypeProxy), ret, fail);
    NCCLCHECKGOTO(ringAllInfo(comm, state, state->peerP2pAddresses, state->peerProxyAddresses, state->peerProxyAddressesUDS, NULL), ret, fail);
    NCCLCHECKGOTO(ncclProxyInit(comm, proxySocket, state->peerProxyAddresses, state->peerProxyAddressesUDS), ret, fail);
  }

  TRACE(NCCL_BOOTSTRAP, "bootstrapSplit: comm %p parent %p rank %d nranks %d color %d key %d prev %d next %d - DONE", comm, parent, rank, nranks,
        color, key, prev, next);

exit:
  return ret;
fail:
  free(proxySocket);
  goto exit;
}

struct socketAckInfo {
  int rank;
  int tag;
};
static ncclResult_t socketConnect(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  struct socketAckInfo ack = (struct socketAckInfo){.rank = state->rank, .tag = tag};
  NCCLCHECKGOTO(ncclSocketInit(sock, state->peerP2pAddresses + peer, state->magic, ncclSocketTypeBootstrap, state->abortFlag), ret, fail);
  NCCLCHECKGOTO(ncclSocketConnect(sock), ret, fail);
  NCCLCHECKGOTO(socketSend(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}
ncclResult_t bootstrapSend(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret = ncclSuccess;
  struct ncclSocket sock;
  TRACE(NCCL_BOOTSTRAP, "Sending to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(socketConnect(commState, peer, tag, &sock));
  NCCLCHECKGOTO(socketSend(&sock, data, size), ret, fail);
  TRACE(NCCL_BOOTSTRAP, "Sent to peer=%d tag=%d size=%d", peer, tag, size);
  NCCLCHECK(ncclSocketClose(&sock));
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}
// Bootstrap send/receive functions
static ncclResult_t unexpectedEnqueue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock) {
  // New unex
  struct unexConn* unex;
  NCCLCHECK(ncclCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct ncclSocket));

  // Enqueue
  struct unexConn* list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return ncclSuccess;
  }
  while (list->next) list = list->next;
  list->next = unex;
  return ncclSuccess;
}
static ncclResult_t unexpectedDequeue(struct bootstrapState* state, int peer, int tag, struct ncclSocket* sock, int* found) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;
  *found = 0;
  while (elem) {
    if (elem->peer == peer && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct ncclSocket));
      free(elem);
      *found = 1;
      return ncclSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  return ncclSuccess;
}

static void unexpectedFree(struct bootstrapState* state) {
  struct unexConn* elem = state->unexpectedConnections;
  struct unexConn* prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev);
  }
  return;
}

// We can't know who we'll receive from, so we need to receive everything at once
static ncclResult_t socketAccept(void* commState, int peer, int tag, struct ncclSocket* sock) {
  ncclResult_t ret = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;

  // Search unexpected connections first
  int found;
  NCCLCHECK(unexpectedDequeue(state, peer, tag, sock, &found));
  if (found) return ncclSuccess;

  // Then look for new connections
  while (1) {
    struct socketAckInfo ack = {0};
    NCCLCHECKGOTO(ncclSocketInit(sock), ret, fail);
    NCCLCHECKGOTO(ncclSocketAccept(sock, &STATE_LISTEN(state, peerSocket)), ret, fail);
    NCCLCHECKGOTO(socketRecv(sock, &ack, sizeof(struct socketAckInfo)), ret, fail);
    if (ack.rank == peer && ack.tag == tag) return ncclSuccess;
    NCCLCHECKGOTO(unexpectedEnqueue(state, ack.rank, ack.tag, sock), ret, fail);
  }
  return ncclSuccess;
fail:
  (void)ncclSocketClose(sock);
  return ret;
}
// We can't know who we'll receive from, so we need to receive everything at once
ncclResult_t bootstrapRecv(void* commState, int peer, int tag, void* data, int size) {
  ncclResult_t ret;
  struct ncclSocket sock;
  NCCLCHECK(socketAccept(commState, peer, tag, &sock));
  TRACE(NCCL_BOOTSTRAP, "Receiving tag=%d peer=%d size=%d", tag, peer, size);
  NCCLCHECKGOTO(socketRecv(&sock, ((char*)data), size), ret, fail);
  NCCLCHECKGOTO(ncclSocketClose(&sock, /*wait*/true), ret, fail);
  return ret;
fail:
  (void)ncclSocketClose(&sock);
  return ret;
}

static ncclResult_t netRingAllGather(ncclNet_t* net, void* sendComm, void* recvComm, int rank, int nranks, char* data, int size, volatile uint32_t* abortFlag) {
  ncclResult_t res;
  uint64_t tFirst = 0, tRest = 0;
  void* sendDataHandle = NULL;
  void* recvDataHandle = NULL;
  NCCLCHECKGOTO(netReg(net, sendComm, data, nranks * size, &sendDataHandle), res, exit);
  NCCLCHECKGOTO(netReg(net, recvComm, data, nranks * size, &recvDataHandle), res, exit);
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "NetRingAllGather started");
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int i = 0; i < nranks - 1; i++) {
    int tag = i;
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;
    void* recv_data = data + rslice * size;
    void* send_data = data + sslice * size;
    NCCLCHECKGOTO(netSendRecv(net, sendComm, send_data, size, sendDataHandle, recvComm, recv_data, size, recvDataHandle, tag, abortFlag), res, exit);
    if (i == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "netRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)", tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  // do not fail in case of error, try to deregister as much as possible
  if (sendDataHandle) netDereg(net, sendComm, &sendDataHandle);
  if (recvDataHandle) netDereg(net, recvComm, &recvDataHandle);
  return res;
}
static ncclResult_t socketRingAllGather(struct ncclSocket* sendSock, struct ncclSocket* recvSock, int rank, int nranks, char* data, int size) {
  ncclResult_t res = ncclSuccess;
  uint64_t tFirst = 0, tRest = 0;
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  TRACE(NCCL_BOOTSTRAP, "socketRingAllGather started");
  BOOTSTRAP_PROF_OPEN(tFirst);
  for (int i = 0; i < nranks - 1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;
    void* recv_data = data + rslice * size;
    void* send_data = data + sslice * size;
    NCCLCHECKGOTO(socketSendRecv(sendSock, send_data, size, recvSock, recv_data, size), res, exit);
    if (i == 0) {
      BOOTSTRAP_PROF_CLOSE(tFirst);
      BOOTSTRAP_PROF_OPEN(tRest);
    }
  }
  BOOTSTRAP_PROF_CLOSE(tRest);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "socketRingAllGather first message in %f (%f MB/sec), rest in %f (%f MB/sec)", tFirst / 1e9, (size / 1e6) / (tFirst / 1e9), tRest / 1e9, (nranks - 1) * (size / 1e6) / (tRest / 1e9));
exit:
  return res;
}
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) {
  ncclResult_t res = ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather", rank, nranks, size);

  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECKGOTO(netRingAllGather(state->net, STATE_RING(state, net.sendComm), STATE_RING(state, net.recvComm), rank, nranks, (char*)allData, size, state->abortFlag), res, exit);
  } else {
    NCCLCHECKGOTO(socketRingAllGather(&STATE_RING(state, socket.send), &STATE_RING(state, socket.recv), rank, nranks, (char*)allData, size), res, exit);
  }
exit:
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapAllGather for %d B done in %f sec: %f MB/sec", size, time / 1e9, (nranks * size / 1e6) / (time / 1e9));
  TRACE(NCCL_BOOTSTRAP, "rank %d nranks %d size %d - AllGather DONE", rank, nranks, size);
  return res;
}

static ncclResult_t bootstrapP2PBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  if (nranks == 1)
    return ncclSuccess;
  /* Simple [intra] process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and Udi Manbet,
   * "Two Algorithms for Barrier Synchronization," International Journal of Parallel Programming, 17(1):1-17, 1988"
   */
  int data[1] = {0};
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    NCCLCHECK(bootstrapSend(commState, ranks ? ranks[dst] : dst, tag, data, sizeof(data)));
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[src] : src, tag, data, sizeof(data)));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, ranks, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapBarrier(void* commState, int rank, int nranks, int tag) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBarrier(commState, NULL, rank, nranks, tag));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBarrier done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeAllGather(void* commState, int* ranks, int rank, int nranks, void* allData, int size) {
  if (nranks == 1) return ncclSuccess;
  TRACE(NCCL_INIT, "rank %d nranks %d size %d - ENTER", rank, nranks, size);

  int prevRank = ranks[(rank - 1 + nranks) % nranks];
  int nextRank = ranks[(rank + 1) % nranks];
  // intraNode bootstrap is done defacto using the socket-based implementation
  struct ncclSocket recvSocket, sendSocket;
  NCCLCHECK(socketConnect(commState, nextRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &sendSocket));
  NCCLCHECK(socketAccept(commState, prevRank, BOOTSTRAP_TAG_INTRANODE_ALLGATHER, &recvSocket));

  NCCLCHECK(socketRingAllGather(&sendSocket, &recvSocket, rank, nranks, (char*)allData, size));

  NCCLCHECK(ncclSocketClose(&sendSocket));
  NCCLCHECK(ncclSocketClose(&recvSocket));

  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return ncclSuccess;
}

// [IntraNode] in-place Broadcast
static ncclResult_t bootstrapP2PBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData, int size) {
  if (nranks == 1) return ncclSuccess;
  if (rank == root) {
    for (int i = 0; i < nranks; i++) {
      if (i != root) NCCLCHECK(bootstrapSend(commState, ranks ? ranks[i] : i, /*tag=*/ranks ? ranks[i] : i, bcastData, size));
    }
  } else {
    NCCLCHECK(bootstrapRecv(commState, ranks ? ranks[root] : root, /*tag=*/ranks ? ranks[rank] : rank, bcastData, size));
  }
  return ncclSuccess;
}

ncclResult_t bootstrapIntraNodeBroadcast(void* commState, int* ranks, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, ranks, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapIntraNodeBroadcast for %d B done in %f sec: %f MB/sec", size, time / 1e9, (nranks * size / 1e6) / (time / 1e9));
  return ncclSuccess;
}
ncclResult_t bootstrapBroadcast(void* commState, int rank, int nranks, int root, void* bcastData, int size) {
  uint64_t time = 0;
  BOOTSTRAP_PROF_OPEN(time);
  NCCLCHECK(bootstrapP2PBroadcast(commState, NULL, rank, nranks, root, bcastData, size));
  BOOTSTRAP_PROF_CLOSE(time);
  TRACE(NCCL_BOOTSTRAP | NCCL_PROFILE, "bootstrapBroadcast done in %f sec", time / 1e9);
  return ncclSuccess;
}

ncclResult_t bootstrapClose(void* commState) {
  if (commState == NULL)
    return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // close unexpected and return an error if we are not aborting and still operations in the pipe
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);
    if (__atomic_load_n(state->abortFlag, __ATOMIC_ACQUIRE) == 0) {
      WARN("Unexpected connections are not empty");
      return ncclInternalError;
    }
  }
  if (ncclParamBootstrapNetEnable()) {
    NCCLCHECK(state->net->closeSend(STATE_RING(state, net.sendComm)));
    NCCLCHECK(state->net->closeRecv(STATE_RING(state, net.recvComm)));
    NCCLCHECK(state->net->closeListen(STATE_LISTEN(state, net.comm)));
  } else {
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.send)));
    NCCLCHECK(ncclSocketClose(&STATE_RING(state, socket.recv)));
    NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, socket)));
  }
  // close the p2p socket
  NCCLCHECK(ncclSocketClose(&STATE_LISTEN(state, peerSocket)));

  // proxy things are free'd elsewhere
  free(state->peerP2pAddresses);
  free(state);
  return ncclSuccess;
}

ncclResult_t bootstrapAbort(void* commState) {
  if (commState == NULL)
    return ncclSuccess;
  struct bootstrapState* state = (struct bootstrapState*)commState;
  // when aborting we need to close the proxy here (maybe?)
  free(state->peerProxyAddresses);
  free(state->peerProxyAddressesUDS);
  NCCLCHECK(bootstrapClose(commState));
  return ncclSuccess;
}
