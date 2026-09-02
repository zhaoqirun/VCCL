/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "param.h"
#include "debug.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <pwd.h>

const char* userHomeDir() {
  //getuid() 获取当前进程的用户 ID  getpwuid() 从系统用户数据库（/etc/passwd）查找该用户的记录  返回 home 目录路径（如 /home/user），找不到则返回 NULL// 比 getenv("HOME") 更可靠，不依赖环境变量是否被正确设置
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

// 这个函数是 「配置文件 → 环境变量」的转换桥梁：输入一个 NCCL/VCCL 配置文件路径，打开后逐行解析 KEY=VALUE 格式，把每一对键值转成真正的进程环境变量（配合之前 initEnvFunc 的优先级调用顺序 + setenv 的特性，
// 实现了配置的优先级覆盖逻辑） 最终优先级（从高到低）【用户 export 的环境变量】>【高优先级配置文件】>【低优先级配置文件】>【代码硬编码默认值】│
void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char envVar[1024];
  char envValue[1024];
  size_t n = 0;
  ssize_t read;
  // getline 的正确用法细节：
  // 初始 line=NULL, n=0：getline 会自动 malloc 足够大的内存存当前行，无需手动分配
  // 返回值 read 是读取的字符数（包括行尾的 \n），-1 表示结束
  // 必须手动 free(line)（后面循环结束会做），否则内存泄漏
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[0] == '#') continue;
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Env Var Size     变量s被复用：先存「=的位置」，后存「值的起始位置」
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023,s));  // 复制 KEY 到 envVar，最多 1023 字节，留一个字节给 '\0'
    envVar[std::min(1023,s)] = '\0'; // 强制补\0（strncpy不保证结尾是\0，手动兜底）
    s++;
    strncpy(envValue, line+s, 1023);
    envValue[1023]='\0';
    // setenv(envVar, envValue, 0);  // 第三个参数是核心！ 如果环境变量已存在，不要覆盖！
    setenv(envVar, envValue, 0);
    //printf("%s : %s->%s\n", fileName, envVar, envValue);
  }
  if (line) free(line);
  fclose(file);
}

static void initEnvFunc() {
  // 优先级	路径来源	触发条件
  // 1（最高）	环境变量 NCCL_CONF_FILE 指定的绝对路径	用户显式设置了该环境变量且非空
  // 2	用户家目录下的 ~/.nccl.conf	未设置 NCCL_CONF_FILE 且能获取到用户家目录
  // 3（最低）	系统全局 /etc/nccl.conf	无条件注册（兜底配置，所有用户共用
  char confFilePath[1024];
  const char* userFile = getenv("NCCL_CONF_FILE");
  if (userFile && strlen(userFile) > 0) {
    snprintf(confFilePath, sizeof(confFilePath), "%s", userFile);
    setEnvFile(confFilePath);
  } else {
    const char* userDir = userHomeDir();
    if (userDir) {
      // snprintf 是 C/C++ 标准库（C99 引入）中的「安全格式化字符串拷贝函数」，避免缓冲区溢出
      snprintf(confFilePath, sizeof(confFilePath), "%s/.nccl.conf", userDir);
      setEnvFile(confFilePath);
    }
  }
  snprintf(confFilePath, sizeof(confFilePath), "/etc/nccl.conf");
  setEnvFile(confFilePath);
}

void initEnv() {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, initEnvFunc);
}

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
  static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  // 静态互斥锁，多线程安全地首次加载
  pthread_mutex_lock(&mutex);
  // 双检查：通过原子读判断缓存是否已填充  
  // mutex + 原子操作的组合确保多线程环境下只有一个线程执行首次解析，其他线程等待结果。
  if (__atomic_load_n(cache, __ATOMIC_RELAXED) == uninitialized) {
    // 调用底层函数获取环境变量字符串值
    const char* str = ncclGetEnv(env);
    int64_t value = deftVal;
    if (str && strlen(str) > 0) {
      errno = 0;
      // 将字符串转为整数；base=0 支持 0x（十六进制）和 0（八进制）前缀自动识别
      value = strtoll(str, nullptr, 0);
      // strtoll 溢出时 errno 被置为 ERANGE，此时回退到默认值并打印警告
      if (errno) {
        value = deftVal;
        // INFO(NCCL_ALL, ...) —— 值无效回退默认值时，用 NCCL_ALL 级别提示
        INFO(NCCL_ALL,"Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
      } else {
        // INFO(NCCL_ENV, ...) —— 用户主动设置了环境变量时，用 NCCL_ENV 级别记录
        INFO(NCCL_ENV,"%s set by environment to %lld.", env, (long long)value);
      }
    }
    // 	原子写入缓存，后续调用直接命中缓存
    __atomic_store_n(cache, value, __ATOMIC_RELAXED);
  }
  pthread_mutex_unlock(&mutex);
}

const char* ncclGetEnv(const char* name) {
  initEnv();
  return getenv(name);
}
