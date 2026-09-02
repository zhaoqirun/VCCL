/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CORE_H_
#define NCCL_CORE_H_

#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm> // For std::min/std::max
#include "nccl.h"

// // #ifdef PROFAPI  // 如果定义了 PROFAPI 宏（性能分析模式）
//     // 复杂的函数包装定义
// 性能分析：允许在调用原始函数前后插入计时或统计代码
// 函数拦截：通过弱符号机制，用户可以用自己的实现覆盖默认函数
// 保持兼容性：同时提供原始 API 和 profiler API

// #else           // 正常模式
//     // 简单的函数声明
// #endif

//如果定义了 PROFAPI 宏（性能分析模式）
#ifdef PROFAPI
#define NCCL_API(ret, func, args...)        \
    // ******同时声明两个符号**********：pfunc（别名）和 func（弱符号）

    __attribute__ ((visibility("default"))) \
        //为一个函数创建别名，多个符号名指向同一个函数实现。
        // # - 字符串化运算符
    __attribute__ ((alias(#func)))          \
        // ## - 宏连接运算符 // 如果 func 是 ncclSend，则展开为pncclSend
        // pfunc 是 profiler 工具使用的钩子入口，通过 alias 指向真正的实现
        // ret 在这个宏定义中是一个宏参数，代表返回值类型（return type）
    ret p##func (args);                      \  
        // // 告诉 C++ 编译器使用 C 语言的命名规则，而不是 C++ 的名称修饰（name mangling）
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
        // 作用：定义弱符号，func 标记为 weak，允许 profiler 库提供自己的强符号实现来拦截调用
    __attribute__ ((weak))                  \
    ret func(args)
#else
#define NCCL_API(ret, func, args...)        \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    ret func(args)
#endif // end PROFAPI

#include "debug.h"
#include "checks.h"
#include "cudawrap.h"
#include "alloc.h"
#include "utils.h"
#include "param.h"
#include "nvtx.h"

#endif // end include guard
