# Design: 昇腾 Multi-Process Mode 完整实现

**Date:** 2026-04-30
**Version:** LMCache-Ascend 0.4.3
**Reference:** LMCache 0.4.3 multiprocess implementation

---

## 1. 概述

本设计文档描述如何在 LMCache-Ascend 0.4.3 中实现完整的 Multi-Process Mode，支持昇腾 NPU 单机多卡部署，集成 vLLM 推理引擎。

Multi-Process Mode 通过 ZMQ 消息队列实现 vLLM 推理进程与 Cache Server 之间的通信，实现 KV Cache 的分布式存储和检索。

---

## 2. 架构

```
vLLM/SGLang Process (NPU)  <--ZMQ(HTTP)-->  [MessageQueueServer]  <--->  [NPUCacheEngine]
                                           (port 5555)                 |
                                                                      [ThreadPools]
                                                                         |
                                          [AffinityThreadPool]  [ThreadPoolExecutor]
                                          (STORE/RETRIEVE)      (LOOKUP/other ops)
```

---

## 3. 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| `NPUCacheContext` | `gpu_context.py` | 管理 NPU KV cache tensors 的 shape 和指针 |
| `MessageQueueServer` | `mq.py` | ZMQ ROUTER socket 服务器，处理客户端请求 |
| `MessageQueueClient` | `mq.py` | 客户端，连接 message queue server |
| `AffinityThreadPool` | `affinity_pool.py` | 亲和性线程池，相同 affinity_key 路由到同一 worker |
| `SessionManager` | `session.py` | Session 管理，token 追踪和 chunk hash 计算 |
| `NPUCacheEngine` | `server.py` | 核心缓存引擎，管理 GPU KV cache 和存储 |
| `MPServerConfig` | `config.py` | 配置类 |
| `http_server.py` | HTTP 前端 | FastAPI/uvicorn HTTP 服务器（可选）|

---

## 4. 与 LMCache (CUDA) 的差异

| 功能 | LMCache (CUDA) | LMCache-Ascend (NPU) |
|------|----------------|---------------------|
| IPC Wrapper | `CudaIPCWrapper` | `AscendIPCWrapper` (已有) |
| Device API | `torch.cuda` | `torch.npu` |
| Stream | `torch.cuda.Stream` | `torch.npu.Stream` |
| Event | `torch.cuda.Event` | `torch.npu.Event` |
| Memory IPC | `_share_cuda_` / `_new_shared_cuda_` | `_share_npu_` / `_new_shared_npu_` |
| KV Transfer | `lmcache.c_ops` (CUDA kernels) | 需要适配 NPU kernels |

---

## 5. 文件结构

```
lmcache_ascend/v1/multiprocess/
├── __init__.py
├── server.py          # NPUCacheEngine, run_cache_server, parse_args
├── mq.py             # MessageQueueServer, MessageQueueClient
├── gpu_context.py    # NPUCacheContext
├── affinity_pool.py  # AffinityThreadPool
├── session.py        # Session, SessionManager
├── config.py         # MPServerConfig
├── custom_types.py   # AscendIPCWrapper (已有，需更新)
├── protocol.py       # Request types
└── http_server.py    # HTTP frontend (可选)
```

---

## 6. 实现方案

### 6.1 配置模块 (config.py)

- `MPServerConfig`: host, port, chunk_size, max_workers, max_gpu_workers, max_cpu_workers, hash_algorithm
- `add_mp_server_args()`: 添加命令行参数
- `parse_args_to_mp_server_config()`: 解析参数

### 6.2 Session 管理 (session.py)

直接复用 LMCache 实现，无 device-specific 依赖。

- `Session`: 追踪每个请求的 token IDs 和 chunk hashes
- `SessionManager`: 线程安全的 session 管理，支持 TTL 清理

### 6.3 亲和性线程池 (affinity_pool.py)

直接复用 LMCache 实现，无 device-specific 依赖。

- `AffinityThreadPool`: 相同 affinity_key 的任务路由到同一 worker 线程

### 6.4 GPU Context (gpu_context.py)

基于 `GPUCacheContext` 适配 NPU：

- 使用 `torch.npu` 替代 `torch.cuda`
- 使用 `AscendIPCWrapper` 替代 `CudaIPCWrapper`
- 使用 NPU Stream 替代 CUDA Stream
- 调用 NPU kernel (`lmc_ops`) 进行 KV transfer

主要类：
- `NPUCacheContext`: 管理 NPU KV cache tensors
- `unwrap_kv_cache_tensors()`: 将 IPC wrapper 转换为 tensor
- `list_to_gpu_tensor()`: 创建 GPU tensor

### 6.5 消息队列 (mq.py)

基于 LMCache 实现，主要修改：

- 使用 `AscendIPCWrapper` 编码/解码
- 确保 NPU 可用性检查

主要类：
- `MessageQueueServer`: ZMQ ROUTER socket 服务器
- `MessageQueueClient`: ZMQ DEALER socket 客户端

### 6.6 缓存引擎 (server.py)

基于 `MPCacheEngine` 适配 NPU：

- `NPUCacheEngine`: 核心缓存引擎
  - `register_kv_cache()`: 注册 NPU KV cache
  - `store()`: 存储 KV cache (D2H)
  - `retrieve()`: 检索 KV cache (H2D)
  - `lookup()`: 前缀查找
  - `query_prefetch_status()`: 查询预取状态

- `run_cache_server()`: 启动缓存服务器

### 6.7 自定义类型 (custom_types.py)

已有 `AscendIPCWrapper`，可能需要添加：
- NPU 特定的 BlockAllocationRecord
- 其他 NPU 特定类型

### 6.8 协议定义 (protocol.py)

直接复用 LMCache 的 protocol definitions：
- `RequestType`: REGISTER_KV_CACHE, STORE, RETRIEVE, LOOKUP, etc.
- `HandlerType`: SYNC, BLOCKING, NON_BLOCKING

---

## 7. 实现顺序

1. `config.py` - 配置类
2. `session.py` - Session 管理（无依赖）
3. `affinity_pool.py` - 线程池（无依赖）
4. `protocol.py` - 协议定义（无依赖）
5. `custom_types.py` - AscendIPCWrapper 更新
6. `gpu_context.py` - NPUCacheContext
7. `mq.py` - MessageQueue
8. `server.py` - NPUCacheEngine
9. `http_server.py` - HTTP 前端（可选）

---

## 8. 测试策略

1. **单元测试**: 每个模块独立测试
2. **集成测试**: 完整的 store/retrieve 流程测试
3. **多进程测试**: 模拟多个 vLLM 实例连接
4. **NPU 特定测试**: 确保 NPU IPC 正常工作

---

## 9. 风险与注意事项

1. **NPU Kernel**: `lmc_ops` 需要有 NPU 实现或适配
2. **Stream 同步**: NPU stream 与 CUDA stream 行为可能不同
3. **IPC 性能**: NPU IPC 性能需要验证
4. **错误处理**: 确保 NPU 错误正确传播
