# 发展路线图

本页概述了 TuFT 的开发愿景与未来规划。

## 核心方向：Agent 场景的后训练

我们专注于 Agent 模型的后训练。RL 训练中的 rollout 阶段涉及推理、多轮对话和工具使用，这往往相对于训练阶段是异步的。我们旨在提高整体系统的吞吐量和资源效率，构建易于使用和集成到现有工作流程中的工具。

## 架构与定位

- **水平平台**：不是垂直整合的微调解决方案，而是一个灵活的平台，可以接入不同的训练框架和计算基础设施
- **代码优先的 API**：通过编程接口将 Agent 训练工作流与计算基础设施连接起来
- **AI 技术栈中的层级**：位于基础设施层（Kubernetes、云平台、GPU 集群）之上，将训练框架（PeFT、FSDP、vLLM、DeepSpeed）作为实现依赖进行集成
- **集成方式**：与现有生态系统协同工作，而非取代它们

## 近期目标（3 个月）

- **多机多卡训练**：支持使用 PeFT、FSDP、vLLM、DeepSpeed 等的分布式架构
- **云原生部署**：与 AWS、阿里云、GCP、Azure 和 Kubernetes 编排集成
- **可观测性**：具有实时日志、GPU 指标、训练进度和调试工具的监控系统
- **Serverless GPU**：面向多样化部署场景的轻量级运行时，支持多用户和多租户 GPU 资源共享以提高利用效率

## 长期目标（6 个月）

- **环境驱动的学习循环**：与 WebShop、MiniWob++、BrowserEnv、Voyager 等 Agent 训练环境的标准化接口
- **自动化流水线**：任务执行 → 反馈收集 → 数据生成 → 模型更新
- **高级 RL 范式**：RLAIF、错误回放和环境反馈机制
- **模拟沙箱**：用于快速实验的轻量级本地环境

## 开放协作

这个路线图不是固定的，而是我们与开源社区共同旅程的起点。每个功能设计都将通过 GitHub Issue 讨论、PR 和原型验证来实现。我们真诚地欢迎您提出真实的使用场景、性能瓶颈或创新想法——正是这些声音将共同定义 Agent 后训练的未来。

---

## 社区

我们欢迎社区的建议和贡献！加入我们：

- [钉钉群](https://qr.dingtalk.com/action/joingroup?code=v1,k1,UWvzO6HHSeuvRQ5WXCOMJEijadQV+hDjhMIpiVr8qCs=&_dt_no_comment=1&origin=11?)
- [Discord](https://discord.gg/BCNCaQGxBH)（在 AgentScope 服务器上）
