# Roadmap

This page outlines our development vision and plans for TuFT.

## Core Focus: Post-Training for Agent Scenarios

We focus on post-training for agentic models. The rollout phase in RL training involves reasoning, multi-turn conversations, and tool use, which tends to be asynchronous relative to the training phase. We aim to improve the throughput and resource efficiency of the overall system, building tools that are easy to use and integrate into existing workflows.

## Architecture & Positioning

- **Horizontal platform**: Not a vertically integrated fine-tuning solution, but a flexible platform that plugs into different training frameworks and compute infrastructures
- **Code-first API**: Connects agentic training workflows with compute infrastructure through programmatic interfaces
- **Layer in AI stack**: Sits above the infrastructure layer (Kubernetes, cloud platforms, GPU clusters), integrating with training frameworks (PeFT, FSDP, vLLM, DeepSpeed) as implementation dependencies
- **Integration approach**: Works with existing ecosystems rather than replacing them

## Near-Term (3 months)

- **Multi-machine, multi-GPU training**: Support distributed architectures using PeFT, FSDP, vLLM, DeepSpeed, etc.
- **Cloud-native deployment**: Integration with AWS, Alibaba Cloud, GCP, Azure and Kubernetes orchestration
- **Observability**: Monitoring system with real-time logs, GPU metrics, training progress, and debugging tools
- **Serverless GPU**: Lightweight runtime for diverse deployment scenarios, with multi-user and multi-tenant GPU resource sharing to improve utilization efficiency

## Long-Term (6 months)

- **Environment-driven learning loop**: Standardized interfaces with WebShop, MiniWob++, BrowserEnv, Voyager and other agent training environments
- **Automated pipeline**: Task execution → feedback collection → data generation → model updates
- **Advanced RL paradigms**: RLAIF, Error Replay, and environment feedback mechanisms
- **Simulation sandboxes**: Lightweight local environments for rapid experimentation

## Open Collaboration

This roadmap is not fixed, but rather a starting point for our journey with the open source community. Every feature design will be implemented through GitHub Issue discussions, PRs, and prototype validation. We sincerely welcome you to propose real-world use cases, performance bottlenecks, or innovative ideas—it is these voices that will collectively define the future of Agent post-training.

---

## Community

We welcome suggestions and contributions from the community! Join us on:

- [DingTalk Group](https://qr.dingtalk.com/action/joingroup?code=v1,k1,UWvzO6HHSeuvRQ5WXCOMJEijadQV+hDjhMIpiVr8qCs=&_dt_no_comment=1&origin=11?)
- [Discord](https://discord.gg/wEahC7ZJ) (on AgentScope's Server)
