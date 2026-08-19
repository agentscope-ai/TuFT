# LoRA 目标模块

本页说明 TuFT 如何把客户端 LoRA 标志（`train_attn`、`train_mlp`、
`train_unembed`）解析为具体的模块名，以及围绕它们的规则。

## 解析方式

TuFT 读取模型的 `config.json` 获取模型类型，并把每个标志映射为模块名。对于
Qwen 和 Llama 模型，`train_attn` 覆盖 `q_proj`/`k_proj`/`v_proj`/`o_proj`，
`train_mlp` 覆盖 `gate_proj`/`up_proj`/`down_proj`；在 Llama 上，
`train_unembed` 覆盖 `lm_head`。当本地还没有 `config.json`（例如尚未下载的
Hugging Face 模型 ID）时，由模型名决定。

对于 Qwen3.5 系列模型（模型类型 `qwen3_5`，Qwen3.6 和 Qwen3.8 也使用该类型），
每四个文本层中有三个使用 Gated DeltaNet 而不是完整注意力，因此 `train_attn`
额外覆盖 `linear_attn.in_proj_qkv`、`linear_attn.in_proj_z` 和
`linear_attn.out_proj`。

## 0.2.0 中的破坏性变更：Qwen3.5 系列模型

0.2.0 版本（issue #149 的修复）为 Qwen3.5 系列模型的 LoRA 目标模块列表新增了
与 Tinker 兼容的 `linear_attn.in_proj_qkv`、`linear_attn.in_proj_z` 和
`linear_attn.out_proj` 模块。0.2.0 之前保存的检查点和训练运行使用旧的、更短的
列表，而两份列表必须完全一致。升级到 0.2.0 后：

- 这些模型的旧检查点无法加载。
- 服务器重启时，旧的 FSDP 训练运行会被标记为损坏（corrupted）。
- 如果服务器配置中的 `fsdp_target_modules` 仍是旧列表，服务器会在启动时报错
  退出，错误信息会说明如何修改配置。
- 启用持久化时，新增的 `qwen_gated_deltanet_full_lora` 模型字段会改变已存储的
  配置签名，启动会因“配置不匹配”而失败；请切换到新命名空间或清除旧命名空间
  （参见[安全更改配置](persistence.md#安全更改配置)）。

每条错误信息都会指出原因是这项变更。要继续训练，请创建新的训练运行。

## 完整 Gated DeltaNet 覆盖（可选项）

在模型配置中设置 `qwen_gated_deltanet_full_lora: true`，可以额外为
`linear_attn.in_proj_a` 和 `linear_attn.in_proj_b` 添加 LoRA。此运维侧选项
同时应用于 HF 和 FSDP，并会产生不同于默认值的检查点模块结构；更改该选项同样
需要新建训练运行（FSDP 还需要重启）。默认值保持与 Tinker 公共 `train_attn`
行为一致。

## MoE 模型

对于 MoE 变体（模型类型 `qwen3_5_moe`，如 Qwen3.6-35B-A3B），目标模块列表相同
并覆盖共享专家。路由专家是融合的 3D 参数，没有独立的
`gate_proj`/`up_proj`/`down_proj` 模块，因此 `train_mlp` 会额外通过 peft
`target_parameters` 定位 `mlp.experts.gate_up_proj` 和
`mlp.experts.down_proj`。这与 Tinker 文档中 `train_mlp` 覆盖 MoE 层的行为一致：
注意力、Gated DeltaNet 投影、共享专家和所有路由专家都会训练。vLLM 推理时会解析
peft 格式的专家适配器键，训练和推理涉及的目标一致。

模块列表和参数列表共同定义检查点的结构；两者都会记录在训练运行、检查点元数据
和 peft `adapter_config.json` 中，加载检查点时两者都必须完全一致。在 FSDP 上，
`fsdp_target_parameters` 可以显式覆盖参数侧，与 `fsdp_target_modules` 覆盖模块
侧的方式相同。

## 每个目标都必须匹配真实模块

TuFT 要求解析出的每个目标模块名都必须在加载的模型中匹配到至少一个真实模块。
HF 后端在创建适配器时检查；FSDP 后端在 worker 启动时检查槽位列表。如果模型的
`config.json` 与真实架构不符，请求会被拒绝并列出未匹配的模块名，而不是只训练
其中存在的部分模块。

## Qwen 模型上的 train_unembed

TuFT 目前会接受 Qwen 系列模型上的 `train_unembed=True`，但不会添加
embedding 或 unembedding 目标模块。与 Tinker 官方服务保持一致的
`embed_tokens` 训练和服务支持跟踪于
[issue #153](https://github.com/agentscope-ai/TuFT/issues/153)，并且依赖于包含 Qwen3.5
embedding 模块支持的 vLLM 正式发布版。
