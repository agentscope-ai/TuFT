---
sd_hide_title: true
---

# TuFT

<div class="hero-logo" align="center">
  <img class="only-light" src="_static/logo_light.svg" alt="TuFT Logo" width="280"/>
  <img class="only-dark" src="_static/logo_dark.svg" alt="TuFT Logo" width="280"/>
</div>

<p class="hero-subtitle"><strong>TuFT (Tenant-unified Fine-Tuning)</strong> 是一个多租户平台，支持多用户通过统一 API 在共享基础设施上微调大语言模型。可通过 Tinker SDK 或兼容客户端接入使用。</p>

```{image} https://img.alicdn.com/imgextra/i3/O1CN01M7FlDa1LkOf90UsHk_!!6000000001337-2-tps-4000-2250.png
:alt: TuFT 概览
:class: hero-image
```

<div class="install-command">

<p class="install-title">快速安装</p>

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/agentscope-ai/tuft/main/scripts/install.sh)"
```

<p class="install-more-link">更多安装方式（PyPI、源码、Docker），请参阅<a href="getting-started/installation.html">安装指南</a>。</p>

</div>

<div class="quickstart-cta">
  <a class="quickstart-cta-link" href="getting-started/quickstart.html">快速开始 →</a>
</div>

```{admonition} 🚀 没有 GPU？没问题！
:class: tip

运行 TuFT 无需自备 GPU。将其部署到按量付费的云服务商——
**[Modal](deployment/modal.md)**（无服务器，缩容至零）或
**[Lambda Cloud](deployment/lambda.md)**，然后在本地（笔记本）驱动微调。
参阅 **[部署指南](deployment/index.md)**。
```

```{toctree}
:maxdepth: 2
:hidden:

getting-started/index
deployment/index
user-guide/index
development/index
roadmap
```
