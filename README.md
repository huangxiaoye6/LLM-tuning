# 大模型（LLMs）微调训练 快速入门指南

![ ](https://img.shields.io/badge/python-3.10.0-brightgreen.svg) ![ ](https://img.shields.io/badge/huggingface-0.31.2-brightgreen.svg) ![ ](https://img.shields.io/badge/transfermer-4.51.3-brightgreen.svg)![ ](https://img.shields.io/badge/peft-0.18.1-brightgreen.svg)



## 项目概述

本项目旨在介绍大模型微调相关算法，并以开源模型 Qwen3-0.6B 为例展开，实现了 Lora、P-tuning 等微调算法，同时涵盖模型增量预训练、GPTQ/AWQ 等量化操作，并提供了对应的完整代码实现。

## 拉取代码

大家可以克隆仓库到本地或者服务器上运行

```
git clone https://github.com/huangxiaoye6/LLM-tuning.git
```



## 搭建开发环境

本项目对于硬件有一定要求，GPU显存不少于6G，否则可能没办法进行训练。

建议使用云服务器进行模型训练和微调。

- Python环境管理  [uv](https://docs.astral.sh/uv/)

- Python交互式开发环境  [jupyter notebook](https://jupyter.org/)

- LLM开源模型  [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)



**以下是详细的安装指导（以 Window11操作系统为例）**：

### 安装 CUDA Toolkit 和 GPU 驱动

根据你的实际情况，找到对应的 [CUDA 12.04](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)：

下载并安装 CUDA 12.4 Toolkit（包含GPU驱动）：

<img src="./docs/nvidia.png" alt="nvidia" style="zoom:30%;" align="left"/>



安装完成后，使用 `nvidia-smi` 指令查看版本和显存大小：

<img src="./docs/gpu-check.png" style="zoom:43%;"  align="left"/>

### 安装 uv 环境管理工具

```python
pip install uv
```

### 安装指定Python版本

```
uv python install 3.10
```

### 安装项目依赖

```
uv sync
```

## 微调算法对比

### 1. 全量微调（Fine-tuning）

- **原理**：对预训练模型**所有参数**进行更新。
- **优点**：能够充分利用数据，可能得到更好的模型性能。
- **缺点**：计算资源消耗大，训练时间长，需要大量的显存，容易灾难性遗忘。
- **适用场景**：当有足够的计算资源和时间，且数据量较大时适用。

### 2. LoRa 微调

- **原理**：在不改变预训练模型原有参数的基础上，通过在模型的某些层添加可训练的低秩自适应（LoRA）矩阵，只对这些矩阵进行训练。
- **优点**：减少了可训练参数的数量，降低了计算资源的需求，训练速度相对较快。
- **缺点**：性能提升可能不如全量微调。
- **适用场景**：计算资源有限，希望在较短时间内完成微调时适用。

### 3. QLoRa 微调

- **原理**：结合了量化技术和 LoRA 技术，进一步减少了内存占用。
- **优点**：在保证一定性能的前提下，大幅降低了内存需求，使得在资源受限的情况下也能进行微调。
- **缺点**：可能会因为量化带来一定的精度损失。
- **适用场景**：计算资源极度受限，如在消费级 GPU 上进行微调时适用。

### 4、软提示微调（Prompt-Tuning）

- **原理**：冻结模型参数，仅在输入 embedding 前拼接一段可训练的 **软提示向量**。
- **优点**：训练参数量极小，显存消耗极低。
- **缺点**：表达能力有限，小模型效果差，对复杂生成任务适配能力弱。
- **适用场景**：超大模型（10B+），分类、简单生成任务。

### 5、Prefix微调

- **原理**：冻结模型参数，在每层 attention的key或value里添加可训练的 prefix向量

- **优点**：效果比 Prompt-Tuning 强，参数量小。

- **缺点**：训练复杂，对小模型仍有限制。

- **适用场景**：生成类任务，中等规模模型。

### 6、P-Tuning

- **原理**：冻结模型参数，在每个transformer层的输入位置里加入可学习的Prompt向量。
- **优点**：小模型效果较好，表现接近全量微调。
- **缺点**：对大规模结构变动能力有限。
- **适用场景**：中小模型任务适配，NLP 下游任务。

### 7、IA³-Tuning

- **原理**：冻结模型参数，对 attention 和 FFN 输出做逐元素缩放。
- **优点**：训练参数更少，推理几乎无额外开销，训练稳定。
- **缺点**：表达能力比 LoRA 弱，对复杂任务适配能力有限。
- **适用场景**：轻量部署，资源极其有限。

## 量化方法

### 1、GPTQ量化

- **优点**：对4bit 仍保持较高精度，推理速度快，显存降低 75%。
- **缺点**：量化过程慢，不适合训练，主要优化权重，不优化 activation。
- **适用场景**：纯推理部署，单卡部署 7B/13B等。

### 2、AWQ量化

- **优点**：量化快，更适合大模型，比 GPTQ 更稳定。
- **缺点**：精度略低于 GPTQ。
- **适用场景**：大规模部署。



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=huangxiaoye6/LLM-tuning&type=date&legend=top-left)](https://www.star-history.com/#huangxiaoye6/LLM-tuning&type=date&legend=top-left)
