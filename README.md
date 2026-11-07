# 毛选风格 LoRA — 小模型微调项目

用 1B 模型（MiniCPM5-1B）学习《毛泽东选集》的文风。**这是本项目第一次成功的实验。**

---

## 快速开始

**双击 `START.bat`**，等 15 秒，浏览器打开 **http://127.0.0.1:8081**

System Prompt 留空即可 —— `proxy_system.py` 会自动补。

提问建议用训练格式，效果明显更好：

```
以《毛泽东选集》的文风和论证方式，围绕「为人民服务」写一段论述，约300字。可涉及：人民、利益、牺牲、责任、同志
```

---

## 结果

| | |
|---|---|
| 底座 | MiniCPM5-1B（1.09B） |
| 语料 | 76.5 万字《毛泽东选集》全四卷 |
| 训练集 | 1,645 条 / 67.3 万字 |
| 训练 | 822 步，105 分钟，CPU 8 核 |
| **风格相似度** | **base 9.9 → 微调后 77.1**（人写 79.8，达到 **97%**） |
| 量化 | Q4_K_M 656MB，压缩 68.2%，clf 只掉 5.8 |
| 推理速度 | CPU 4 线程 **17~21 t/s** |

完整实验报告见 [总结.md](总结.md)。

---

## 目录

```
├── START.bat                  一键启动（模型 + 代理）
├── 总结.md                    完整实验报告
├── chat_template.jinja        改过的模板：永久关闭 thinking
├── models/
│   ├── mao-Q4_K_M.gguf        656 MB  ← 日常用这个
│   └── mao-f16.gguf           2066 MB ← 无损参照
├── adapter/                   LoRA 权重 31.5MB，可挂到别处
├── tools/                     llama.cpp（自包含）
├── src/
│   ├── data/
│   │   ├── fetch_maoxuan.py   抓语料（求是网官方源）
│   │   ├── build_mao_data.py  建训练集
│   │   └── build_exam20.py
│   ├── train/
│   │   ├── train_mao.py       LoRA 训练
│   │   ├── merge_lora.py      合并 adapter 到基座
│   │   └── rebuild.py         一键：下基座→合并→转GGUF→量化
│   ├── eval/
│   │   ├── train_mao_clf.py   训练风格判别器
│   │   ├── score_with_clf.py  单 checkpoint 评分
│   │   ├── batch_score.py     批量比对多个 checkpoint
│   │   ├── eval_local_show.py 本地生成 + 评测
│   │   └── eval_gguf_quant.py 量化精度验证
│   └── deploy/
│       ├── proxy_system.py    自动补 system prompt 的代理
│       └── quantize_gguf.py   纯 Python 量化
├── data_mao/                  语料与训练集
└── results/                   mao_clf.pkl（判别器）、local_eval.json
```

---

## 三个必须知道的坑

### 1. 不关 thinking 会满嘴英文

模型自带模板里：

```jinja
{%- if enable_thinking is defined %}
    {%- if enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```

**没有 else 分支。** 网页界面不传这个参数，模板就什么都不插入，模型开始思考，1B 还会用英文数词：

> Okay, let me try to think about this. The user is asking for a 200-300 word response...

**已修**：`chat_template.jinja` 无条件插入关闭标记，原文件备份为 `.bak`。

### 2. LoRA 不是万能开关

训练时每条样本都带 system prompt：

```
你是毛泽东，用《毛泽东选集》的语言风格和论证方式写作。
```

界面不发这个，LoRA 就失去触发条件，退回基座人格 —— 问「写为人民服务」会答「我不会说这个的话」。`proxy_system.py` 负责自动补。

### 3. 训练数据的 prompt 里不能有答案

抽原文首句当提纲，等于把答案抄进 prompt，模型学会的是**复制**，loss 降得漂亮但什么都没学到。

最终采用「主题 + 关键词 → 论述」：内容范围给定，模型只负责「怎么说」。

---

## 能力边界

**能做**：看毛选腔、学句式、300 字以内的段落（8 秒出）

**不能做**：写正式文章、当资料引用 —— **会编造史实**

例：要求 800 字时生成「我们有三个人……我们称他们三位人为『工农军』」。历史上没有这回事。

**根因**：1B 的容量只够装「怎么说」，装不下「说什么」。句法学会了，知识没学会。

要解决必须换 3B 以上底座。这套流程（语料、训练脚本、判别器）全部可复用。

---

## 主要结论

> **1B + LoRA 改造不了能力，但能改造表达方式。**

前四轮数学实验全部失败、这一轮成功，验证的就是这条判据：

- 数学推理要求模型**长出**它没有的能力 → 不可能
- 文体模仿只要求把它**已经会的**往某个方向推 → 可行
