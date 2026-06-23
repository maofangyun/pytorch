# Seq2Seq 编解码器计算与维度变换流程图

在序列到序列（Seq2Seq）学习中，张量的维度（Shape）变换是理解模型数据流的核心。以下是为您生成的 Seq2Seq 维度流直观计算图及详细解析。

---

## 1. 维度直观流向图

![Seq2Seq 编解码器三维维度计算流向示意图](C:/Users/maofa/.gemini/antigravity/brain/a06417ee-f2b3-4cd7-9050-ca055cae42d5/seq2seq_dimension_flow_visualization_1780908835401.png)

---

## 2. 核心维度变化对比表

| 阶段 / 模块 | 输入形状 | 输出形状 | 核心变换说明 |
| :--- | :--- | :--- | :--- |
| **编码器 Embedding** | `(batch_size, num_steps)` | `(batch_size, num_steps, embed_size)` | 每个词的索引转换为稠密词向量。 |
| **编码器 Permute** | `(batch_size, num_steps, embed_size)` | `(num_steps, batch_size, embed_size)` | 满足 PyTorch RNN 默认的时间步优先（Time-step First）格式。 |
| **编码器 GRU** | `(num_steps, batch_size, embed_size)` | **output:** `(num_steps, batch_size, num_hiddens)`<br/>**state:** `(num_layers, batch_size, num_hiddens)` | 提取所有时间步的顶层输出，以及最后一时间步的所有层隐状态。 |
| **解码器 Context 提取** | 编码器 `state`: `(num_layers, batch_size, num_hiddens)` | `(batch_size, num_hiddens)` | 仅提取最顶层的最终隐状态 `state[-1]`。 |
| **解码器 Context 广播** | `(batch_size, num_hiddens)` | `(num_steps, batch_size, num_hiddens)` | 复制 `num_steps` 次，使其可以与解码器输入在时间步上一一对应。 |
| **解码器输入拼接** | **X:** `(num_steps, batch_size, embed_size)`<br/>**context:** `(num_steps, batch_size, num_hiddens)` | `(num_steps, batch_size, embed_size + num_hiddens)` | 在特征维度拼接，使得解码器每个步均可直接访问编码器的全局语义。 |
| **解码器 GRU** | `(num_steps, batch_size, embed_size + num_hiddens)` | **output:** `(num_steps, batch_size, num_hiddens)`<br/>**state:** `(num_layers, batch_size, num_hiddens)` | 接收拼接输入，并由编码器的 `state` 初始化隐藏状态。 |
| **解码器 Dense 输出** | `(num_steps, batch_size, num_hiddens)` | `(num_steps, batch_size, vocab_size)` | 将隐藏状态映射到目标词表的几率空间。 |
| **解码器 维度还原** | `(num_steps, batch_size, vocab_size)` | `(batch_size, num_steps, vocab_size)` | 还原为标准的 Batch 优先格式，便于后续与 Label 计算损失。 |
