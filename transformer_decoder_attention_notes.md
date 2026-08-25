# Transformer 解码器注意力机制深度解析笔记

> **主题**：Transformer 解码器为什么需要两套注意力机制、掩蔽自注意力（Masked Self-Attention）与交叉注意力（Cross-Attention）的底层工作流程与数学原理。
> **适用环境**：Google NotebookLM 来源笔记 / 深度学习复习文档。

---

## 目录
1. [解码器架构总览](#1-解码器架构总览)
2. [为什么解码器需要两套注意力机制？](#2-为什么解码器需要两套注意力机制)
3. [第一套机制：掩蔽多头自注意力（Masked Self-Attention）](#3-第一套机制掩蔽多头自注意力masked-self-attention)
   - [3.1 核心任务与 Q/K/V 来源](#31-核心任务与-qkv-来源)
   - [3.2 为什么必须加 Mask？（训练并行性与自回归漏洞）](#32-为什么必须加-mask训练并行性与自回归漏洞)
   - [3.3 Mask 矩阵与计算流程](#33-mask-矩阵与计算流程)
4. [第二套机制：编码器－解码器交叉注意力（Cross-Attention）](#4-第二套机制编码器解码器交叉注意力cross-attention)
   - [4.1 核心任务与 Q/K/V 来源](#41-核心任务与-qkv-来源)
   - [4.2 跨序列检索与计算流程](#42-跨序列检索与计算流程)
   - [4.3 为什么不需要因果 Mask？](#43-为什么不需要因果-mask)
5. [两套注意力机制对比汇总](#5-两套注意力机制对比汇总)
6. [PyTorch 代码实现结构对应](#6-pytorch-代码实现结构对应)

---

## 1. 解码器架构总览

在 Transformer 的标准 Encoder-Decoder 架构中，每个解码器层（`DecoderBlock`）由三个核心子层级联而成：
1. **掩蔽多头自注意力层（Masked Multi-Head Self-Attention）** + 残差连接与层规范化（`AddNorm`）
2. **编码器－解码器交叉注意力层（Encoder-Decoder Cross-Attention）** + 残差连接与层规范化（`AddNorm`）
3. **基于位置的前馈网络（PositionWise FFN）** + 残差连接与层规范化（`AddNorm`）

```text
                    ┌─────────────────────────┐
                    │    解码器的一层 (Block)   │
                    └────────────┬────────────┘
                                 │
     [解码器输入 / 上层输出] ──► (1) 掩蔽自注意力层 (Masked Self-Attention)
                                 │ (自回归建模，仅看已生成历史)
                                 ▼
                         Add & LayerNorm
                                 │
  [编码器最终输出 enc_outputs] ─► (2) 交叉注意力层 (Cross-Attention)
                                 │ (提取输入源端的全局上下文语义)
                                 ▼
                         Add & LayerNorm
                                 │
                             (3) 前馈全连接网络 (FFN)
                                 │
                                 ▼
                         Add & LayerNorm ──► [输出给下一层 / 预测头]
```

---

## 2. 为什么解码器需要两套注意力机制？

在序列到序列（Seq2Seq）任务（如机器翻译、文本摘要）中，解码器承担着**双重生成任务**：

### 任务一：建模目标序列内部的语言连贯性（解决“自身说了什么”）
* **问题**：生成当前词必须参考**前面已经生成的词**。例如翻译输出“我 爱 深度 学习”，在预测“深度”时，必须知道前文已经输出了“我 爱”，确保语法合乎逻辑。
* **解决方案**：由**掩蔽多头自注意力机制**完成。

### 任务二：提取源端上下文并建立语义对齐（解决“输入讲了什么”）
* **问题**：如果只知道自己前面说了什么，解码器就会变成“脱缰野马自言自语”，根本不知道源语言讲了什么。解码器必须将当前生成的语义与输入源（如英文原句 "I love deep learning"）进行关联匹配。
* **解决方案**：由**编码器－解码器交叉注意力机制**完成。

### 为什么缺一不可？
* **只有自注意力**：模型完全丢失了输入的源语言信息，无法完成翻译/问答。
* **只有交叉注意力**：模型每次生成都只看输入源，忽略了目标语言自身已经生成的历史，会导致语句断续、词语严重重复或语法混乱。

---

## 3. 第一套机制：掩蔽多头自注意力（Masked Self-Attention）

### 3.1 核心任务与 Q/K/V 来源
* **$Q, K, V$ 来源**：**全部来自解码器自身的上一层输出**（第 1 层解码器则来自目标端词嵌入与位置编码）。
  $$Q = K = V = X_{\text{dec}}$$
* **主要职责**：让目标序列中的每个词元仅与自身及它之前的词元计算注意力。

---

### 3.2 为什么必须加 Mask？（训练并行性与自回归漏洞）

#### 1. 真实推理场景（自回归模式）
推理时是逐词生成的：
* 步 1：输入 `[<bos>]` $\to$ 生成 `[我]`
* 步 2：输入 `[<bos>, 我]` $\to$ 生成 `[爱]`
* 步 3：输入 `[<bos>, 我, 爱]` $\to$ 生成 `[深度]`
在预测第 2 步时，未来的词（`深度`、`学习`）在客观世界上**根本还未生成**。

#### 2. 训练加速需求（Teacher Forcing 并行模式）
如果训练时也一步一步循环计算，GPU 的并行计算性能将无法发挥。因此训练时采用 **Teacher Forcing**，把**整个目标序列一次性打包成矩阵**送入解码器。

#### 3. 致命漏洞：自注意力的全向关注（信息泄露）
标准自注意力机制默认是没有方向限制的：
* 矩阵乘法 $Q K^T$ 会让序列里的**每一个位置都和所有其他位置（包括未来位置）计算关联**。
* 当输入 `[<bos>, 我, 爱, 深度]` 时，位置 1（`<bos>`）需要预测 `我`，但由于没有限制，位置 1 会直接从第 2 行读取 `我` 的特征。
* **作弊后果**：模型在训练时学会了直接“抄袭下一个位置的输入”，损失（Loss）虽然迅速降为 0，但在真实推理时（后面没有答案可抄），模型会彻底瘫痪。

---

### 3.3 Mask 矩阵与计算流程

为了在**单次矩阵运算**中既保持并行加速，又杜绝信息泄露，引入了因果掩码矩阵 $M$：

#### 1. 掩码矩阵元素定义
对于长度为 $N$ 的序列，掩码矩阵 $M \in \mathbb{R}^{N \times N}$ 定义为：
$$M_{i,j} = \begin{cases} 0, & j \le i \ (\text{历史与当前位置}) \\ -\infty, & j > i \ (\text{未来位置}) \end{cases}$$

#### 2. 掩码运算过程
将原始打分矩阵 $S$ 加上掩码矩阵 $M$：
```text
原始打分加掩码 S_masked = S + M：
[  s_11,   -inf,   -inf,   -inf  ]  <-- 位置 1 (<bos>) 只能看自身
[  s_21,   s_22,   -inf,   -inf  ]  <-- 位置 2 (我) 只能看位置 1, 2
[  s_31,   s_32,   s_33,   -inf  ]  <-- 位置 3 (爱) 只能看位置 1, 2, 3
[  s_41,   s_42,   s_43,   s_44  ]  <-- 位置 4 (深度) 可看全部历史
```

#### 3. Softmax 归一化
由于 $e^{-\infty} = 0$，对每一行做 $\text{softmax}$ 后，所有未来位置的权重被**严格归零**：
```text
最终注意力权重 A = softmax(S_masked)：
[  1.0,     0,     0,     0  ]
[  a_21,  a_22,    0,     0  ]
[  a_31,  a_32,  a_33,    0  ]
[  a_41,  a_42,  a_43,  a_44 ]
```

#### 4. 输出加权汇聚
$$O_1 = A V$$
输出后经过残差连接与层归一化（`AddNorm`）：
$$Y = \text{LayerNorm}(X_{\text{dec}} + \text{Dropout}(O_1))$$

---

## 4. 第二套机制：编码器－解码器交叉注意力（Cross-Attention）

### 4.1 核心任务与 Q/K/V 来源
交叉注意力是连接 Encoder 和 Decoder 的桥梁：
* **查询 $Q$（Query）**：来自**解码器内部第一套注意力的输出 $Y$**（代表解码器当前已解析的目标端语义状态）。
  $$Q = Y W_q$$
* **键 $K$（Key）与值 $V$（Value）**：来自**编码器最后一层的完整输出 `enc_outputs`**（代表源输入文本的全局语义表示）。
  $$K = \text{enc\_outputs} W_k, \quad V = \text{enc\_outputs} W_v$$

---

### 4.2 跨序列检索与计算流程

1. **计算跨序列关联度**：
   $$S_{\text{cross}} = \frac{Q K^T}{\sqrt{d_k}}$$
   * 维度变化：$(B, T_{\text{dec}}, d_k) \times (B, d_k, T_{\text{enc}}) \to (B, T_{\text{dec}}, T_{\text{enc}})$
   * 含义：目标序列的每一个词元充当“探针”，在源序列的所有词元中寻找最相关的上下文。

2. **填充掩码（Padding Mask）**：
   使用 `enc_valid_lens` 屏蔽源序列中的 `<pad>` 占位符，避免模型关注无意义的填充内容。

3. **Softmax 归一化与加权求和**：
   $$A_{\text{cross}} = \text{softmax}(S_{\text{cross}})$$
   $$O_2 = A_{\text{cross}} V$$

4. **残差连接与层归一化（AddNorm）**：
   $$Z = \text{LayerNorm}(Y + \text{Dropout}(O_2))$$

---

### 4.3 为什么交叉注意力不需要因果 Mask？
* 编码器处理的是**源输入序列**（如用户说的一整句英语），源句子在解码开始前就已经完整输入并编码完毕。
* 解码器在生成任何一个词时，理应拥有**观察源输入全文所有词的完整视野**，因此不需要遮蔽未来的编码器词元。

---

## 5. 两套注意力机制对比汇总

| 比较维度 | 1. 掩蔽自注意力（Masked Self-Attention） | 2. 编码器－解码器交叉注意力（Cross-Attention） |
| :--- | :--- | :--- |
| **所属层级** | 解码器 Block 的第 1 个子层 | 解码器 Block 的第 2 个子层 |
| **Query ($Q$) 来源** | 解码器自身上一层的输出 | 解码器第一子层的输出 $Y$ |
| **Key ($K$) & Value ($V$) 来源** | 解码器自身上一层的输出 ($Q=K=V$) | **编码器顶层的输出 `enc_outputs`** |
| **序列关系** | 单序列内部（目标端 $\leftrightarrow$ 目标端） | 跨序列对齐（目标端 $\to$ 源端） |
| **掩码类型** | **因果掩码（Causal Mask）** + 目标端填充掩码 | **仅源端填充掩码（Padding Mask）** |
| **核心目的** | 保证自回归语言连贯性，防止未来信息泄露 | 提取输入源语义，实现内容翻译与对齐 |

---

## 6. PyTorch 代码实现结构对应

在《动手学深度学习》（D2L）的 `DecoderBlock` 中，两套注意力的串联调用逻辑如下：

```python
class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        # 第一套：掩蔽多头自注意力
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        
        # 第二套：编码器－解码器交叉注意力
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        
        # 前馈网络
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        
        # 1. 掩蔽自注意力：Q, K, V 全部来自解码器自身输入 X
        # dec_valid_lens 提供下三角掩码，屏蔽未来词元
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        
        # 2. 编码器－解码器交叉注意力：
        # Q 来自解码器状态 Y，K 和 V 来自编码器输出 enc_outputs
        # enc_valid_lens 提供源端填充掩码
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        
        # 3. 前馈全连接网络
        return self.addnorm3(Z, self.ffn(Z)), state
```
