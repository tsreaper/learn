# 从 Transformer 注意力机制到 RoPE

Rotary Position Embeddings（RoPE，旋转位置编码）是 Transformer 中的一种位置编码方法。要理解 RoPE，需要先理解 Transformer 为什么需要注意力机制，以及注意力机制如何工作。

## 1. Transformer 要解决什么问题？

假设语言模型看到：

> 小猫吃了鱼，因为它很饿。

为了理解“它”指什么，模型需要联系前面的“小猫”；为了理解“很饿”，也需要结合“它”和整句话。

因此，每个词在形成自己的表示时，都应该能从其他相关词那里收集信息。Attention（注意力机制）就是负责这件事的。

Transformer 则是一种以 Attention 为核心，再配合前馈神经网络等组件的模型架构。

## 2. 文字怎样进入神经网络？

神经网络不能直接处理文字，所以通常经过两步。

第一步是分词，把文本切成 token。token 不一定恰好是一个汉字或单词。例如：

> 我喜欢机器学习

可能被切成：

> “我” / “喜欢” / “机器” / “学习”

第二步是把每个 token 映射成一个向量：

$$
x_i \in \mathbb{R}^d
$$

这个向量称为 token embedding。可以把它想成模型内部对 token 的数字表示。

于是，一个包含 $n$ 个 token 的句子会变成一个矩阵：

$$
X=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}
\in \mathbb{R}^{n\times d}
$$

其中每一行对应一个 token。

## 3. Attention 的核心问题

当模型更新第 $i$ 个 token 的表示时，它需要回答三个问题：

- 我现在想寻找什么信息？
- 其他 token 能否提供这种信息？
- 如果某个 token 很相关，我应该从它那里取走什么信息？

Attention 用 Query、Key 和 Value 表示这三个角色：

- Query（查询）：当前 token 想找什么。
- Key（键）：当前 token 能以什么特征被其他 token 找到。
- Value（值）：当前 token 真正提供给其他 token 的信息。

可以类比搜索引擎：

- Query 是搜索词；
- Key 是网页标签；
- Value 是网页内容。

不过，Query、Key、Value 都不是人工定义的，而是模型训练出来的向量。

## 4. Query、Key、Value 从哪里来？

对每个 token 向量 $x_i$，分别进行三种线性变换：

$$
q_i=x_iW_Q
$$

$$
k_i=x_iW_K
$$

$$
v_i=x_iW_V
$$

$W_Q$、$W_K$ 和 $W_V$ 是模型通过训练学到的参数。

同一个 token 因此会有三种不同表示：

- $q_i$：我需要什么；
- $k_i$：我具有什么特征；
- $v_i$：我能传递什么内容。

这里最容易混淆的一点是：Query、Key、Value 不是三种不同的输入。它们都是由同一个 token 表示经过不同变换得到的。

## 5. 怎样判断两个 token 是否相关？

假设我们正在更新位置 $i$ 的 token，并想知道位置 $j$ 的 token 是否重要。

我们计算 Query 和 Key 的点积：

$$
s_{ij}=q_i^\top k_j
$$

如果两个向量方向比较接近，点积通常较大，表示：

> 位置 $j$ 所拥有的信息，与位置 $i$ 正在寻找的信息比较匹配。

实际 Attention 会除以 $\sqrt{d_k}$：

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_k}}
$$

这里 $d_k$ 是 Key 和 Query 的维度。除以 $\sqrt{d_k}$ 是为了避免维度较高时点积数值过大，导致后面的 softmax 过于极端。

## 6. 怎样把相关性变成权重？

对于位置 $i$，模型会计算它与所有位置的分数：

$$
s_{i1},s_{i2},\ldots,s_{in}
$$

然后使用 softmax 把它们转换成总和为 $1$ 的非负权重：

$$
a_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{\ell=1}^{n}\exp(s_{i\ell})}
$$

例如，某个 token 对其他 token 的注意力权重可能是：

| token | 权重 |
|---|---:|
| 小猫 | $0.60$ |
| 吃了 | $0.10$ |
| 鱼 | $0.05$ |
| 它 | $0.20$ |
| 很饿 | $0.05$ |

这些只是示意值。真实权重由模型根据输入和训练结果计算。

## 7. 怎样收集信息？

最后，用注意力权重对所有 Value 加权求和：

$$
y_i=\sum_{j=1}^{n}a_{ij}v_j
$$

如果 $a_{ij}$ 很大，位置 $j$ 的 Value 就会对新的表示 $y_i$ 产生较大影响。

所以，Self-Attention 可以概括为：

1. 用 Query 和 Key 判断“应该关注谁”；
2. 用 softmax 得到关注程度；
3. 按照关注程度汇总 Value。

关键区别是：

> Key 用来判断相关性，Value 才是最终被取走的信息。

## 8. 完整公式

把所有 token 一起写成矩阵：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

Attention 的完整公式是：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V
$$

其中：

- $QK^\top$：所有 token 两两之间的匹配分数；
- softmax：把每一行变成注意力权重；
- 乘以 $V$：根据权重汇总信息；
- $M$：可选的注意力掩码。

如果序列中有 $n$ 个 token，那么 $QK^\top$ 的形状是 $n\times n$。矩阵中的第 $i$ 行、第 $j$ 列表示：

> 位置 $i$ 对位置 $j$ 的关注程度。

## 9. 为什么叫 Self-Attention？

因为 Query、Key 和 Value 都来自同一个序列 $X$：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

所以称为 Self-Attention。

如果 Query 来自一个序列，而 Key、Value 来自另一个序列，就称为 Cross-Attention。例如在机器翻译中，英文输出可以关注中文输入。

## 10. GPT 为什么需要 causal mask？

GPT 的任务是根据前文预测下一个 token。预测位置 $i$ 时，不能偷看未来位置。

例如输入：

> 今天天气很……

模型不能在预测“好”时提前看到答案“好”。

因此会使用 causal mask，把所有未来位置的分数设成负无穷：

$$
M_{ij}=
\begin{cases}
0,&j\le i\\
-\infty,&j>i
\end{cases}
$$

softmax 之后，这些未来位置的权重就会变成 $0$。

注意力范围大致如下：

| 当前位置 | 可以关注 |
|---|---|
| 第 1 个 token | 第 1 个 |
| 第 2 个 token | 第 1～2 个 |
| 第 3 个 token | 第 1～3 个 |
| 第 $i$ 个 token | 第 1～$i$ 个 |

Causal mask 表示“能不能看”，位置编码表示“谁在什么位置”。两者不是同一件事。

## 11. 为什么需要多头注意力？

单组 Query、Key、Value 只能在一个表示空间中比较 token。Transformer 通常同时进行多组 Attention，称为 Multi-Head Attention。

不同的头可以学习不同类型的关系，例如：

- 代词和它指代的名词；
- 动词和主语；
- 相邻 token；
- 长距离依赖；
- 标点和句子边界。

每个头都有自己的一组投影参数，分别计算 Attention。各个头的结果会被拼接，再经过一次线性变换。

这里的“每个头学什么”不是人为规定的，只是帮助理解的直觉。真实模型中的注意力头不一定有如此清晰的语言学含义。

## 12. Transformer 不只有 Attention

一个简化的 Transformer 层通常包括：

1. Multi-Head Self-Attention：让 token 之间交换信息；
2. 前馈神经网络：分别处理每个 token 的表示；
3. 残差连接：保留原有信息并帮助深层网络训练；
4. Layer Normalization：控制数值尺度，使训练更稳定。

许多这样的层叠加起来，就形成了 Transformer。

对于 GPT 一类模型，整体过程可以概括成：

$$
\text{token}
\rightarrow
\text{embedding}
\rightarrow
\text{多层 Transformer}
\rightarrow
\text{词表概率}
\rightarrow
\text{下一个 token}
$$

## 13. Attention 本身为什么不知道顺序？

假设只有下面三个 token：

> 狗 咬 人

如果只看 token 向量和普通 Self-Attention，模型并没有一个天然机制知道谁是第一个、谁是第二个、谁是第三个。

但下面两句话含义完全不同：

> 狗咬人
>
> 人咬狗

因此 Transformer 还必须加入位置信息。这就是位置编码存在的原因。

早期 Transformer 常把位置向量直接加到 token embedding 上：

$$
x_i'=x_i+p_i
$$

RoPE 则使用另一种办法：它主要不把位置向量直接加到输入上，而是根据 token 的位置旋转 Query 和 Key。

## 14. 二维旋转

在理解 RoPE 前，先考虑一个二维向量：

$$
x=
\begin{bmatrix}
x_1\\
x_2
\end{bmatrix}
$$

将它逆时针旋转角度 $\phi$ 的旋转矩阵是：

$$
R(\phi)=
\begin{bmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{bmatrix}
$$

旋转后的向量是：

$$
x'=R(\phi)x
$$

旋转只改变向量的方向，不改变向量的长度。

## 15. RoPE 怎样旋转高维向量？

真实模型中的 Query 和 Key 是高维向量。RoPE 会把相邻维度两两分组：

$$
(x_0,x_1),\ (x_2,x_3),\ \ldots
$$

每一组都被看成一个二维向量，并使用不同的旋转频率。常见的频率形式是：

$$
\theta_r=10000^{-2r/d}
$$

对于位于位置 $m$ 的 token，第 $r$ 对维度会旋转角度：

$$
m\theta_r
$$

有些维度旋转得快，能敏感地表示较短距离；有些维度旋转得慢，能表示更长尺度上的位置变化。

## 16. RoPE 为什么能表示相对位置？

考虑一对二维分量。对于位置 $i$ 的 Query 和位置 $j$ 的 Key，RoPE 分别进行旋转：

$$
q_i'=R(i\theta)q_i
$$

$$
k_j'=R(j\theta)k_j
$$

它们的注意力分数变成：

$$
(q_i')^\top k_j'
=
q_i^\top R(i\theta)^\top R(j\theta)k_j
$$

旋转矩阵满足：

$$
R(i\theta)^\top R(j\theta)
=
R((j-i)\theta)
$$

因此：

$$
(q_i')^\top k_j'
=
q_i^\top R((j-i)\theta)k_j
$$

这里出现了 $j-i$，也就是两个 token 的相对位置。

所以 RoPE 可以理解为：

> 先按照各自的绝对位置旋转 Query 和 Key；当它们计算点积时，最终体现为两者之间的相对距离。

旋转方向和公式中 $i-j$ 或 $j-i$ 的符号可能因具体约定而不同，但“点积依赖相对位置”这一结论不变。

通常只有 Query 和 Key 会进行这种旋转，Value 不旋转。

## 17. RoPE 的复数视角

还可以把相邻的两个维度看成一个复数：

$$
z_r=x_{2r}+\mathrm{i}x_{2r+1}
$$

二维旋转就相当于乘以一个单位复数：

$$
z_r'=z_r e^{\mathrm{i}m\theta_r}
$$

这种表示能直观地说明：

- 旋转不会改变向量长度；
- 两个位置之间的相位差只取决于相对距离；
- 不同维度对使用不同的旋转频率。

## 18. RoPE 的特点和限制

RoPE 有以下特点：

- 通常只作用于 Query 和 Key；
- 不需要为每个位置学习单独的位置 embedding；
- 旋转保持向量范数不变；
- 相对位置信息会直接进入注意力分数；
- 同一套旋转公式可以计算训练长度之外的位置。

不过，最后一点不代表模型能够无损地外推到任意长度。超出训练上下文后，模型可能遇到没有见过的旋转相位分布，也可能受到旋转周期性的影响。因此，长上下文模型经常还会采用额外的 RoPE scaling 方法。

另外，RoPE 和 causal mask 解决的是两个不同的问题：

- RoPE 告诉模型 token 之间的位置关系；
- causal mask 阻止模型看到未来 token。

使用 RoPE 后，GPT 仍然需要 causal mask。

## 19. 总结

Attention 的主要信息流是：

$$
\text{文字}
\rightarrow
\text{token 向量}
\rightarrow
Q,K,V
\rightarrow
QK^\top\text{ 计算相关性}
\rightarrow
\text{softmax}
\rightarrow
\text{加权汇总 }V
$$

其中：

- Query 表示当前 token 想寻找什么；
- Key 表示一个 token 可以怎样被匹配；
- Value 表示一个 token 真正传递的内容；
- causal mask 决定当前位置能看哪些 token；
- 位置编码告诉模型 token 的顺序；
- RoPE 通过旋转 Query 和 Key，把相对位置信息放进注意力分数。

最简短的概括是：

$$
\boxed{
\text{RoPE = 用位置控制 Q/K 的旋转，用角度差表示相对位置}
}
$$
