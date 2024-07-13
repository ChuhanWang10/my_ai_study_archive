# Transformer
paper: https://arxiv.org/abs/1706.03762 

## Basic concepts 

### Attention
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

## Q&A
1. **为什么交替使用attention和FFN?**
attention处理的是整个sequence的全局信息，即context information。 对于sequence种任意一个vector, 它都会和任意一个其它的vector计算相关度，计算得到的attention output因此考虑到了全局信息（sequence中vectors分别两两计算相关度，因此attention的计算复杂度是$O(n^2)$）。将每个vector对应的attention output （输入的sequence中有多少个vector，就有多少个attention output）丢进FFN，让FFN处理每个特定位置（考虑了全局信息）的输出。这样扩大了FFN对输入的感受范围（？），因为它相当于将FFN的window扩大到全部的输入，但减轻了计算负担。

2. **交替attention和FFN只能使用一次吗?**
不是。可以交替使用多次。

3. **如何计算attention score?**
如下图所示，如何根据$a^1$,$a^2$,$a^3$,$a^4$计算得到$b^1$? 

![alt text](/home/chuhan/projects/my_ai_study_archive/Deep_Learning/Transformer/images/compute_attention_score.png)
最常见的计算方式是做dot product。例如，计算$a^1$和$a^2$之间的相关度时，将$a^1$与query weight matrix $W^q$相乘得到 $q^1 = W^q a^1$，将$a^2$与key weight matrix $W^k$相乘得到 $k^1 = W^k a^1$, 再将$q^1$和$k^1$做**dot product**得到attention score $a_{1,2}$。 attention score即代表了$a^1$和$a^2$之间的相关度。对$a^1$和sequence中的全部vector（包括自己）分别做上述计算得到attention score，并对所有的attention score做normalization，即做softmax计算 （理论上所有的activation function都可以）。具体过程如下图所示：![alt text](/home/chuhan/projects/my_ai_study_archive/Deep_Learning/Transformer/images/compute_attention_score2.png)

4. **Transformer中attention计算公式 $\text{Attention}(Q, K, V) = \text{Softmax} \big( \frac{QK^T}{\sqrt{d_k}} \big)V$ 中, $\sqrt{d_k}$的作用是什么？**
$\sqrt{d_k}$起到调节作用，使得内积不至于太大（太大的话softmax后就非0即1了，不够“soft”了）

5. **attention layer中的训练参数是什么？**
$W^Q$, $W^k$, $W^V$

6. **什么是self-attention?**
即 $Q=K=V$ (=输入的sequence $X$) 

7. **Multi-head self-attention 需要多少个head?**
这是一个hyperparameter, 需要在实验中调试。

8. **Transformer 为何使用 Multi-Head Attention 机制？**
两个vector之间的关联度可能存在多个dimension，使用multi-head可以使model学习不同representation subspaces的information。Multi-head即相同attention计算重复多次，每个head有自己的$W^Q$, $W^k$, $W^V$，不同head之间参数不共用。