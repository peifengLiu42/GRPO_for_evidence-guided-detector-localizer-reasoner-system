# GRPO 奖励函数设计

本文档形式化描述 ms-swift GRPO 阶段使用的奖励函数。对应实现位于
`src/realtext_grpo/msswift_reward_plugin.py` 和
`src/realtext_grpo/rewards.py`。

## 任务定义

对于每张图像 \(x\)，prompt 中包含检测器和定位器预测得到的参考证据。
这些证据可能存在错误，模型需要输出一份取证报告 \(y\)。从模型输出中解析：

\[
\hat{c}(y) \in \{\mathrm{FORGED}, \mathrm{AUTHENTIC}\},
\qquad
\hat{B}(y)=\{\hat{b}_j\}_{j=1}^{m}.
\]

其中 \(\hat{c}(y)\) 是模型预测的图像级类别，\(\hat{B}(y)\) 是模型输出的
归一化定位框集合。奖励函数隐藏使用真实标签和真实框：

\[
c^\star \in \{\mathrm{FORGED}, \mathrm{AUTHENTIC}\},
\qquad
B^\star=\{b_i^\star\}_{i=1}^{n}.
\]

所有框坐标均为 \([0,999]\) 范围内的归一化整数坐标。真实标签、真实框和真实
mask 不会出现在 prompt 中，只用于训练目标、采样元信息或奖励计算。

最终标量奖励定义为：

\[
R(y)=\lambda_{\mathrm{fmt}} R_{\mathrm{fmt}}(y)
     +\lambda_{\mathrm{loc}} R_{\mathrm{loc}}(y;B^\star,c^\star),
\]

默认权重为：

\[
\lambda_{\mathrm{fmt}}=0.05,\qquad
\lambda_{\mathrm{loc}}=0.95.
\]

因此，奖励函数主要优化定位质量，格式奖励只作为轻量约束，保证输出可解析。

## 格式奖励

格式奖励用于鼓励模型输出结构化、可解析的取证报告。我们检查四类必要字段：

\[
R_{\mathrm{fmt}}(y)=
\frac{1}{4}\Big(
\mathbb{1}_{\mathrm{Conclusion}}
+\mathbb{1}_{\mathrm{RiskScore}}
+\mathbb{1}_{\mathrm{Summary}}
+\mathbb{1}_{\mathrm{GroundingOrNoAnomaly}}
\Big).
\]

四个指示函数分别表示：

- 是否能解析出 `[Conclusion]`；
- 是否能解析出 `[RISK_SCORE]`；
- 是否包含 `SUMMARY`；
- 是否包含 `GROUNDING` 或无异常相关描述。

每项贡献 \(0.25\)，因此：

\[
R_{\mathrm{fmt}}(y)\in[0,1].
\]

## 框级匹配

定位奖励首先计算真实框和预测框之间的两两 IoU：

\[
\operatorname{IoU}(b_i^\star,\hat b_j)
=
\frac{|b_i^\star\cap \hat b_j|}
{|b_i^\star\cup \hat b_j|}.
\]

随后使用 Hungarian 算法进行最大 IoU 的一对一匹配：

\[
\mathcal{M}
=
\arg\max_{\text{one-to-one matching}}
\sum_{(i,j)\in \mathcal{M}}
\operatorname{IoU}(b_i^\star,\hat b_j).
\]

匹配后的 IoU 集合记为：

\[
\mathcal{I}=
\{\operatorname{IoU}(b_i^\star,\hat b_j):(i,j)\in\mathcal{M}\}.
\]

### Box F1

一个匹配对只有在 IoU 达到阈值时才被视为真正例：

\[
\tau_{\mathrm{tp}}=0.3.
\]

其中 \(n=|B^\star|\) 表示真实框数量，\(m=|\hat{B}(y)|\) 表示模型预测框数量。

因此：

\[
TP=\sum_{u\in\mathcal{I}}\mathbb{1}[u\ge \tau_{\mathrm{tp}}],
\]

\[
FP=m-TP,\qquad FN=n-TP,
\]

\[
F1_{\mathrm{box}}
=
\frac{2TP}{2TP+FP+FN}.
\]

特殊情况定义为：

\[
n=0,m=0 \Rightarrow F1_{\mathrm{box}}=1,
\]

\[
(n=0,m>0)\ \mathrm{or}\ (n>0,m=0)
\Rightarrow F1_{\mathrm{box}}=0.
\]

也就是说，真实图像没有伪造区域且模型也没有输出框时，框级预测完全正确；若
只有一侧存在框，则视为完全错误。

### Set IoU

为了同时惩罚漏检框和误检框，我们定义集合级 IoU：

\[
\operatorname{SetIoU}
=
\frac{\sum_{u\in\mathcal{I}}u}{\max(n,m)}.
\]

该指标将未匹配的真实框和预测框都视为零贡献。若两侧均为空：

\[
n=0,m=0 \Rightarrow \operatorname{SetIoU}=1.
\]

相比只看最大匹配 IoU，\(\operatorname{SetIoU}\) 对多框场景更稳定，因为它同时
考虑框的质量和数量。

## 像素并集指标

为了让奖励更贴近最终像素级评估指标，我们将真实框和预测框分别转为框并集区域：

\[
G=\bigcup_{i=1}^{n} b_i^\star,
\qquad
P=\bigcup_{j=1}^{m} \hat b_j.
\]

基于并集区域定义像素级 precision、recall 和 IoU：

\[
\operatorname{Prec}_{\mathrm{pix}}
=
\frac{|G\cap P|}{|P|},
\]

\[
\operatorname{Rec}_{\mathrm{pix}}
=
\frac{|G\cap P|}{|G|},
\]

\[
\operatorname{IoU}_{\mathrm{pix}}
=
\frac{|G\cap P|}{|G\cup P|}.
\]

若 \(G\) 和 \(P\) 同时为空，则三项指标均设为 \(1\)。若只有一侧为空，则对应
匹配质量为 \(0\)。

引入像素并集指标的目的是避免模型只学会输出“数量正确”的框，而忽略框的覆盖
范围和最终 mask 质量。

## 高 IoU 奖励

为了让较高 IoU 获得更多额外奖励，我们对 \(\operatorname{SetIoU}\) 加入线性
bonus：

\[
B_{\mathrm{hiIoU}}
=
\max\left(
0,\,
\frac{\operatorname{SetIoU}-\tau_{\mathrm{hi}}}{1-\tau_{\mathrm{hi}}}
\right),
\]

其中：

\[
\tau_{\mathrm{hi}}=0.5.
\]

当 \(\operatorname{SetIoU}<0.5\) 时没有 bonus；超过 \(0.5\) 后，IoU 越高，
bonus 越大。这样可以在 `box_f1` 使用较宽松 TP 阈值 \(0.3\) 的同时，仍然鼓励
模型进一步提升定位精度。

## 定位质量项

未扣除惩罚前的定位质量定义为加权平均：

\[
Q_{\mathrm{loc}}
=
\frac{
w_c C
+w_f F1_{\mathrm{box}}
+w_s \operatorname{SetIoU}
+w_h B_{\mathrm{hiIoU}}
+w_p \operatorname{Prec}_{\mathrm{pix}}
+w_r \operatorname{Rec}_{\mathrm{pix}}
+w_u \operatorname{IoU}_{\mathrm{pix}}
}{
w_c+w_f+w_s+w_h+w_p+w_r+w_u
},
\]

其中：

\[
C=\mathbb{1}[\hat c(y)=c^\star].
\]

默认组件权重如下：

| 符号 | 组件 | 默认值 |
|---|---|---:|
| \(w_c\) | 图像级分类正确性 | 0.25 |
| \(w_f\) | 一对一匹配 Box F1 | 0.75 |
| \(w_s\) | Set IoU | 1.75 |
| \(w_h\) | 高 IoU bonus | 1.00 |
| \(w_p\) | 像素 precision | 1.25 |
| \(w_r\) | 像素 recall | 0.60 |
| \(w_u\) | 像素 union IoU | 1.75 |

归一化分母为：

\[
Z=w_c+w_f+w_s+w_h+w_p+w_r+w_u=7.35.
\]

可以看到，奖励更重视区域质量相关项：Set IoU、像素 precision、像素 recall
和像素 IoU，而图像级分类只作为较小权重的稳定项。

## 惩罚项

最终定位奖励会扣除三类惩罚：非法框、真实图像误检框、伪造图像过大框。

### 非法框惩罚

若预测框超出 \([0,999]\)，或满足 \(x_2\le x_1\)、\(y_2\le y_1\)，则视为非法框。
设非法框数量为 \(N_{\mathrm{invalid}}\)，惩罚为：

\[
P_{\mathrm{invalid}}
=
\min(1,\ 0.25N_{\mathrm{invalid}}).
\]

### 真实图像误检惩罚

对于真实图像，任何 grounding 框都属于误检：

\[
P_{\mathrm{authFP}}
=
\begin{cases}
0.80, & c^\star=\mathrm{AUTHENTIC}\ \land\ m>0,\\
0, & \text{otherwise}.
\end{cases}
\]

该项用于抑制模型在真实样本上幻觉伪造区域。

### 过框惩罚

对于伪造图像，如果预测区域远大于真实区域，则进行过框惩罚。定义预测面积与
真实面积之比：

\[
\rho=\frac{|P|}{\max(|G|,1)}.
\]

惩罚从预测面积超过真实面积两倍时开始：

\[
\rho_0=2.0.
\]

因此：

\[
P_{\mathrm{over}}
=
\begin{cases}
0.60\cdot
\min\left(1,\frac{\rho-\rho_0}{\rho_0}\right),
& c^\star=\mathrm{FORGED},\ n>0,\ m>0,\ \rho>\rho_0,\\
0, & \text{otherwise}.
\end{cases}
\]

该项用于避免模型通过输出过大的框来换取较高 recall。

总惩罚项为：

\[
P
=
P_{\mathrm{invalid}}+P_{\mathrm{authFP}}+P_{\mathrm{over}}.
\]

## 最终定位奖励

定位奖励为：

\[
R_{\mathrm{loc}}
=
\operatorname{clip}(Q_{\mathrm{loc}}-P,\ -1,\ 1).
\]

因此完整奖励函数为：

\[
\boxed{
R(y)
=
0.05\,R_{\mathrm{fmt}}(y)
+0.95\,
\operatorname{clip}(Q_{\mathrm{loc}}-P,\ -1,\ 1)
}
\]

其中：

\[
Q_{\mathrm{loc}}
=
\frac{
0.25C
+0.75F1_{\mathrm{box}}
+1.75\operatorname{SetIoU}
+1.00B_{\mathrm{hiIoU}}
+1.25\operatorname{Prec}_{\mathrm{pix}}
+0.60\operatorname{Rec}_{\mathrm{pix}}
+1.75\operatorname{IoU}_{\mathrm{pix}}
}{7.35}.
\]

## 设计动机

该奖励函数的核心目标是提升文档伪造定位质量，而不是优化解释文本相似度。最终
ms-swift GRPO 设置中没有使用 explanation reward，原因是：

- 解释相似度需要额外 embedding 模型，分布式训练成本更高；
- 当前任务的主要瓶颈是定位框和像素级 mask 质量；
- 我们最终关注的验证指标是图像级 balanced accuracy / weighted F1，以及像素级
  precision、recall、F1 和 IoU。

因此，奖励函数采用“轻格式约束 + 强定位约束”的结构：格式奖励保证输出可解析，
定位奖励直接优化分类正确性、框匹配质量、像素覆盖质量，并通过误检和过框惩罚
约束模型不要在 authentic 样本上幻觉框，也不要在 forged 样本上输出过大的区域。
