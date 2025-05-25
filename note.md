# HG-SPADE

## 测量得到的结果

$$
p(q) = \frac{e^{-\lambda}\lambda^q}{q!}\qq{where}\lambda:=\frac{s^2}{4\sigma^2}
$$

 显然服从泊松分布, 所以在噪声比较小的情况下, 样本均值即为 $\lambda$ 的最优估计, 所以
$$
\hat s = 2\sigma \sqrt{\sum_q \frac{q n_q}{n}}\qq{where} n = \sum_q n_q
$$
假设 00 和 01 模式近完备, 所以
$$
\hat s = 2\sigma \sqrt\frac{n_1}{n_0 + n_1}
$$
导数:
$$
\dv s p(q) 
= \frac{e^{-\lambda}}{q!} \lambda^{q-1} (q - \lambda) \frac{s}{2\sigma^2}
$$

$$
q=0
$$

$$
p(0) = e^{-\lambda}
$$

$$
\dv{s}p(0)=-\frac s{2\sigma^2}e^{-\lambda}
$$

$$
\sigma^2\eval{\gamma^\text{(HG)}}_{q=0} =\frac{1}{e^{-\lambda}+b} \lambda e^{-2\lambda}
$$


$$
q=1
$$

$$
p(1)=\lambda e^{-\lambda}
$$

$$
\dv{s}p(1)=(1-\lambda) \frac{s}{2\sigma^2}e^{-\lambda}
$$

$$
\sigma^2\eval{\gamma^\text{(HG)}}_{q=1} = \frac{(1-\lambda)^2}{\lambda e^{-\lambda}+b}\lambda e^{-2\lambda}
$$







如果定义 $\dv{\gamma}{b}$ 为抗噪性指标.
$$
\gamma = \frac{1}{p+b}\qty(\dv{p}{s})^2\\

\dv{\gamma}{b} = -\qty(\frac{1}{p+b}\dv{p}{s})^2
$$

$$
\qty(\frac{1}{p+b}\dv{p}{s})^2\leq\qty(\dv{p}{s})^2\\
\dv{\gamma}{b}\geq\qty(\dv{p}{s})^2
$$

$$
\qty(\frac{1}{p+b})^2\geq1
$$





$\gamma$ 越大越好, 问题为 $\arg\max_p \gamma(p)$
$$
\gamma = \frac{1}{p+b}\qty(\dv{p}{s})^2
$$




问题在于, 如果不采用 smoothing 数值会出现错误. 但如果采用, 则数值结果会和理论值相差较大.

可能取对数能解决数值问题.