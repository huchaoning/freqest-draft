# HG-SPADE

## HG-SPADE 的一些基本结论

测量得到的结果: 
$$
p(q) = \frac{e^{-\eta}\eta^q}{q!}\qq{where}\eta:=\frac{s^2}{4\sigma^2}
$$
对 $s$ 的导数: 
$$
\dv s p(q) 
= \frac1{\sigma}\frac{e^{-\eta}}{q!} \eta^{q-\frac12}(q - \eta)
$$
令 $\gamma = \sum_q \gamma_q$, 得到
$$
\sigma^2\gamma_q = \frac{e^{-2\eta}}{q!(e^{-\eta}\eta^q+q!b')} \eta^{2q-1}(q - \eta)^2\qq{where} b':=\frac{b}{\nu}
$$

## HG-SPADE 的 $\gamma$

当 $s=0$ 的时候, HG-SAPDE 处于最差抗噪性状态, 此时即便是再微小的噪声都将完全破坏 HG-SPADE. 在数值上体现出 $\gamma \approx 0$. 这是由于当 $s=0$ 时, 所有的关于 $s$ 的 CFI 均包含在 $\phi_1$ 模式之上, 而此时 $\phi_1$ 模式上的光子数期望为 0, 因此抗噪性也为 0. 仅当 严格等于 0 时 $\gamma = 1$. 

该结论可以由 (3) 式看出, 代入 $\eta = 0$ 后, 当 $b\neq0$ 时, (3) 式中的第一项分式是有限大小的. 此时将导致对于所有的 $q$ 均有 $\sigma^2\gamma_q = 0$. 当 $b=0$ 时, $\sigma^2\gamma_1 = 1$.

正是由于这个特性, 导致在数值上计算出现一些问题. 这是由于在数值上令 $b,\eta=0$ 会导致除零错误. 对于 DI 和 PM-SPADE, 之前采用 $b=10^{-10}$ 平滑, 保证数值稳定性, 因为如此小的噪声不会对 PM-SPADE 和 DI 造成影响. 而 HG-SPADE 的问题就在于即便是采用 $b=10^{-10}$, 也会完全破坏 HG-SPADE.

因此, 或许可以考虑定义一个衡量抗噪性能的量 ($\gamma$ 是具体问题得到的结果不是原因). 借助这个量, 我们可以证明当 $s=0$ 时 $\phi_1$ 为最差抗噪性, 即便是最小的噪声也会破坏测量. 首先, 这个量应当和 Fisher 信息以及该信号光子的概率有关. 

## HG-SPADE 的 CFI

当 $s$ 比较小但不为 0 时, HG-SPADE 才能体现出一定的抗噪性. 



