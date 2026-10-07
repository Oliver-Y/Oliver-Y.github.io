---
title: Stats review
date: 2026-09-24
blurb: Review of Ch 2-4 of Probabilistic Machine Learning — random variables, the distribution catalogue, Gaussians, and Bayesian inference.
---

I wish I took the time to understand stats in college. It's 
one of those subjects that's important to learn and internalize. At
the root of it (at least in my opinion) is it provides a way to quantify
uncertainty. It has heavy applications in all these ML papers I read but also,
feels like it'd be a useful tool in confronting a world that feels incredibly
more complex by the second. So here's my, verbose and probably imprecise notes
on learning/re-learning the foundations.

## 0. RV — Random variables

I find myself constantly needing to remind myself, that at the core of it, it's
a random variable X with some set of outcomes (state space) and it's
about assigning a probability to every outcome.

The nature of X can change which gives us more information, e.g. continuous,
discrete, as well as having various relationships w/ other random variables
such as conditionals, joint probabilities, marginals. But at the end of the
day, we're just placing probability densities/mass over RVs.

### PMF, CDF, PDF

- PMF: probability mass function: discrete X describing $P(X = x)$ over state
  space
- CDF: cumulative distribution function: describing $P(X \le x)$ (discrete or
  continuous X)
- PDF: probability density function, derivative of CDF, integrates to
  probability. $p(x)\,dx$ is the probability of landing in a small interval around $x$;
  density is probability per unit length.
  - Note: the terms "mass"/"density" help here. Density can only give mass
    when given (volume) — or in this case a closed interval to integrate over.

### Conditionals, Independence, Joint probabilities

Random variables are often related to each other and that gives more
information.

- Joint probability: $P(A, B)$ — if independent, then $P(A) \cdot P(B)$
- Conditional probability: $P(A \mid B) = \dfrac{P(A, B)}{P(B)}$
- Marginal: $P(X{=}x) = \sum_y P(X{=}x, Y{=}y)$ — think collapsing (summing) the
  plane onto one axis here; slicing it at a fixed $y$ gives the conditional
  instead

### Bayes rule

- $p(\theta \mid D) = \dfrac{p(\theta)\, p(D \mid \theta)}{p(D)}$
- Intuitively, it's easy to miss the prior and only look at likelihood when
  examining data. But strong priors are less influenced by weak likelihoods and
  vice versa.
  - Simply put, if the probability of something happening is low already, just
    cause we see data that supports the claim doesn't fully mean the
    probability is suddenly high.

## 1. Distributions

The nature of random events often give rise to common distributions, which can
be presented in any of the PMF, CDF, or PDF forms above, but ultimately are
parameterized ways to describe common RVs and their probability mass/densities. 
Below is a subset from the book. 

### Bernoulli / Binomial

Bernoulli: params θ, S results in binary events.

$$\mathrm{Ber}(s \mid \theta) \triangleq \theta^s (1-\theta)^{1-s}$$

Binomial: params: θ probability of true, N number of binary events (still
discrete over N).

$$\mathrm{Bin}(s \mid N, \theta) \triangleq \binom{N}{s}\, \theta^s (1-\theta)^{N-s}$$

- N choose s: all the ways we can have s (successes/trues) out of N events
  (how many heads out of N coin tosses)

### Categorical / Multinomial

Categorical: params: (C−1) θ, one θ each class, all θ sum up to 1.

$$\mathrm{Cat}(y \mid \theta) \triangleq \prod_{c=1}^{C} \theta_c^{y_c}$$

- One-hot encoding lets this happen: $(y_0, y_1, y_2, y_3, \dots) = (0, 0, 1,
  0)$, so only one probability is active at one time.
- Quick example with $C = 3$,
  $\theta = (0.2, 0.5, 0.3)$, and class 2 observed, $y = (0, 1, 0)$:

  $$\mathrm{Cat}(y \mid \theta) = 0.2^0 \cdot 0.5^1 \cdot 0.3^0 = 1 \cdot 0.5 \cdot 1 = 0.5$$


Multinomial: params θ (same as categorical), N number of categorical events.
$s_c$ counts how many times class c showed up.

$$\mathrm{Mult}(\mathbf{s} \mid N, \theta) \triangleq \binom{N}{s_1, \dots, s_C} \prod_{c=1}^{C} \theta_c^{s_c}$$

- Quick example with the same $\theta = (0.2, 0.5, 0.3)$, $N = 4$ draws,
  counts $s = (1, 2, 1)$:

  $$\mathrm{Mult}(s \mid 4, \theta) = \frac{4!}{1!\,2!\,1!} \cdot 0.2^1 \cdot 0.5^2 \cdot 0.3^1 = 12 \cdot 0.015 = 0.18$$

- Softmax w/ temperature is often used to deal with the θ-sums-up-to-1
  constraint so we can work with raw logits. Temperature controls how spread
  the probabilities are.
  - Softmax equation: exponentiates + normalizes

  $$\mathrm{softmax}(z)_c = \frac{e^{z_c / T}}{\sum_{c'=1}^{C} e^{z_{c'} / T}}$$
- Logistic regression on multinomial case is just the multi-dimensional
  version of $w x + b$. $W = C \times D$ matrix, $b$ = C dim vector. We
  parameterize by $W$ instead, which indirectly controls the θ's through a
  linear transform followed by softmax. Will see this example down the road.

### Gaussians

Gaussian: params: mean, variance over continuous RV X, in PDF form:

$$\mathcal{N}(y \mid \mu, \sigma^2) \triangleq \frac{1}{\sqrt{2\pi\sigma^2}}\, e^{-\frac{1}{2\sigma^2}(y - \mu)^2}$$

- The square in the exponent, $e^{-(y-\mu)^2}$, gives it the bell curve shape
- Mean = expectation, value weighted by probability:
  $\mathbb{E}[Y] \triangleq \int_Y y\, p(y)\, dy$
- Variance = spread of data: $V[Y] \triangleq \mathbb{E}\big[(Y - \mu)^2\big]$
- Note: $p(y \mid x; \theta) = \mathcal{N}(y \mid w^T x + b, \sigma^2)$ is
  useful in this form as well (linear regression), s.t. we parameterize by
  input X and weights w, b with set variance
- Central limit theorem: the standardized sum of IID samples (finite variance) approaches a gaussian as
  N grows. Making it
  pretty common for modeling residuals/noisy information.
  - Noise usually is a group of independent things that add onto the end result
  - CLT generally doesn't care what the underlying distribution is. As long as
    we sum IID terms with finite variance (Cauchy breaks this), the standardized sum
    approaches Gaussian (neat trick: just pick a space where it's additive e.g log)
- Maximum entropy: of all distributions with a given mean and variance, the
  Gaussian has the highest entropy, so it assumes the least beyond those two
  numbers

Couple other distributions I didn't go into too much detail: Cauchy, Gamma,
Student-T.

## 2. Deep dive on Gaussians, multi-dim + manipulating Gaussians

### Multi-dimensional Gaussians

Takes the univariate above and turns it into vectors and matrices. By
putting one joint Gaussian over the whole vector, we encode information
between dimensions through covariance. This additional information allows us
to extract relationships between dimensions and describe them through joints,
conditionals, and marginals.

{% include "figures/joint-vs-marginal.html" %}

Note: Gaussian along each axis isn't enough for "joint". On the right,
$Y = \pm X$ (coin flip): both axes are bells, but the cloud is an X. Joint
means every $aX + bY$ is Gaussian.

#### Covariance

- Cov of 2 RVs: $\mathrm{Cov}[X, Y] \triangleq \mathbb{E}\big[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])\big]$
  - Verbose: expectation of how much one RV differs from its mean multiplied
    by how much the other RV differs from its mean. Simply put, how do the
    deviations of two variables move together.
    - Why multiply? Let's say X and Y are comprised of $S$, $N_x$, $N_y$,
      where $X = S + N_x$ and $Y = S + N_y$, with $S$, $N_x$, $N_y$
      independent. Assume mean 0 for everything, so $\mathbb{E}[X] =
      \mathbb{E}[Y] = 0$ too and the deviations are just $X$ and $Y$. Then
      fitting this into the equation we get:
      - Lining up $V[X]$ next to $\mathrm{Cov}[X, Y]$:

        $$\begin{aligned}
        V[X] &= \mathbb{E}[(S + N_x)(S + N_x)] \\
        &= \mathbb{E}[S^2] + 2\,\mathbb{E}[S N_x] + \mathbb{E}[N_x^2] = V[S] + V[N_x] \\
        \mathrm{Cov}[X, Y] &= \mathbb{E}[(S + N_x)(S + N_y)] \\
        &= \mathbb{E}[S^2] + \mathbb{E}[S N_y] + \mathbb{E}[N_x S] + \mathbb{E}[N_x N_y] = V[S]
        \end{aligned}$$

      - Only the $\mathbb{E}[S^2]$ term survives in the covariance, because
        the rest are independent and mean 0: $\mathbb{E}[S N_y] = \mathbb{E}[S]\,\mathbb{E}[N_y] = 0$.
        And since $\mathbb{E}[S] = 0$, $\mathbb{E}[S^2] = V[S]$. So
        covariance is the variance of the shared part.
      - Rule: a term survives only when something is multiplied by itself. In
        $V[X]$ both $S$ and $N_x$ meet themselves, so both count; in
        $\mathrm{Cov}[X, Y]$ only $S$ is on both sides. $N_x$ and $N_y$ make
        each variable bigger without adding to the covariance, which is why
        they pull $\rho$ down.
    - Cov carries units (X's units times Y's units), so its size depends on
      the scale of each variable. Correlation normalizes by the standard
      deviation along each axis, so it's unitless and always between -1 and 1:
      how tightly the points hug a line, not the slope of that line.
      - $\rho = \mathrm{corr}[X, Y] \triangleq \dfrac{\mathrm{Cov}[X, Y]}{\sqrt{V[X]\, V[Y]}} = \dfrac{\mathrm{Cov}[X, Y]}{\sigma_X \sigma_Y}$
    - Geometrically, how do the deviations of X and Y line up (how tightly they
      hug the line is $\rho$'s job, above).

<figure class="cov-fig"><svg viewBox="0 0 648 260" width="100%" role="img" aria-label="Three point clouds: correlation 0.9 hugging a tilted line; correlation 0 forming a round cloud; the first cloud stretched 2x along X, where correlation stays 0.9 but covariance doubles and the slope halves."><rect x="0" y="8" width="200" height="200" style="fill:var(--panel);stroke:var(--line)"/><line x1="0" y1="108.0" x2="200" y2="108.0" style="stroke:var(--line-soft)"/><line x1="100.0" y1="8" x2="100.0" y2="208" style="stroke:var(--line-soft)"/><circle cx="94.5" cy="105.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="110.5" cy="108.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="95.2" cy="113.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="93.3" cy="106.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="80.5" cy="119.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="95.4" cy="100.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="123.0" cy="101.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="108.7" cy="103.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="121.4" cy="91.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="105.0" cy="98.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="108.1" cy="92.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="103.7" cy="126.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="65.2" cy="131.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="117.7" cy="103.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="110.4" cy="93.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="110.2" cy="111.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="64.6" cy="139.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="63.6" cy="132.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="81.3" cy="126.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="90.1" cy="115.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="106.2" cy="96.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="98.9" cy="108.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="110.7" cy="99.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="86.5" cy="108.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="106.3" cy="94.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="108.1" cy="103.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="86.1" cy="98.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="135.6" cy="84.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="111.4" cy="90.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="124.8" cy="87.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="86.9" cy="119.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="84.5" cy="117.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="92.7" cy="113.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="97.6" cy="105.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="113.0" cy="108.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="105.0" cy="116.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="90.5" cy="112.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="79.9" cy="134.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="89.0" cy="126.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="125.3" cy="97.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="83.0" cy="113.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="105.0" cy="97.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="108.7" cy="88.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="68.8" cy="144.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="100.9" cy="107.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="127.0" cy="92.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="57.9" cy="140.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="93.2" cy="101.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="97.6" cy="117.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="82.8" cy="111.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="110.2" cy="91.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="98.6" cy="111.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="69.4" cy="152.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="117.1" cy="81.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="113.8" cy="96.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="119.5" cy="95.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="129.8" cy="77.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="107.4" cy="98.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="102.3" cy="94.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="72.8" cy="141.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="112.7" cy="87.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="87.1" cy="108.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="90.4" cy="105.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="73.5" cy="134.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="79.7" cy="133.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="88.8" cy="110.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="126.7" cy="82.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="57.6" cy="146.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="69.5" cy="124.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="104.8" cy="106.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="129.9" cy="99.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="111.9" cy="100.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="60.3" cy="159.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="47.4" cy="150.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="107.3" cy="99.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="84.5" cy="127.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="76.5" cy="130.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="120.2" cy="83.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="122.8" cy="86.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="103.1" cy="94.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="105.0" cy="104.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="108.9" cy="91.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="133.0" cy="65.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="112.7" cy="83.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="110.7" cy="104.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="111.3" cy="90.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="67.2" cy="153.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="126.5" cy="92.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="119.7" cy="106.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="110.9" cy="89.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="58.8" cy="156.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="86.7" cy="120.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="117.4" cy="93.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="62.2" cy="143.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="96.0" cy="116.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="121.1" cy="87.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="72.6" cy="119.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="133.4" cy="77.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="111.3" cy="93.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="96.7" cy="103.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="106.6" cy="103.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="113.4" cy="106.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="102.4" cy="110.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="123.7" cy="77.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="86.1" cy="134.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="91.2" cy="121.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="121.5" cy="80.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="100.4" cy="101.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="81.5" cy="125.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="119.6" cy="83.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="130.4" cy="79.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="90.6" cy="126.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="71.1" cy="147.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="97.1" cy="116.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="96.8" cy="103.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="93.7" cy="118.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="129.1" cy="88.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="78.5" cy="134.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="126.1" cy="96.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="73.5" cy="133.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="83.5" cy="133.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="113.0" cy="93.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="123.4" cy="106.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="117.7" cy="89.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="107.0" cy="107.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="102.8" cy="121.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="103.0" cy="99.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="111.8" cy="99.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="96.2" cy="129.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="105.6" cy="110.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="111.8" cy="95.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="99.9" cy="112.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="115.8" cy="87.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="111.6" cy="91.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="141.7" cy="64.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="106.6" cy="99.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="91.0" cy="105.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="92.1" cy="110.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="99.6" cy="105.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="119.1" cy="107.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="92.9" cy="107.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="107.9" cy="90.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="138.1" cy="75.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="46.5" cy="161.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="76.5" cy="114.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="104.9" cy="118.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="108.1" cy="97.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="104.8" cy="84.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="90.9" cy="124.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="113.5" cy="90.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="105.7" cy="87.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="89.0" cy="119.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="150.4" cy="57.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="107.2" cy="94.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="88.3" cy="126.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="97.8" cy="111.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="95.2" cy="110.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="98.6" cy="103.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="43.1" cy="161.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="89.7" cy="119.4" r="2.4" style="fill:var(--ink);opacity:.5"/><line x1="0.0" y1="198.0" x2="200.0" y2="18.0" style="stroke:var(--amber-link);stroke-width:2;stroke-dasharray:6 4"/><text x="100.0" y="230" text-anchor="middle" style="fill:var(--ink);font:600 15px var(--sans)">ρ = 0.9</text><text x="100.0" y="250" text-anchor="middle" style="fill:var(--muted);font:14px var(--sans)">Cov = 0.9, slope = 0.9</text><rect x="224" y="8" width="200" height="200" style="fill:var(--panel);stroke:var(--line)"/><line x1="224" y1="108.0" x2="424" y2="108.0" style="stroke:var(--line-soft)"/><line x1="324.0" y1="8" x2="324.0" y2="208" style="stroke:var(--line-soft)"/><circle cx="318.5" cy="90.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="334.5" cy="130.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="319.2" cy="110.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="317.3" cy="91.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="304.5" cy="93.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="319.4" cy="81.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="347.0" cy="139.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="332.7" cy="115.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="345.4" cy="114.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="329.0" cy="97.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="332.1" cy="88.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="327.7" cy="158.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="289.2" cy="90.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="341.7" cy="134.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="334.4" cy="95.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="334.2" cy="136.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="288.6" cy="107.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="287.6" cy="88.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="305.3" cy="112.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="314.1" cy="105.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="330.2" cy="93.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="322.9" cy="106.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="334.7" cy="110.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="310.5" cy="81.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="330.3" cy="89.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="332.1" cy="113.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="310.1" cy="58.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="359.6" cy="128.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="335.4" cy="91.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="348.8" cy="112.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="310.9" cy="107.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="308.5" cy="96.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="316.7" cy="105.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="321.6" cy="97.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="337.0" cy="136.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="329.0" cy="136.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="314.5" cy="97.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="303.9" cy="127.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="313.0" cy="128.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="349.3" cy="135.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="307.0" cy="86.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="329.0" cy="94.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="332.7" cy="81.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="292.8" cy="127.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="324.9" cy="108.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="351.0" cy="128.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="281.9" cy="96.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="317.2" cy="79.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="321.6" cy="125.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="306.8" cy="80.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="334.2" cy="90.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="322.6" cy="112.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="293.4" cy="147.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="341.1" cy="81.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="337.8" cy="110.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="343.5" cy="119.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="353.8" cy="100.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="331.4" cy="100.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="326.3" cy="80.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="296.8" cy="129.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="336.7" cy="87.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="311.1" cy="81.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="314.4" cy="82.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="297.5" cy="113.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="303.7" cy="123.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="312.8" cy="90.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="350.7" cy="105.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="281.6" cy="108.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="293.5" cy="83.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="328.8" cy="113.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="353.9" cy="150.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="335.9" cy="115.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="284.3" cy="145.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="271.4" cy="96.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="331.3" cy="102.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="308.5" cy="121.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="300.5" cy="110.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="344.2" cy="92.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="346.8" cy="106.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="327.1" cy="84.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="329.0" cy="109.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="332.9" cy="89.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="357.0" cy="79.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="336.7" cy="78.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="334.7" cy="120.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="335.3" cy="92.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="291.2" cy="145.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="350.5" cy="127.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="343.7" cy="144.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="334.9" cy="88.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="282.8" cy="133.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="310.7" cy="109.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="341.4" cy="111.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="286.2" cy="111.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="320.0" cy="120.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="345.1" cy="103.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="296.6" cy="76.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="357.4" cy="106.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="335.3" cy="98.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="320.7" cy="90.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="330.6" cy="112.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="337.4" cy="131.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="326.4" cy="119.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="347.7" cy="87.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="310.1" cy="140.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="315.2" cy="120.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="345.5" cy="89.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="324.4" cy="94.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="305.5" cy="109.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="343.6" cy="93.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="354.4" cy="104.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="314.6" cy="131.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="295.1" cy="139.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="321.1" cy="120.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="320.8" cy="91.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="317.7" cy="119.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="353.1" cy="124.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="302.5" cy="124.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="350.1" cy="136.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="297.5" cy="112.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="307.5" cy="131.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="337.0" cy="101.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="347.4" cy="151.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="341.7" cy="101.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="331.0" cy="120.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="326.8" cy="144.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="327.0" cy="95.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="335.8" cy="113.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="320.2" cy="150.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="329.6" cy="124.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="335.8" cy="102.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="323.9" cy="117.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="339.8" cy="93.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="335.6" cy="94.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="365.7" cy="94.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="330.6" cy="102.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="315.0" cy="84.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="316.1" cy="97.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="323.6" cy="100.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="343.1" cy="146.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="316.9" cy="92.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="331.9" cy="84.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="362.1" cy="112.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="270.5" cy="120.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="300.5" cy="73.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="328.9" cy="141.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="332.1" cy="99.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="328.8" cy="63.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="314.9" cy="126.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="337.5" cy="95.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="329.7" cy="73.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="313.0" cy="111.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="374.4" cy="96.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="331.2" cy="91.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="312.3" cy="126.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="321.8" cy="110.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="319.2" cy="103.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="322.6" cy="93.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="267.1" cy="112.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="313.7" cy="113.0" r="2.4" style="fill:var(--ink);opacity:.5"/><line x1="224.0" y1="108.0" x2="424.0" y2="108.0" style="stroke:var(--amber-link);stroke-width:2;stroke-dasharray:6 4"/><text x="324.0" y="230" text-anchor="middle" style="fill:var(--ink);font:600 15px var(--sans)">ρ = 0</text><text x="324.0" y="250" text-anchor="middle" style="fill:var(--muted);font:14px var(--sans)">Cov = 0, slope = 0</text><rect x="448" y="8" width="200" height="200" style="fill:var(--panel);stroke:var(--line)"/><line x1="448" y1="108.0" x2="648" y2="108.0" style="stroke:var(--line-soft)"/><line x1="548.0" y1="8" x2="548.0" y2="208" style="stroke:var(--line-soft)"/><circle cx="537.1" cy="105.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="569.0" cy="108.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="538.3" cy="113.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="534.6" cy="106.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="509.0" cy="119.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="538.8" cy="100.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="594.0" cy="101.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="565.4" cy="103.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="590.9" cy="91.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="558.1" cy="98.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="564.1" cy="92.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="555.4" cy="126.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="478.4" cy="131.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="583.3" cy="103.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="568.8" cy="93.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="568.5" cy="111.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="477.3" cy="139.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="475.1" cy="132.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="510.7" cy="126.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="528.2" cy="115.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="560.4" cy="96.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="545.8" cy="108.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="569.4" cy="99.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="521.0" cy="108.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="560.6" cy="94.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="564.1" cy="103.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="520.2" cy="98.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="619.2" cy="84.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="570.9" cy="90.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="597.5" cy="87.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="521.9" cy="119.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="516.9" cy="117.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="533.4" cy="113.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="543.3" cy="105.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="574.0" cy="108.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="558.1" cy="116.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="529.1" cy="112.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="507.9" cy="134.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="526.0" cy="126.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="598.5" cy="97.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="514.1" cy="113.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="557.9" cy="97.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="565.5" cy="88.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="485.7" cy="144.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="549.7" cy="107.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="602.1" cy="92.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="463.8" cy="140.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="534.3" cy="101.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="543.3" cy="117.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="513.7" cy="111.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="568.4" cy="91.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="545.1" cy="111.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="486.7" cy="152.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="582.2" cy="81.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="575.6" cy="96.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="587.1" cy="95.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="607.7" cy="77.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="562.8" cy="98.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="552.7" cy="94.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="493.6" cy="141.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="573.3" cy="87.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="522.2" cy="108.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="528.9" cy="105.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="495.1" cy="134.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="507.4" cy="133.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="525.6" cy="110.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="601.4" cy="82.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="463.1" cy="146.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="487.0" cy="124.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="557.7" cy="106.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="607.8" cy="99.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="571.8" cy="100.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="468.6" cy="159.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="562.6" cy="99.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="517.1" cy="127.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="501.1" cy="130.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="588.4" cy="83.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="593.6" cy="86.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="554.3" cy="94.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="557.9" cy="104.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="565.8" cy="91.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="614.1" cy="65.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="573.5" cy="83.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="569.3" cy="104.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="570.5" cy="90.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="482.4" cy="153.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="601.1" cy="92.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="587.5" cy="106.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="569.8" cy="89.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="465.5" cy="156.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="521.3" cy="120.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="582.8" cy="93.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="472.3" cy="143.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="540.1" cy="116.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="590.2" cy="87.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="493.1" cy="119.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="614.7" cy="77.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="570.7" cy="93.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="541.5" cy="103.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="561.2" cy="103.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="574.8" cy="106.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="552.7" cy="110.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="595.4" cy="77.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="520.2" cy="134.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="530.4" cy="121.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="591.1" cy="80.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="548.8" cy="101.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="511.1" cy="125.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="587.1" cy="83.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="608.7" cy="79.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="529.2" cy="126.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="490.3" cy="147.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="542.1" cy="116.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="541.5" cy="103.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="535.3" cy="118.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="606.2" cy="88.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="505.0" cy="134.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="600.2" cy="96.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="494.9" cy="133.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="514.9" cy="133.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="574.0" cy="93.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="594.7" cy="106.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="583.5" cy="89.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="562.1" cy="107.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="553.6" cy="121.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="554.1" cy="99.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="571.7" cy="99.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="540.4" cy="129.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="559.3" cy="110.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="571.6" cy="95.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="547.7" cy="112.2" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="579.5" cy="87.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="571.3" cy="91.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="631.4" cy="64.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="561.2" cy="99.6" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="529.9" cy="105.9" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="532.2" cy="110.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="547.2" cy="105.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="586.2" cy="107.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="533.7" cy="107.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="563.8" cy="90.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="624.2" cy="75.7" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="500.9" cy="114.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="557.9" cy="118.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="564.3" cy="97.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="557.6" cy="84.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="529.8" cy="124.3" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="575.0" cy="90.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="559.5" cy="87.8" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="526.0" cy="119.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="562.5" cy="94.4" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="524.6" cy="126.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="543.6" cy="111.1" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="538.3" cy="110.5" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="545.1" cy="103.0" r="2.4" style="fill:var(--ink);opacity:.5"/><circle cx="527.4" cy="119.4" r="2.4" style="fill:var(--ink);opacity:.5"/><line x1="448.0" y1="153.0" x2="648.0" y2="63.0" style="stroke:var(--amber-link);stroke-width:2;stroke-dasharray:6 4"/><text x="548.0" y="230" text-anchor="middle" style="fill:var(--ink);font:600 15px var(--sans)">X stretched ×2: ρ = 0.9</text><text x="548.0" y="250" text-anchor="middle" style="fill:var(--muted);font:14px var(--sans)">Cov = 1.8, slope = 0.45</text></svg><figcaption>Same axes in every panel; the dashed line is the best-fit line for predicting Y from X (slope = Cov / Var(X)). Stretching X keeps ρ at 0.9 but doubles Cov and halves the slope.</figcaption></figure>

- Cov matrix: $D \times D$ covariance matrix, encodes that pair-wise
  information for D multi-dimension Gaussians.

  $$\Sigma = \begin{pmatrix} V[X_1] & \mathrm{Cov}[X_1, X_2] & \cdots & \mathrm{Cov}[X_1, X_D] \\ \mathrm{Cov}[X_2, X_1] & V[X_2] & \cdots & \mathrm{Cov}[X_2, X_D] \\ \vdots & \vdots & \ddots & \vdots \\ \mathrm{Cov}[X_D, X_1] & \mathrm{Cov}[X_D, X_2] & \cdots & V[X_D] \end{pmatrix}$$

  - Geometrically, for multivariate Gaussians: variance captures how spread it
    is along each axis, cov captures how the two variables move together, and correlation (cov
    normalized by the two standard deviations) captures how tightly the point
    cloud hugs the line when viewed on a plane with 2 variables.
    - Note: hugging a line means that the deviations relative to their own
      means move in a fixed ratio and direction, since variance is a measure
      of deviation, not the means themselves.
    - Note 2: $\rho$ is set by the ratio of shared signal to independent noise.
      More shared signal raises it, more independent noise lowers it. Rescaling
      an axis stretches both by the same factor, so Cov and slope change but
      $\rho$ doesn't.
- Overall: covariance matrix allows us to mathematically describe the
  relationship between RVs laid along different dimensions.

#### Conditioning on Gaussians

$$\boxed{\begin{aligned}
p(y_1 \mid y_2) &= \mathcal{N}(y_1 \mid \mu_{1|2}, \Sigma_{1|2}) \\
\mu_{1|2} &= \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (y_2 - \mu_2) \\
\Sigma_{1|2} &= \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}
\end{aligned}}$$

This formula answers how one block of a joint Gaussian is conditioned on the
other block.
Specifically, given 2 RVs that may be correlated, how much do we shift our
"prior guess" based on how strong the relationship of the 2 RVs is, as
measured by covariance.

Mean:

- $\mu_1$ is our prior guess
- $\Sigma_{12}\Sigma_{22}^{-1}$ tells us how related the 2 vars are.
  Specifically, this is a ratio of how much of the total noise in
  $\Sigma_{22}$ (which is $V[S] + V[N_2]$, if you want to think about the
  framing about "shared" noise from above) is shared.
  - Note: don't $y_1$ and $y_2$ have their own unit scales? Quick example:
    measuring $y_2$ in cm instead of m multiplies $\Sigma_{12}$ by 100 and
    $\Sigma_{22}$ by 10,000, so $\Sigma_{12}\Sigma_{22}^{-1}$ shrinks 100x,
    but $(y_2 - \mu_2)$ grows 100x. But notice how the scale contributes to
    both $\Sigma_{12}$ and $\Sigma_{22}$, and the mean eventually absorbs the
    diff here. Thus, $\Sigma_{22}$ can only grow
    independently of $\Sigma_{12}$ if $y_2$ has some extra noise that $y_1$
    doesn't have.
  - Note 2: loosely, $\Sigma_{12}\Sigma_{22}^{-1} = \dfrac{V[S]}{V[S] + V[N_2]}$.
    If $y_2$ has more independent noise, this shrinks and we lean on $y_2$
    less, because intuitively $y_2$ gives less information about $y_1$: more
    of its movement is its own noise, not the shared part.
- $(y_2 - \mu_2)$: correlation is defined relative to means. It says if $y_2$
  deviates this much from average, $y_1$ moves by the shared fraction of that
  deviation.

Variance intuition is similar:

- Similarly, we start with $\Sigma_{11}$ as the original guess
- The variance decreases by $\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$: the same
  shared ratio from the mean ($\Sigma_{12}\Sigma_{22}^{-1}$), times the shared
  portion $\Sigma_{21} = V[S]$. So we only remove the shared noise we can
  actually recover through $y_2$, which works out to $\rho^2\Sigma_{11}$.
  What's left is $y_1$'s own noise $V[N_1]$, plus the part of $S$ we can't
  pin down because $y_2$ is noisy.
  - Note: same units question here, but $y_2$'s scale gets multiplied on
    both sides: $\Sigma_{12}$ and $\Sigma_{21}$ each carry it once (x100 each
    for cm vs m), $\Sigma_{22}$ carries it twice (x10,000), so it cancels
    within the formula itself.

1. Once we're able to condition on a Gaussian, we're able to do Bayesian
   inference with Gaussians, which is closed because the conditional is
   Gaussian as well.
   1. Review this again, it's kinda hard to reason through:
      1. Conditioning on a Gaussian
      2. Strong + weak priors, strong + weak likelihood
2. Linear Gaussian models: when you apply weight $W$ that is $D \times L$,
   where $z \in \mathbb{R}^L$ is an unknown value vector and
   $y \in \mathbb{R}^D$ is some noisy observation of $z$. Similar to
   regression?

- Overall: the conditional of a Gaussian is still a Gaussian. With it, now
  we're able to work through some interesting examples.

#### Worked examples

**Example #0: 2D Gaussian conditioning.** Concretely, this is how the formula
above looks when applied:

$$p(y_1 \mid y_2) = \mathcal{N}\left(y_1 \,\middle|\, \mu_1 + \frac{\rho\sigma_1\sigma_2}{\sigma_2^2}(y_2 - \mu_2),\; \sigma_1^2 - \frac{(\rho\sigma_1\sigma_2)^2}{\sigma_2^2}\right)$$

If $\sigma_1 = \sigma_2 = \sigma$, we get:

$$p(y_1 \mid y_2) = \mathcal{N}\big(y_1 \mid \mu_1 + \rho(y_2 - \mu_2),\; \sigma^2(1 - \rho^2)\big)$$

- We can see some general trends: if $y_2$ is mostly its own noise (low
  $\rho$), we don't trust it to provide a strong signal for $y_1$. If most of
  $y_2$ is the shared portion (high $\rho$), then $y_2$ becomes a much more
  helpful signal.

**Example #1: Hidden feature imputation.** When presented with only some
dimensions/features (but the full distribution, which is a little
unrealistic), we can leverage covariance to calculate confidence in the other
dimensions.

- Side note: feels similar to the "dark knowledge" concept in Hinton's
  distillation paper, where the relative probabilities of the softmax encode a
  lot of information beyond the most likely choice.
- $p(y_{n,h} \mid y_{n,v}, \theta)$, where $v$ are the indices of the visible
  entries in that example, $h$ are the remaining indices of the hidden
  entries, and $\theta = (\mu, \Sigma)$.
- For the "hidden" block, we apply conditioning given all the "visible" axes at once
  to obtain the posterior mean. In this case, $y_{n,v}$ is a vector since
  intuitively we want to use all the visible data we see and their deviations
  to shift the mean over, and the relevant "blocks" of the covariance matrix
  are the ones with $v$ entries. And $\Sigma_{hv}\Sigma_{vv}^{-1}$ is the
  same shared ratio as $\Sigma_{12}\Sigma_{22}^{-1}$ above, now across all the
  visible entries at once: the relative strength of the relationship, not the
  signal itself.

  $$\mu_{h|v} = \mu_h + \Sigma_{hv}\Sigma_{vv}^{-1}(y_{n,v} - \mu_v), \qquad \Sigma_{h|v} = \Sigma_{hh} - \Sigma_{hv}\Sigma_{vv}^{-1}\Sigma_{vh}$$

**Example #2: Bayes rule updating w/ noisy measurements.**
$\mathbf{y} = (y_1, \dots, y_N)$, $W = \mathbf{1}_N$,
$\Sigma_y^{-1} = \mathrm{diag}(\lambda_y I)$.

$$\begin{aligned}
p(z \mid \mathbf{y}) &= \mathcal{N}(z \mid \mu_N, \lambda_N^{-1}) \\
\lambda_N &= \lambda_0 + N\lambda_y \\
\mu_N &= \frac{N\lambda_y \bar{y} + \lambda_0 \mu_0}{\lambda_N} = \frac{N\lambda_y}{N\lambda_y + \lambda_0}\bar{y} + \frac{\lambda_0}{N\lambda_y + \lambda_0}\mu_0
\end{aligned}$$

- The more trials, the more the posterior mean goes to $\bar{y}$, and the
  posterior precision grows as $N\lambda_y$ (the variance shrinks toward 0).

Sequential update when $N = 1$, and rewritten in terms of variance:

$$\begin{aligned}
p(z \mid y) &= \mathcal{N}(z \mid \mu_1, \Sigma_1) \\
\Sigma_1 &= \left(\frac{1}{\Sigma_0} + \frac{1}{\Sigma_y}\right)^{-1} = \frac{\Sigma_y \Sigma_0}{\Sigma_0 + \Sigma_y} \\
\mu_1 &= \Sigma_1\left(\frac{\mu_0}{\Sigma_0} + \frac{y}{\Sigma_y}\right)
\end{aligned}$$

$$\begin{aligned}
\mu_1 &= \frac{\Sigma_y}{\Sigma_y + \Sigma_0}\mu_0 + \frac{\Sigma_0}{\Sigma_y + \Sigma_0}y \\
&= \mu_0 + (y - \mu_0)\frac{\Sigma_0}{\Sigma_y + \Sigma_0} \\
&= y - (y - \mu_0)\frac{\Sigma_y}{\Sigma_y + \Sigma_0}
\end{aligned}$$

This third equation gives a sense of how the posterior mean is updated based
on the relative "strengths" of prior vs data, which is just a relationship
between the variances.

- If prior variance is small, then this reduces down to just $\mu_0$, which
  means shrinkage is big (shrinkage means the estimate gets pulled from $y$ toward $\mu_0$). If
  prior variance is big, then $y$ is weighted more heavily.

SNR also measures this "shrinkage":

$$\mathrm{SNR} \triangleq \frac{\mathbb{E}[Z^2]}{\mathbb{E}[\epsilon^2]} = \frac{\Sigma_0 + \mu_0^2}{\Sigma_y}$$

where $z \sim \mathcal{N}(\mu_0, \Sigma_0)$ is the true signal,
$y = z + \epsilon$ is the observed signal, and
$\epsilon \sim \mathcal{N}(0, \Sigma_y)$ is the noise.

Multivariate version, with $N$ observations of a vector $z$:

$$p(\mathcal{D} \mid z) = \prod_{n=1}^{N} \mathcal{N}(y_n \mid z, \Sigma_y) \propto \mathcal{N}\left(\bar{y} \,\middle|\, z, \tfrac{1}{N}\Sigma_y\right)$$

- This is saying that when we make a bunch of observations (but
  multivariate), which is the extension of the small example above, it
  is exactly as informative as one observation of the data mean $\bar{y}$, with
  covariance $\Sigma_y/N$.

**Example #3: Sensor fusion.**

$$p(z, \mathbf{y}) = p(z) \prod_{m=1}^{M} \prod_{n=1}^{N_m} \mathcal{N}(y_{n,m} \mid z, \Sigma_m)$$

where $M$ is the number of sensors (measurement devices), $N_m$ is the number
of observations from sensor $m$, and
$\mathbf{y} = y_{1:N, 1:M} \in \mathbb{R}^K$. Our goal is to combine the
evidence together, to compute $p(z \mid \mathbf{y})$.

- Following along with the equation of Bayes from above, when we make
  multiple observations in one instance, we basically weigh them based on
  their relative precisions to each other and how they inform us of the
  ground truth $z$ (the 3rd axis here).

## 3. Bayes, Bayesian inference, MLE, MAP, Bayesian posterior predictive

So far we've been given $\theta$. Usually, we have to find it.

Aleatoric vs epistemic uncertainty: aleatoric is what we've been working with in
the sections above. One set of parameters (true variance) capture 
inherent uncertainty in data. When evaluating multiple sets of paramters, 
there's an extra layer of epistemic uncertainty since we're guessing waht the true 
paramters might be. 

Point estimates (MLE, MAP, even the posterior mean) collapse $\theta$ to one
number before predicting, so the epistemic part is thrown away. The Bayesian
way carries the whole distribution over $\theta$ through the prediction. 

### Point estimate methods

#### MLE

**MLE:** highest probability to training data (equals MAP under a flat prior), gives one
theta (think one set of weights)

$$\boxed{\hat\theta_{\mathrm{mle}} = \arg\max_\theta \, p(D \mid \theta)}$$

1. $p(D \mid \theta) = \prod_{n=1}^{N} p(y_n \mid x_n, \theta)$, we take the log to
   turn the product into a sum (and negate it, NLL, to minimize).
   - Note: why a product? We assume the data is iid: each example is an
     independent draw from the same distribution, given $\theta$. That lets the
     joint over all $N$ examples factor into one term per example, same as
     $P(A, B) = P(A) \cdot P(B)$:

     $$p(y_1, \dots, y_N \mid x_1, \dots, x_N, \theta) = p(y_1 \mid x_1, \theta) \cdots p(y_N \mid x_N, \theta)$$

     It's an assumption about how the data was collected, not something we
     prove, and it breaks for things like time series.
2. $\hat\theta_{\mathrm{mle}} = \arg\max_\theta \sum_{n=1}^{N} \log p(y_n \mid x_n, \theta)$
   and $\hat\theta_{\mathrm{mle}} = \arg\min_\theta -\sum_{n=1}^{N} \log p(y_n \mid x_n, \theta)$.

Why MLE is acceptable:

- Flat prior: MLE equals MAP when the prior is flat in $\theta$ (it favors no
  value before seeing data), so $\log p(\theta)$ is a constant and MAP collapses
  into MLE.
- KL divergence: minimizing the KL from the empirical data distribution to the
  model, $\mathrm{KL}(p_{\mathrm{emp}} \,\|\, p_\theta)$, is the same as
  minimizing NLL, since the entropy of the data doesn't depend on $\theta$.

  $$\mathrm{KL}(p_{\mathrm{emp}} \,\|\, p_\theta) = \underbrace{-\mathbb{H}(p_{\mathrm{emp}})}_{\text{no } \theta} \; - \; \frac{1}{N}\sum_{n=1}^{N} \log p(y_n \mid \theta)$$

**Worked examples:** these 3 examples all show that MLE works out to intuitive
solutions that represent the empirical data. The general approach is to plug the formula for the
distribution into the NLL, then solve for the critical point.

1. MLE on Bernoulli ($N_1$ = heads):

   $$\boxed{\hat\theta_{\mathrm{mle}} = \frac{N_1}{N_0 + N_1}}$$

   An intuitive result: just the fraction of heads out total coin clips

2. MLE on categorical (Lagrange multiplier for the constraint
   $\sum_k \theta_k = 1$):

   $$\boxed{\hat\theta_k = \frac{N_k}{N}}$$

   the empirical fraction of times event $k$ occurs.

3. MLE on Gaussians (solve for the partials with respect to the mean and
   $\sigma$):

   $$\hat\mu_{\mathrm{mle}} = \bar{y}, \qquad \hat\sigma^2_{\mathrm{mle}} = \frac{1}{N}\sum_{n=1}^{N} y_n^2 - \bar{y}^2 = \overline{y^2} - \bar{y}^2, \qquad \overline{y^2} \triangleq \frac{1}{N}\sum_{n=1}^{N} y_n^2$$

   - Sufficient statistics: $\bar{y}$ and $\overline{y^2}$ are enough to
     calculate the MLE, so we don't need the full raw data representation all
     the time.
   - Multivariate: it works out to the dataset mean and the dataset covariance
     (with $1/N$, the MLE, which is biased low; the unbiased version uses
     $1/(N-1)$).

     $$\hat\mu = \frac{1}{N}\sum_{n=1}^{N} y_n = \bar{y}, \qquad \hat\Sigma = \frac{1}{N}\sum_{n=1}^{N} (y_n - \bar{y})(y_n - \bar{y})^\top$$

**Linear regression:** there are simplifications that can be made to show that
other things are analogous to MLE.

- Reminder that linear regression is simply rewriting the mean in relation to a
  matrix transform $w^\top x$. Assuming variance is fixed, we plug into the NLL
  the same Gaussian, except with $w^\top x_n$ as the mean,
  $\mathcal{N}(y_n \mid w^\top x_n, \sigma^2)$.
- Full NLL, plugging in the Gaussian PDF:

  $$\begin{aligned}
  \mathrm{NLL}(w) &= -\sum_{n=1}^{N} \log \left[ \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( -\frac{(y_n - w^\top x_n)^2}{2\sigma^2} \right) \right] \\
  &= \underbrace{\frac{N}{2} \log(2\pi\sigma^2)}_{\text{no } w} + \underbrace{\frac{1}{2\sigma^2}}_{\text{scale}} \sum_{n=1}^{N} (y_n - w^\top x_n)^2
  \end{aligned}$$
- If we drop all constants ($\sigma$ included), the NLL boils down to
  - RSS $= \sum_{n=1}^{N} (y_n - w^\top x_n)^2$. The expression inside,
    $r_n = y_n - w^\top x_n$, is known as the residual error, and this
    objective is known as the residual sum of squares (more simplified
    objective).
  - MSE (mean squared error) $= \frac{1}{N}\mathrm{RSS}$, so normalized.
  - RMSE $= \sqrt{\mathrm{MSE}}$, the root mean squared error.
  - Note to self: why do we need all 3, I can see that RSS is simplified
    without constants. If $\sigma^2$ is unknown but shared across
    points, it drops out of the $w$ gradient, so these still work; only a
    per-point variance changes it (weighted least squares).

**ERM:** generalize MLE by replacing the log loss term.

$$\hat\theta = \arg\min_\theta \frac{1}{N}\sum_{n=1}^{N} \ell(y_n, \theta; x_n)$$

MLE is the special case $\ell = -\log p(y_n \mid x_n, \theta)$.

**Method of moments:** a simpler alternative for when solving
$\nabla\mathrm{NLL}(\theta) = 0$ has no closed form (e.g. mixture models),
usually less statistically efficient than MLE.

- Theoretical moment of the distribution = empirical moment of the
  distribution, solve a set of $K$ simultaneous equations.
- The theoretical moments are given by $\mu_k = \mathbb{E}[Y^k]$, for $k = 1, \dots, K$.
- The empirical moments are given by $\hat\mu_k = \frac{1}{N}\sum_{n=1}^{N} y_n^k$.
- (Work through this)

#### Regularization and MAP

MLE, ERM (and MoM) can overfit to the data set. The
underlying issue is that the empirical distribution is often not the same as
the true distribution, so the model won't be able to generalize.

- Solution: add a regularization term or parameter. $C(\theta)$ is the
  complexity penalty while $\lambda$ = regularization strength.
- MAP: by setting $C(\theta) = -\log p(\theta)$, the negative log of some prior,
  we get MAP estimation.

**MAP:** the mode of the posterior (single most probable theta given $D$), point estimate, gives one theta →
literally one set of weights, `model.pt`.

$$\boxed{\hat\theta_{\mathrm{map}} = \arg\max_\theta \, p(\theta \mid D)}$$

**Regularization to prevent overfitting from MLE**

$$\mathcal{L}(\theta; \lambda) = \left[\frac{1}{N}\sum_{n=1}^{N} \ell(y_n, \theta; x_n)\right] + \lambda C(\theta)$$

where the regularizer is typically the negative log of some prior,
$\lambda C(\theta) = -\log p(\theta)$ (up to the $1/N$ on the loss, which just
rescales $\lambda$), with $\ell$ the NLL. This reduces down to MAP:

$$\hat\theta = \arg\max_\theta \log p(\theta \mid D) = \arg\max_\theta \left[\log p(D \mid \theta) + \log p(\theta) - \text{const}\right]$$

**Example #2: Multivariate MAP Gaussian.** Including the log of some prior:

- Inverse Wishart prior (work through this).

$$\hat\Sigma_{\mathrm{map}} = \lambda \Sigma_0 + (1 - \lambda)\hat\Sigma_{\mathrm{mle}}$$

- This $\lambda \in [0, 1]$ is how much weight the prior gets (it grows with
  prior strength and shrinks as $N$ grows), not the regularization $\lambda$
  above.

**Example #3: Weight decay.** L2 regularization, penalizing big weight vectors.

- $C(\theta) = \|\theta\|_2^2$ is the negative log of a zero-mean Gaussian prior
  (up to scale), so weight decay is MAP with
  $p(\theta) = \mathcal{N}(\theta \mid 0, \sigma_0^2 I)$.

  $$C(\theta) = \|\theta\|_2^2 \quad \Longleftrightarrow \quad -\log \mathcal{N}(\theta \mid 0, \sigma_0^2 I) = \frac{1}{2\sigma_0^2}\|\theta\|_2^2 + \text{const}$$

**Example #4: Early stopping.** Haha, surprisingly self-explanatory.

It's not a $C(\theta)$ term: it regularizes by limiting how far the optimizer
travels from its init, stopping when validation loss stops improving.

#### Choosing the regularization strength

- Grid search over candidate values of $\lambda$ using a validation set, pick
  the $\lambda$ with the lowest validation loss, then retrain on train +
  validation with that strength.
- Generally, with more data, overfit reduces because the empirical data
  distribution becomes close to the actual one.

### Bayesian inference and the posterior predictive (4.6)

Two steps. First, Bayes rule conditions the prior on the data to get the
posterior over the parameters. Same as the Bayes rule setup in section 0, except the unknown is
$\theta$, and we condition on a dataset $D$:

$$p(\theta \mid D) = \frac{p(\theta)\, p(D \mid \theta)}{p(D)}
= \frac{p(\theta)\, p(D \mid \theta)}{\int p(\theta')\, p(D \mid \theta')\, d\theta'}$$

$p(D)$ is the marginal likelihood (the average probability of the data under the
prior). It doesn't depend on $\theta$, so we ignore it when we only want the
relative probabilities of $\theta$ values.

Second, average $p(y \mid x, \theta)$ over $\theta$, weighted by $p(\theta \mid D)$:

$$p(y \mid x, D) = \int p(y \mid x, \theta)\, p(\theta \mid D)\, d\theta$$

Plugging in a point estimate is the same integral with the posterior replaced by
a spike, $p(\theta \mid D) \approx \delta(\theta - \hat\theta)$:

$$\begin{aligned}
\text{MAP:} \quad & p(y \mid x, D) \approx p(y \mid x, \hat\theta_{\text{MAP}}), \quad \hat\theta_{\text{MAP}} = \arg\max_\theta\, p(\theta \mid D) \\
\text{Posterior mean:} \quad & p(y \mid x, D) \approx p(y \mid x, \mathbb{E}[\theta \mid D]) \\
\text{Bayesian:} \quad & p(y \mid x, D) = \int p(y \mid x, \theta)\, p(\theta \mid D)\, d\theta
\end{aligned}$$

MLE is the MAP line with a flat prior, $\hat\theta_{\text{MLE}} = \arg\max_\theta\, p(D \mid \theta)$.
The first two are $p(y \mid x, \text{one }\theta)$; the last is $\mathbb{E}_{p(\theta \mid D)}[\,p(y \mid x, \theta)\,]$.

- At prediction time each $\theta$ makes its own prediction $p(y \mid x, \theta)$;
  the posterior predictive averages these, weighting each $\theta$ by how
  plausible it is after seeing $D$, $p(\theta \mid D)$.

**Example #1: Gaussian-Gaussian model.** It's the same analysis as we did above
about how we distribute the mean + variance of a gaussian based on its strength
(variance); precision format is easier to reason through. Also this is the
"conjugate prior" idea: a Gaussian prior on the mean of a Gaussian likelihood
(known variance) gives a Gaussian posterior, the same family as the prior. Here
$\kappa = 1/\sigma^2$ is the known noise precision, and $\breve\lambda$,
$\breve m$ are the prior precision and mean.

$$\boxed{\begin{aligned}
\lambda_N &= \breve\lambda + N\kappa \\
m_N &= \frac{N\kappa\,\bar y + \breve\lambda\,\breve m}{\lambda_N}
 = \frac{N\kappa}{N\kappa + \breve\lambda}\,\bar y + \frac{\breve\lambda}{N\kappa + \breve\lambda}\,\breve m
\end{aligned}}$$

$$p(y \mid D) = \int \mathcal{N}(y \mid \theta, 1/\kappa)\,\mathcal{N}(\theta \mid m_N, 1/\lambda_N)\, d\theta
= \mathcal{N}\!\left(y \,\middle|\, m_N,\ \tfrac{1}{\kappa} + \tfrac{1}{\lambda_N}\right)$$

- $(m_N, \lambda_N)$ is not a point estimate. It is the whole posterior: a
  bell curve over what $\theta$ could be, centered at $m_N$, with variance
  $1/\lambda_N$. More data slides it toward $\bar y$ and narrows it.
- How it's maintained: the posterior after $N$ points is the prior for point
  $N+1$. Because the posterior is still a Gaussian (conjugacy), the update is
  closed under itself, so we only carry two numbers and fold in each new $y$:
  $\lambda \leftarrow \lambda + \kappa$, $\; m \leftarrow \frac{\kappa y + \lambda m}{\lambda + \kappa}$
  (using the old $\lambda$ on the right). One at a time or all $N$ at once gives
  the same $(m_N, \lambda_N)$.
- The difference from MLE/MAP is the last step. They plug in one $\hat\theta$
  and predict with variance $1/\kappa$ (aleatoric). Here we average over the
  whole curve, so the predictive variance also carries $1/\lambda_N$, our
  uncertainty about $\theta$ (epistemic).
- Posterior variance gives a confidence in the estimate of $\mu$. MLE gives
  $\hat\sigma^2$ (the noise in one $y$), but the point estimate $\hat\mu$
  carries no uncertainty on its own; the posterior hands you one directly (a
  separate sampling-distribution argument gives the same $s/\sqrt{N}$ here). The standard error of the mean is $\text{se}(\mu) = \sqrt{\mathbb{V}[\mu \mid D]}$.
  With an uninformative prior ($\breve\lambda = 0$) the posterior mean equals the
  MLE, $m_N = \bar y$, and approximating $\sigma^2$ by the sample variance $s^2$
  gives $\lambda_N = N/s^2$:

$$\text{se}(\mu) = \frac{1}{\sqrt{\lambda_N}} = \frac{s}{\sqrt{N}}
\qquad\Rightarrow\qquad I_{.95}(\mu \mid D) \approx \bar y \pm 2\,\frac{s}{\sqrt{N}}$$

  So the uncertainty in $\mu$ shrinks at rate $1/\sqrt{N}$, while the predictive
  variance never drops below the noise $s^2$.
