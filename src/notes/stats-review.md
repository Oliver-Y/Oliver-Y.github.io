---
title: Stats review
date: 2026-09-24
blurb: Review of Ch 2-4 of Probabilistic Machine Learning — random variables, the distribution catalogue, Gaussians, and Bayesian inference.
---

I wish I took the time to understand stats in high school or college and I
think it's one of those subjects that's important to learn and internalize. At
the root of it (at least in my opinion) is it provides a way to quantify
uncertainty. It has heavy applications in all these ML papers I read but also,
feels like it'd be a useful tool in confronting a world that feels incredibly
more complex by the second. So here's my, verbose and probably imprecise notes
on learning/re-learning the foundations.

## 0. RV — Random variables

I find myself constantly needing to remind myself, that at the core of it, it's
just some random variable X with some set of outcomes (state space) and it's
about assigning a probability to every outcome.

The nature of X can change which gives us more information, e.g. continuous,
discrete, as well as having various relationships w/ other random variables
such as conditionals, joint probabilities, marginals. But at the end of the
day, we're just placing probability densities/mass over RVs.

### PMF, CDF, PDF

- PMF: probability mass function: discrete X describing $P(X = x)$ over state
  space
- CDF: cumulative distribution function: continuous X describing $P(X \le x)$
- PDF: probability density function, derivative of CDF, integrates to
  probability. Loosely describes a small interval on continuous X where
  probability changes a lot.
  - Note: the terms "mass"/"density" help here. Density can only give mass
    when given (volume) — or in this case a closed interval to integrate over.

### Conditionals, Independence, Joint probabilities

Random variables are often related/un-related to each other and that gives more
insight.

- Joint probability: $P(A, B)$ — if independent, then $P(A) \cdot P(B)$
- Conditional probability: $P(A \mid B) = \dfrac{P(A, B)}{P(B)}$
- Marginal: $P(X{=}x) = \sum_y P(X{=}x, Y{=}y)$ — think slicing the plane along
  an axis here

### Bayes rule

- Super important.
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
parameterized ways to describe common RVs we'll see. This may be discrete /
binary outcomes, one-hot encodings, or even continuous values. I only went
through a subset within this textbook, but it seems distributions generally
provide a structured/parameterized way to categorize probability mass/density.

### Bernoulli / Binomial

Bernoulli: params θ, X results in binary events.

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
  0)$, so only one probability is active at one time. Although, the params (θ)
  kind of already give the direct prob distribution? So the novelty is just the
  encoding? Feels a touch weird here.

Multinomial: params (C−1) θ, N — N number of categorical events.

$$\mathrm{Mult}(\mathbf{s} \mid N, \theta) \triangleq \binom{N}{s_1, \dots, s_C} \prod_{c=1}^{C} \theta_c^{s_c}$$

- Softmax w/ temperature is often used to deal with the θ-sums-up-to-1
  constraint so we can work with raw logits. Temperature controls how spread
  the probabilities are.
  - Softmax equation: exponentiates + normalizes

  $$\mathrm{softmax}(z)_c = \frac{e^{z_c / T}}{\sum_{c'=1}^{C} e^{z_{c'} / T}}$$
- Logistic regression on multinomial case is just the multi-dimensional
  version of $w x + b$. $W = C \times D$ matrix, $b$ = C dim vector. We
  parameterize by $W$ instead, which indirectly controls the θ's through a
  linear transform.

### Gaussians

Gaussian: params: mean, variance over continuous RV X, in PDF form:

$$\mathcal{N}(y \mid \mu, \sigma^2) \triangleq \frac{1}{\sqrt{2\pi\sigma^2}}\, e^{-\frac{1}{2\sigma^2}(y - \mu)^2}$$

- The $e^{-x}$ gives it the bell curve shape
- Mean = expectation, value weighted by probability:
  $\mathbb{E}[Y] \triangleq \int_Y y\, p(y)\, dy$
- Variance = spread of data: $V[Y] \triangleq \mathbb{E}\big[(Y - \mu)^2\big]$
- Note: $p(y \mid x; \theta) = \mathcal{N}(y \mid w^T x + b, \sigma^2)$ is
  useful in this form as well (linear regression), s.t. we parameterize by
  input X and weights w, b with set variance
- Central limit theorem: sum of IID samples are generally gaussian. Making it
  pretty common for modeling residuals/noisy information.
  - Noise usually is a group of independent things that add onto the end result
  - CLT generally doesn't care what the underlying distribution is. As long as
    we sum IID, the net sum ends up being Gaussian (neat trick: just pick a
    space where it's additive (log))
- Maximum entropy: variance is a good way to retain high entropy, not assume
  too much structure

Couple other distributions I didn't go into too much detail: Cauchy, Gamma,
Student-T.

## 2. Deep dive on Gaussians, multi-dim + manipulating Gaussians

### Multi-dimensional Gaussians

Covariance matrix: $\mathrm{Cov}[X, Y] \triangleq \mathbb{E}\big[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])\big]$.
Expectation of (random variable - mean on axis) times (random variable - mean
on other axis). Positive means vary in same direction.

- DxD covariance matrix
- Intuitively, this captures the relationship between 2 variables — when 1
  variable moves does the other move together? This gives us a sense of what
  other dimension values should be given partial dimensions. E.g. imputing a
  hidden value + conditioning on a Gaussian.
- Geometrically, variance quantifies how spread out it is along each axis, but
  covariance is: if u average those things out, does it still give u the
  general slope?
- How does the spherical line slope projected on the XY plane change if
  variance increases, what does that do to the covariance? Points are more
  "spread" along one axis but because it's a gaussian and spread is symmetric,
  that doesn't actually change the covariance right? I mean the number in
  theory should get bigger if variance gets bigger? But the correlation I want
  to say stays constant?

Correlation: $\mathrm{corr}[X, Y] \triangleq \dfrac{\mathrm{Cov}[X, Y]}{\sqrt{V[X]\, V[Y]}}$ —
normalized by variance gives u actual slope of lines.

### Manipulating Gaussians (linear transform, sum, sign flip, conditioning)

#### Conditioning on Gaussians

tbh, kinda hard to reason through.

- $p(y_1 \mid y_2) = \mathcal{N}(y_1 \mid \mu_{1\mid 2}, \Sigma_{1\mid 2})$
- Linear gaussian models: when you apply weight $W$ that is $D \times L$,
  where $z$ is $\mathbb{R}^L$ unknown value vector and $y$ is $\mathbb{R}^D$
  some noisy observation of $z$.
- Once we're able to condition on a Gaussian, we're able to do Bayesian
  inference with gaussians, which is closed because the conditional is Gaussian
  as well.
  - Review this again, it's kinda hard to reason through:
    1. Conditioning on a Gaussian
    2. Strong + weak priors, strong + weak likelihood
- Examples: sensor fusion, hidden vectors. Probably want to work through these
  examples.

## 3. Bayes, Bayesian inference, MLE, MAP, Bayesian posterior predictive

Ch 4 review: estimating parameters from data seen, Bayes on $P(\mathrm{param} \mid \mathrm{data})$.

- MLE + MAP are ways for us to infer (model fitting) to figure out what the
  right parameters are. While the previous ways — Bayes + distributions
  (multi-dim) — are the structures we populate with MLE + MAP.
- MLE: highest probability to training data: $\arg\max_\theta p(D \mid \theta)$,
  assumes uniform prior, gives one theta → literally one set of weights,
  `model.pt`.
- MAP: highest probability to posterior: $\arg\max_\theta p(\theta \mid D)$ →
  point estimate, gives one theta → literally one set of weights, `model.pt`.
- Bayesian posterior predictive: integrate over $p(\theta \mid D)$ and weight
  $p(y \mid x, \theta)$.
  - At prediction time, how well does this one theta explain $y \mid x$, given
    how likely/unlikely we think this $p(\theta \mid D)$ is as modeled by our
    posterior.
- Examples: all the Gaussian examples.
