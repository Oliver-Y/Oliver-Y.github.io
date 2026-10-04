# Checkpoint — "Stats review" blog post (branch `oye_add_stats-notes`)

Last updated: 2026-10-04

**What this is:** a blog post reviewing Murphy, *Probabilistic Machine Learning: An
Introduction*, Ch 2-4 (random variables, distributions, Gaussians, Bayesian inference).
The draft is `src/notes/stats-review.md`. It's written in Oliver's own voice. **Don't write
his prose or fill sections for him.** Help by asking questions, checking his explanations,
and suggesting where things should go.

## Where the source material lives (not in this repo)

- `~/knowledge/Synthesis/probability-and-bayesian-inference.md`: the detailed outline,
  about 500 lines, with `[GAP]` markers. It's the main input for the post. Some sections
  marked `[GAP]` were covered after the markers were added, so the markers overstate what's
  missing.
- `~/knowledge/TIL/2026-06-*`: 7 stats-tagged TILs from Murphy, plus
  `2026-08-25-entropy-cross-entropy-information-theory` (not reviewed yet).
- Oliver also has a private teaching checkpoint on this topic that is local only and
  intentionally not in this repo. If it isn't on this machine, ask him for it rather than
  re-deriving his background.

If `~/knowledge` isn't on this machine, say so and work only from the draft.

## Structure (Oliver's — don't restructure without him)

The outline in the synthesis doc:

```
0. Random variables (PMF/PDF/CDF, conditionals/joint/marginal, Bayes rule)
1. Catalogue of distributions, indexed by SUPPORT
2. Bayes, Bayesian inference, MLE, MAP, uncertainties
   + conjugacy at the end (needs "posterior" first)
   + intractability/MCMC as the closing turn -> motivates Part 3
3. Gaussians + Bayes
   3a multivariate Gaussians / covariance
   3b manipulating Gaussians
   3c Bayesian inference on Gaussians (conditioning)
   3d Kalman filters
```

Decisions already made with Oliver (don't reopen them):
- MLE/MAP sit in Part 2, because epistemic uncertainty depends on them.
- Support -> likelihood goes in Part 1; conjugacy comes after Bayes.
- Intractability is the bridge from Part 2 to Part 3, so Gaussians arrive as "the case where
  the intractable becomes closed-form".
- Kalman is the ending because it's 3c run in a loop.

## Current state of the draft

| Section in draft | State |
|---|---|
| Intro | written |
| 0. Random variables | written: PMF/PDF/CDF, conditionals/joint, Bayes rule |
| 1. Distributions | Bernoulli/Binomial, Categorical/Multinomial, Gaussians |
| 2. Deep dive on Gaussians | multivariate + covariance written; manipulation is a heading only; conditioning is rough notes ("kinda hard to reason through") |
| 3. Bayes / MLE / MAP / posterior predictive | rough bullets only |
| Kalman | not started |
| Conjugacy, intractability/MCMC, aleatoric vs epistemic | not in draft yet (covered in synthesis doc) |

**The draft's order doesn't match the outline:** the draft puts Gaussians before
Bayes/MLE/MAP, while the outline puts Bayes before Gaussians. Ask Oliver which order he
wants; don't pick one yourself.

## Known issues

- **Fonts are broken on this branch.** The KaTeX change in `eleventy.config.js` replaced
  `addPassthroughCopy("src/fonts")` instead of adding to it. `src/css/style.css` still loads
  `/fonts/Inter-*.woff2`, so self-hosted Inter won't be copied to `_site/`. Fix: add the
  `src/fonts` passthrough back next to the KaTeX ones. Check before merging.
- **Posterior predictive wording in section 3:** "how well does this one theta explain
  y|x". At prediction time there is no y to explain. The posterior predictive is a weighted
  vote: each theta's *prediction* p(y|x,theta), weighted by p(theta|D). Raise this with
  Oliver; don't rewrite it.
- Merging to `main` publishes the post right away (there's no draft flag).

## Open questions

- Does entropy / cross-entropy / KL (the 2026-08-25 TIL) go in this post or a separate
  one? For: KL closes Part 2 (MLE = minimizing KL; VI minimizes a KL). Against: it's the
  only TIL not from Murphy.
- One long post or a series? Four parts is a lot.

## Next steps

1. Fix the fonts passthrough.
2. Settle the section order (see above).
3. Bring the draft up to date with the synthesis doc, section by section, with Oliver. The
   biggest gaps are 3b manipulation, 3c conditioning, Part 2 (MLE/MAP/uncertainty/
   conjugacy/MCMC), and 3d Kalman.
4. Conditioning (3c) is the section Oliver flagged as hardest. It blocks both the
   Gaussian-Bayes section and Kalman. A good way in: the tilted-ellipse picture, and what
   happens when the ellipse is flat (the measurement tells you nothing).
5. Decide on the entropy TIL.
