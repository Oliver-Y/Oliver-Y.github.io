# Checkpoint: "Stats review" blog post (branch `oye_add_stats-notes`)

Last updated: 2026-10-06 (end of session; section 3 + 4.6 done, precision-reviewed, pushed). Read the 2026-10-06 update first; the 2026-10-05 addendum below is partly superseded.

## 2026-10-06 update (supersedes the section 3 status below)

**Done:**
- Section 3 is complete through 4.6 in `src/notes/stats-review.md`. Oliver worked through 4.6 himself; I added: the Bayes-rule posterior formula, the point-estimate vs posterior-predictive equation comparison (MAP / posterior mean / integral, delta-spike view), the Gaussian-Gaussian boxed lambda_N/m_N + predictive N(m_N, 1/kappa + 1/lambda_N), the "distribution over theta, maintained sequentially" bullet, the posterior-variance bullet (se = s/sqrt(N)), and two paragraphs at the top of section 3 (aleatoric vs epistemic; point estimates collapse theta before predicting).
- Ran the oracle (Fable) as a precision reviewer twice (full pass, then re-verify); applied all its findings: A1-A4 (regularizer is -log p(theta), weight-decay prose, conjugacy is a prior-likelihood property, "explain y|x" bullet), B1-B12, C1-C12, plus 12 more from the second pass (sections 0-2 too). Formulas were all verified correct; fixes were prose.
- Committed + pushed: `7473ed5` on `oye_add_stats-notes` (origin is up to date). Builds with 0 katex-error (262 formulas). Dev server still on :8080.

**Still open:**
- Oliver's own placeholders: inverse-Wishart prior (MAP Ex #2), method of moments "(Work through this)", "Review this again, it's kinda hard to reason through", "Similar to regression?", sensor-fusion "conditioning 2-fold" wording. Section 2 Ex #2 certainty rephrase and the optional sections-2 flags below are unaddressed.
- Optional oracle note not applied: weight decay == L2 only for plain SGD (AdamW differs).
- This checkpoint file is uncommitted on purpose (`.claude/` must not reach `main`; still undecided whether to drop it before merging or purge it from history).
- Not checked in a real browser at phone width. Fonts passthrough (`src/fonts`) still missing from `eleventy.config.js`; `katex@^0.18.9` in `package.json` is unused.
- Section order question (Gaussians before Bayes) and entropy/KL-in-this-post question unanswered. Kalman (3d) not started. `~/knowledge/Synthesis/probability-and-bayesian-inference.md` is behind the draft.

**Next:** Oliver reviews the post himself; then browser/phone check, fonts passthrough fix, decide `.claude/` handling, then merge only when he says (merge to `main` publishes).

## Section 3 addendum (2026-10-05)

**Status:** `## 3.` is drafted down to the `(4.6)` heading; the 4.6 part is **Oliver's to work
through next** (he said "going to work through that right now"). It builds, 0 `katex-error`
(184 formulas). Still not checked in a real browser.

**Structure now in `src/notes/stats-review.md`:**
- Intro: one line, "So far we've been given theta. Now we have to find it."
- `### Point estimate methods` (no intro sentence; see "Removed" below)
  - `#### MLE`: definition, steps 1-2 (product, NLL), "Why MLE is acceptable" (uniform prior;
    KL(p_emp||p_theta) = -H(p_emp) + NLL/N), Worked examples (Bernoulli, categorical,
    Gaussian incl. multivariate, results only, no derivations), linear regression
    (RSS/MSE/RMSE), ERM, method of moments.
  - `#### Regularization and MAP`: overfitting diagnosis, C(theta)/lambda, MAP, the loss
    formula, Ex #2 MAP Gaussian, Ex #3 weight decay (zero-mean Gaussian prior; I added the
    -log N(theta|0,s0^2 I) formula), Ex #4 early stopping.
  - `#### Choosing the regularization strength`: grid search over lambda on a validation
    set, retrain; more data -> less overfit.
- `### Bayesian inference and the posterior predictive (4.6)`: **the old rough bullets are
  still under it** (posterior predictive integral, the "how well does this theta explain
  y|x" bullet, Gaussian-Gaussian example). The heading wording is mine, not the book's.

**Placeholders Oliver left for himself ("work through this"):** inverse-Wishart prior
(Ex #2 MAP Gaussian) and method of moments.

**Removed at Oliver's request, don't re-add unprompted:** the "Basic premise" paragraph and
the training-vs-prediction bullets; the "Ch 4 review ... MLE+MAP populate the structures"
paragraph; the conjugate-vs-NN note; the "point estimates because argmax reduces to one
theta-hat" sentence (he wants that said **later, in 4.6, as the contrast with NOT taking a
point estimate**). He trims redundancy and outdated text himself; he wants worked examples
as results only.

**Content flags raised, not yet resolved (Oliver decides):**
- `lambda C(theta) = p(theta)` appears in the "Regularization to prevent overfitting" formula
  and in the new MAP bullet. It should be `-log p(theta)` (the loss is minimized).
  Also "lambda = regularization" should be "regularization strength".
- Posterior predictive bullet "how well does this theta explain y|x": there is no y at
  prediction time. Each theta *predicts* y and votes, weighted by p(theta|D). Also
  "fixed theta" holds for MLE/MAP only. (This is his recurring "explain vs predict" gap.)
- MAP is the *mode* of the posterior; MLE Gaussian covariance with 1/N is the biased
  estimator; MAP Gaussian `lambda S0 + (1-lambda) S_mle` is shrinkage and the prior behind
  it is inverse-Wishart (not yet named in prose).
- Early stopping is an implicit regularizer, not a term added to the cost function.
- Conjugacy is a property of the prior-likelihood pair, not of the likelihood alone.
- Method of moments "solving NLL'(theta)=0 is hard" is true only for some models (mixtures),
  not the closed-form examples above.
- Note-to-self answered in chat, not in the draft: RSS/MSE/RMSE share the argmin (MSE
  removes dependence on N, RMSE restores units of y); they still work if sigma is an unknown
  *constant* (sigma drops out of the w-gradient; sigma^2_mle = MSE); they break for
  per-sample variance (heteroscedastic -> weighted least squares).
- Duplicate-ish: the Gaussian MLE appears in the worked examples and again under MAP
  Example #2 context; fine unless he wants it merged.

**Oliver's understanding (from this session):** he asked for a rebuild-from-scratch
approach for section 3 because he doesn't have section-2-level understanding of it. Plan I
proposed: MLE (coin, 3 heads in 3 flips -> MLE=1, predicts 4th flip certain heads) ->
MAP as the fix -> posterior predictive (predict, don't explain) -> Gaussian-Gaussian. **He
never answered the coin question**; he instead pasted notes and wrote the draft. Corrections
he got on his own segue (all still to be re-checked): (1) one Gaussian only works in the
conjugate case; (2) the loss is NOT a separate function, NLL = -log p(D|theta), the
optimizer is the separate thing; (3) MLE/MAP don't sample, they optimize to a point;
sampling is MCMC. The 4.6 section is where the "explain vs predict" gap is most likely to
resurface.

**Process notes:** `.claude/checkpoints/` is **tracked in git on this public repo** and Oliver
said `.claude/` must not reach `main`; keep this file free of private/work details.
Nothing from this session is committed (the section-2 commit `b9bc0d3` is also still
unpushed). Ask before committing or pushing.

**Next steps (section 3):** (1) Oliver writes 4.6 and puts the point-estimate contrast
sentence there; (2) resolve the flags above in a final pass; (3) fill the two "work through
this" placeholders; (4) bring the coin / explain-vs-predict drill back if he wants the
understanding check; (5) then Kalman (3d in the outline) and the older section-2 flags
below.

---

**What this is:** a blog post reviewing Murphy, *Probabilistic Machine Learning: An
Introduction*, Ch 2-4. The draft is `src/notes/stats-review.md`. It's written in Oliver's
own voice.

**How Oliver wants to work on it** (confirmed this session):
- He pastes his own text, and you place it, fix formatting/notation, and **flag** content
  errors instead of rewriting them. Only reword when he asks. **Put pasted text in the
  draft even if he says he's still working through it**; he refines in place.
- If he asks for a note, keep it short and close to his phrasing. He'll cut extra
  explanation (twice this session).
- Retype formulas as KaTeX; don't embed book screenshots. Screenshots he gives as sources
  are in `~/artifacts/`. Box key formulas with `\boxed{\begin{aligned}...\end{aligned}}`.
- Ask before committing or pushing.

## 1. What was done (2026-10-04)

- Rewrote **Multi-dimensional Gaussians** (section "2. Deep dive on Gaussians"):
  - `#### Covariance`: covariance definition; the "why multiply?" derivation with
    X = S + Nx, Y = S + Ny (mean 0), with V[X] and Cov[X,Y] expanded side by side; the
    rule "a term survives only when multiplied by itself"; ρ = Cov/(σX σY); the covariance
    matrix Σ; an inline SVG of three point clouds (ρ=0.9, ρ=0, and the first cloud with X
    stretched ×2: same ρ, covariance doubles, slope halves); an "Overall" line.
  - `#### Conditioning on Gaussians`: the boxed KaTeX formula for p(y1|y2); the mean and
    variance intuition, reworded around the "shared portion"
    (Σ12·Σ22^-1 = V[S]/(V[S]+V[N2])); notes on why units/scale cancel, for the mean and
    for the variance; an "Overall" line.
  - `#### Worked examples`: #0 2-D conditioning (Murphy 3.31/3.32); #1 hidden-feature
    imputation (with the h|v block formula); #2 Bayes updating with noisy measurements
    (3.52–3.54, the N=1 forms, the three ways to write μ1, SNR, and the multivariate
    p(D|z) ∝ N(ȳ | z, Σy/N)); #3 sensor fusion (3.69, precision-weighted).
  - The "Manipulating Gaussians" heading was removed at Oliver's request. **Section 2 is
    done**; only the small content flags below are left.
- **Fixed a KaTeX CSS version mismatch** (`eleventy.config.js`). The HTML is rendered by
  katex 0.16.47 (bundled inside `@vscode/markdown-it-katex`), but the CSS came from the
  top-level katex 0.18.9, which renamed its layout classes. That broke math layout (a line
  through the boxed formula). The CSS and fonts are now copied from whichever katex the
  plugin resolves.
- Repo hygiene: rewrote history to replace a work email with the `Oliver-Y` noreply
  identity on `main` and this branch (force-pushed). The dependabot branches that still had
  the old commits are deleted.

## 2. Current state

- Branch `oye_add_stats-notes`: `b9bc0d3` is committed but **not pushed** (1 ahead of
  origin). Everything after it is **uncommitted** in `src/notes/stats-review.md`: the
  shared-portion trims and notes, the imputation sentence, the multivariate p(D|z), Example
  #3, and the removed heading. This checkpoint file is uncommitted too.
- **Next up: section 3 (Bayes, Bayesian inference, MLE, MAP, posterior predictive).**
  Right now it's rough bullets. Known issue there: the posterior predictive bullet says
  "how well does this one theta explain y|x". At prediction time there's no y; each θ
  *predicts* y and gets a vote weighted by p(θ|D). The richest source is Part 2 of
  `~/knowledge/Synthesis/probability-and-bayesian-inference.md`: roles vs families, why
  p(D) exists, MLE/MAP as a 2×2, aleatoric vs epistemic, conjugacy, intractability/MCMC.
- It builds, and all math renders with 0 `katex-error`. It hasn't been checked in a real
  browser at phone width.
- Dev server: `eleventy --serve --port 8080`, already running since 2026-09-24, using Node
  v22 from nvm. **The default shell `node` is too old for Eleventy 3**; use
  `PATH=~/.nvm/versions/node/v22.16.0/bin:$PATH npx @11ty/eleventy`. When you add a new
  passthrough directory, the running server won't see it; run a one-off build.

## 3. Open questions

- **`.claude/` must not reach `main`** (Oliver: "this should not be checked in"). Still
  undecided: keep it on this branch and drop it before merging, or remove it from the
  repo/history entirely.
- Section order: the draft has Gaussians (2) before Bayes/MLE/MAP (3); the outline in
  `~/knowledge/Synthesis/probability-and-bayesian-inference.md` has Bayes first. Raised
  again just before starting section 3, and not answered. Note that the section 2 examples
  are already Bayesian updating, so "Bayes second, as the general idea" also works.
- Does entropy/KL (TIL 2026-08-25) go in this post? One post or a series?

## 4. Next steps

1. Commit the uncommitted edits; push only after Oliver confirms.
2. **Fonts bug:** add `eleventyConfig.addPassthroughCopy("src/fonts")` back. `style.css`
   loads `/fonts/Inter-*.woff2`; the dev server only works because `_site/fonts` is left
   over from an old build.
3. Optionally remove the now-unused `katex@^0.18.9` from `package.json`.
4. Content flags in section 2 raised but not resolved yet (Oliver decides; can wait for a
   final pass):
   - Multivariate p(D|z) line says it "approximates loosely to the data mean". It's
     exact: N observations equal one observation of ȳ with covariance Σy/N, so ȳ is a
     sufficient statistic.
   - Ex #2: "variance is dominated by data variance". Posterior variance is 1/(λ0+Nλy),
     which shrinks about like Σy/N. Oliver's own rephrase: "more trials make me more
     certain regardless; the certainty gained per trial is set by how certain each trial
     is". Not yet in the draft.
   - Ex #2 shrinkage aside "(I guess it means shrinking the data maybe?)". Answer:
     shrinkage means the estimate is pulled from y toward μ0.
   - The sensor-fusion line ("conditioning 2-fold") is unclear.
   - "Review this again, it's kinda hard to reason through" and "Similar to regression?"
     would be published as written.
   - Optional note: Σvv^-1 also stops double-counting redundant visible features.
   - Max-entropy bullet in the Gaussians section ("variance is a good way to retain high
     entropy"): variance is the *constraint*, and the Gaussian is the max-entropy
     distribution given it.
   - A possible bridge between Ex #0 and Ex #2: for y = z + ε,
     ρ² = Σ0/(Σ0+Σy), so the prior-vs-noise comparison is ρ² in disguise.
5. Fill "Manipulating Gaussians" (linear transform, sum, sign flip), then Part 3
   (Bayes/MLE/MAP) and Kalman.
6. Update `~/knowledge/Synthesis/probability-and-bayesian-inference.md` with this
   session's material (it's behind the draft).

## 5. Key context

- Repo is **public**, and Pages deploys only on push to `main`. There's no draft flag:
  merging publishes the post.
- Corrections that are already in the draft and shouldn't be reopened: covariance has
  units and correlation doesn't; correlation ≠ slope (slope = Cov/Var(input)); the
  variance reduction is ρ²σ1², so σ2's scale cancels and only shared vs unshared variation
  matters; ρ² = (shared fraction of X)(shared fraction of Y) in the S+N model.
- Figure generator: `covfig.py` (not in repo). It used 160 points, seed 7, whitened so
  the sample statistics are exact. Regenerate if the panels need changing.
- Source material lives outside the repo: `~/knowledge/Synthesis/...`,
  `~/knowledge/TIL/2026-06-*`, and `~/artifacts/` screenshots. There's also a private
  teaching checkpoint (local only, intentionally not in this repo).
