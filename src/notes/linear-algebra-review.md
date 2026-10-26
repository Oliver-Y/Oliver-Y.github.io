---
title: Linear algebra review
date: 2026-10-17
blurb: Notes on what it actually means for something to be linear, and an intuitive picture of vector spaces.
---

## 0. Notation

- A vector $x \in \mathbb{R}^n$ is a list of $n$ numbers.
- A matrix $A \in \mathbb{R}^{m \times n}$ has $m$ rows and $n$ columns. Here
  it's a row of $n$ column vectors, each living in $\mathbb{R}^m$.
- $A^T$ flips rows and columns: the rows of $A$ become the columns of $A^T$. A
  few identities:
  - $(A^T)^T = A$
  - $(AB)^T = B^T A^T$
  - $(A + B)^T = A^T + B^T$
- **Order** (of a tensor): the number of dimensions of the array (vector = 1,
  matrix = 2).

A **vector space** (also called a linear space) is a set of points where stretch
and add are defined and keep you inside the set. Everything below assumes that
addition and scaling exist.

- Note: a set where this fails is the unit circle in $\mathbb{R}^2$. Adding
  $(1, 0) + (0, 1) = (1, 1)$ lands off the circle (its length is
  $\sqrt{2}$), and scaling by $0$ gives the origin, which isn't on the circle
  either. So the circle isn't a vector space, even though it sits inside one.

### Linear independence, span, basis

- **Linear independence**: a set $\{x_1, \dots, x_n\}$ is linearly independent
  if no $x_i$ can be written as a linear combination of the others.
  - Note: another useful way to think about it is that the only way to get
    $c_1 x_1 + \dots + c_n x_n = 0$ is with every $c_i = 0$.
- **Span**: all vectors that can be expressed as a linear combination of
  $\{x_1, \dots, x_n\}$:
  $\operatorname{span}(x_1, \dots, x_n) = \{\, c_1 x_1 + \dots + c_n x_n \;:\; c_i \in \mathbb{R} \,\}$.
- **Basis**: a set of linearly independent vectors that spans the whole space.
  Any $n$ linearly independent vectors in $\mathbb{R}^n$ form a basis.

With basis vectors $\{x_1, \dots, x_n\}$, any point in a vector space can be expressed as 
stretches and adds: 

$$
x = c_1 x_1 + c_2 x_2 + \dots + c_n x_n
$$

So infinitely many points **compress** down to $n$ basis vectors: each point is
identified by exactly $n$ numbers $(c_1, \dots, c_n)$, its coordinates in that
basis.

- Note: only independent vectors give each point one set of coordinates;
  otherwise there is an infinite set of coordinates a point can be expressed
  in. Specifically, let's say we use 3 vectors in $\mathbb{R}^2$ and $v_3$
  depends on $v_1, v_2$. Then we can shift $t$ units of weight from
  $(c_1, c_2)$ onto $c_3$ and land on the same point, so every point has
  infinitely many coordinates. It's still compression (3 vectors describe the
  whole plane) just not the most efficient one, and each point's coordinates
  are not unique anymore.

{% include "figures/span-basis-3d.html" %}

## 1. Linear maps: what it means to be linear

A linear map is a relationship between two linear spaces that preserves the
stretch and add. As a result, the outputs it actually hits form a linear space
too (a subspace of the output space).

- **Homogeneity**: $f(a v) = a f(v)$
- **Additivity**: $f(v + w) = f(v) + f(w)$

$f(a v + b w) = a f(v) + b f(w)$.

{% include "figures/stretch-add.html" %}

If a point in a vector space is just a linear combination of basis vectors built
only from stretches and adds, if a function $f$ preserves both, then mapping where
the $n$ basis vectors land tells you where every point in the output space lands:

$$
\begin{aligned}
f(x) &= f(x_1 e_1 + \dots + x_n e_n) \\
     &= f(x_1 e_1) + \dots + f(x_n e_n) \qquad \text{(additivity)} \\
     &= x_1 f(e_1) + \dots + x_n f(e_n) \qquad \text{(homogeneity)}
\end{aligned}
$$

There exists a similar intuition of compression here. Instead of needing to map
where every single point lies, we only need to keep track of how the basis
vectors change.

Given a vector space and a linear map, we can get another vector space in the
output that has a very specific relationship to our input vector space.
Specifically, we just need to see where the landed vectors $f(e_i)$ go to
determine things about the output vector space. This could be stuff like how it
scales or where coordinates go, or also if it loses a rank if, let's say,
$f(e_1) = f(e_2)$ after the transform.

### Counterexamples

Drop either rule and the compression breaks. If we treat $f$ as a black box, then
we can check linearity by doing the operation then mapping, against mapping
first then the operation. Since we need to check over the entire input space,
the grid lines give a better geometric picture of whether or not this is a
linear map.

- **Additivity** keeps gridlines parallel and evenly spaced.
- **Homogeneity** keeps lines evenly spaced.

**Homogeneous, not additive.** $f(v) = (1 + \cos^2\theta)\, v$: stretch each
point by an amount that depends only on its direction.

{% set ltMaps = "homog" %}{% set ltTest = "add" %}{% set ltTests = "add,hom" %}
{% include "figures/linearity-tests.html" %}

**Neither (bend).** $f(x, y) = (x, y + x^2/4)$ curves the grid: doubling $v$
quadruples the bend instead of doubling it.

{% set ltMaps = "bend" %}{% set ltTest = "hom" %}{% set ltTests = "add,hom" %}
{% include "figures/linearity-tests.html" %}

**Affine.** $f(x) = x + 1$ keeps the grid straight, parallel and evenly
spaced but moves the origin, so $f(2) = 3 \neq 2 f(1)$. Straight is not the
same as linear.

{% set ltMaps = "shift" %}{% set ltTest = "add" %}{% set ltTests = "add,hom" %}
{% include "figures/linearity-tests.html" %}

### Why linear

Things are locally linear: zoom in on any smooth map and it looks linear (its
derivative is a matrix). So we choose to model systems linearly, and it
composes: chaining linear maps gives another linear map (multiply the
matrices), so a whole stack compresses to one matrix.

### Determinant

The determinant is how much each grid cell's area (or volume) gets scaled by
the map. If $\det A = 0$, the area collapses to zero: the map squishes space
down into a lower dimension, and some direction is lost.

## 2. The four subspaces

What we've done above is really examine what it means for something to be
linear, and it basically imposes this set of rules saying that there are
additive and stretching laws that are preserved. Linear maps enable us to go
from an input space to an output space and draw relationships through them.
Linear maps are usually represented by matrices. So, given a matrix that
basically defines a linear map, how do we break it down into meaningful
subspaces to interpret it? The linear map is the function expressed as $A$, and
then $x$ is a set of knobs, which is basically the inputs we're feeding into
the linear map.

### Notation

For $A \in \mathbb{R}^{m \times n}$:

- **Dimension** (of a space): how many independent directions it has, i.e. the
  number of vectors in a basis for it.
- **Rank** $r$: the number of independent columns of $A$. This always equals
  the number of independent rows, even though the columns live in
  $\mathbb{R}^m$ and the rows in $\mathbb{R}^n$.

| Subspace | Symbol | Lives in | Dimension |
|---|---|---|---|
| Column space | $C(A)$ | $\mathbb{R}^m$ (output) | $r$ |
| Null space | $N(A)$ | $\mathbb{R}^n$ (input) | $n - r$ |
| Row space | $C(A^T)$ | $\mathbb{R}^n$ (input) | $r$ |
| Left null space | $N(A^T)$ | $\mathbb{R}^m$ (output) | $m - r$ |

### Through knobs

I like to think of $Ax$ where $x$ is a set of knobs, one per column of $A$.

- **Column space**: every output $A$ can produce. Each knob $x_i$ scales column
  $i$, so the column space is the span of the columns, and its dimension is the
  number of independent columns (the rank).
- **Null space**: every input that maps to $0$. These are knob settings that
  cancel each other out and don't change the output. It always contains $0$, and
  it's nontrivial whenever the columns are dependent (rank $< n$).
- **Row space**: the span of the rows. Each row is one constraint on the input,
  so the row space is the set of input directions that actually change the
  output.

Row space and null space both live in the input space and split it cleanly: they
are perpendicular, and their dimensions add up to $n$. Any input breaks into a
row-space part, which determines the output, and a null-space part, which gets
thrown away. Restricted to the row space, $A$ maps 1:1 onto the column space.

### Example

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \qquad
A \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} y_1 \\ y_2 \end{bmatrix}
$$

- The two rows are independent, so the rank is 2.
- **Row space**: the plane in $\mathbb{R}^3$ spanned by $[1, 2, 3]$ and
  $[4, 5, 6]$.
- **Null space**: $3 - 2 = 1$ dimension, the line through $[1, -2, 1]$
  (check: $1 - 4 + 3 = 0$ and $4 - 10 + 6 = 0$). This line is perpendicular to
  the row-space plane.
- **Column space**: rank 2 in $\mathbb{R}^2$, so all of $\mathbb{R}^2$. Every
  output is reachable.

So the 3D input space is the plane times the line. Any input on the plane maps
1:1 to a point in $\mathbb{R}^2$, and sliding an input along the null-space
direction doesn't change its output at all.

## 3. Geometry: inner products and projections

*TODO*

## 4. Eigenvalues and eigenvectors

*TODO*

## 5. SVD

*TODO*
