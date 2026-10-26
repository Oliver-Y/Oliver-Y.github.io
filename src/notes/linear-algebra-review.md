---
title: Linear algebra review
date: 2026-10-17
blurb: Notes on what it actually means for something to be linear, and an intuitive picture of vector spaces.
---

## 0. Notation

- A vector $x \in \mathbb{R}^n$ is a list of $n$ numbers.
- A matrix $A \in \mathbb{R}^{m \times n}$ has $m$ rows and $n$ columns. It can be
  viewed as a stack of row vectors or a row of column vectors.
- $A^T$ flips rows and columns. A few identities:
  - $(A^T)^T = A$
  - $(AB)^T = B^T A^T$
  - $(A + B)^T = A^T + B^T$
- **Order** (of a tensor): the number of dimensions of the array (vector = 1,
  matrix = 2).
- **Rank** (of a matrix): the number of linearly independent columns.

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

### Vector space

With basis vectors $\{x_1, \dots, x_n\}$, any point in a vector space can be expressed as 
stretches and adds: 

$$
x = c_1 x_1 + c_2 x_2 + \dots + c_n x_n
$$

So infinitely many points compress down to $n$ basis vectors: each point is
identified by exactly $n$ numbers $(c_1, \dots, c_n)$, its coordinates in that
basis.

- Note: "exactly" needs the vectors to be independent. Say we use 3 vectors in
  $\mathbb{R}^2$ and $v_3$ depends on $v_1, v_2$. Then we can shift $t$ units
  of weight from $(c_1, c_2)$ onto $c_3$ and land on the same point, so
  every point has infinitely many coordinates. It's still compression (3
  vectors describe the whole plane) just not the most efficient one, and each point is not unique anymore. 

{% include "figures/span-basis-3d.html" %}

### Matrix operations and properties

*TODO*

## 1. Linear maps: what it means to be linear

A linear map is a relationship between two spaces that preserves the stretch and add.

- **Homogeneity**: $f(a v) = a f(v)$
- **Additivity**: $f(v + w) = f(v) + f(w)$

$f(a v + b w) = a f(v) + b f(w)$.

{% include "figures/stretch-add.html" %}

### What it buys: the map compresses to a matrix

A point in a vector space is a linear combination of basis vectors, built only
from stretches and adds. If $f$ preserves both, then where the $n$ basis vectors
land tells you where every point lands:

$$
\begin{aligned}
f(x) &= f(x_1 e_1 + \dots + x_n e_n) \\
     &= f(x_1 e_1) + \dots + f(x_n e_n) \qquad \text{(additivity)} \\
     &= x_1 f(e_1) + \dots + x_n f(e_n) \qquad \text{(homogeneity)}
\end{aligned}
$$

Same coordinates $x_i$, applied to the landed vectors $f(e_i)$ instead of the
$e_i$. A matrix is just those landed vectors written side by side as columns.

Concretely, take $f: \mathbb{R}^2 \to \mathbb{R}^3$ with
$f(e_1) = (1, 0, 2)$ and $f(e_2) = (3, 1, -1)$:

$$
A = \begin{bmatrix} | & | \\ f(e_1) & f(e_2) \\ | & | \end{bmatrix}
  = \begin{bmatrix} 1 & 3 \\ 0 & 1 \\ 2 & -1 \end{bmatrix}
$$

- $n = 2$ columns: one per input basis vector.
- $m = 3$ rows: each landed vector lives in the output space $\mathbb{R}^3$.
- So $m \times n = 6$ numbers describe $f$ on all of $\mathbb{R}^2$.

To map $x = (2, 1)$, reuse its coordinates on the columns:

$$
A x = 2 \begin{bmatrix} 1 \\ 0 \\ 2 \end{bmatrix}
    + 1 \begin{bmatrix} 3 \\ 1 \\ -1 \end{bmatrix}
    = \begin{bmatrix} 5 \\ 1 \\ 3 \end{bmatrix}
$$

Compression is not the same as 1:1, though. If the landed vectors $f(e_i)$ are
dependent (say, projecting $\mathbb{R}^3$ onto a plane), many inputs share an
output. Whether a map is 1:1 is a rank question, which is what Section 2 is
about.

### Counterexamples

Drop either rule and the compression breaks.

- **Homogeneous, not additive**: $f(x, y) = \dfrac{x^3}{x^2 + y^2}$ (with
  $f(0, 0) = 0$). Stretching works, but $f(1, 0) = 1$ and $f(0, 1) = 0$ while
  $f(1, 1) = \tfrac{1}{2} \neq 1 + 0$. Each line through the origin maps
  cleanly on its own, but the basis vectors no longer tell you the points in
  between. (The "twist" below is its 2D version.)
- **Additive, not homogeneous**: barely possible. Additivity alone forces
  $f(\tfrac{p}{q} v) = \tfrac{p}{q} f(v)$ for every rational $\tfrac{p}{q}$, so
  only irrational stretches are left free, and a map that uses that freedom
  can't be continuous (building one needs the axiom of choice). Additivity plus
  continuity already gives linearity.
- **Neither**: $f(x) = x^2$ curves the grid. $f(x) = x + 1$ keeps it straight,
  parallel and evenly spaced but moves the origin, so $f(2) = 3 \neq 2 f(1)$.
  Straight is not the same as linear.

{% set ltMaps = "linear,homog,bend,shift" %}{% set ltTest = "add" %}{% set ltTests = "add,hom" %}{% set ltCaption = "Drag v and w on the left, and switch maps and tests." %}
{% include "figures/linearity-tests.html" %}

### Why linear, and not something richer

Richer families of maps can describe more, at the cost of more numbers. For
$\mathbb{R}^2 \to \mathbb{R}^2$:

| Family | Numbers to store | Compose two of them |
|---|---|---|
| Linear, $Ax$ | 4 | linear: 4 |
| Affine, $Ax + b$ | 6 | affine: 6 |
| Quadratic | 12 | degree 4: 30 |
| Cubic | 20 | degree 9: 110 |

The last column is what makes linear win. Composing linear maps gives another
linear map (multiply the matrices), so a chain of a thousand of them is still
one $2 \times 2$ matrix. Polynomials don't close: chain three quadratics and you
need degree 8. Affine maps close too, and appending a 1 to every vector turns
them into linear maps, which is why 3D transforms are $4 \times 4$ matrices.

Everything else that makes linear algebra useful falls out of this: $Ax = y$
has a complete answer (Section 2), and any smooth map looks linear up close
(its derivative is a matrix), which is why Newton's method, backprop and the
extended Kalman filter all linearize and then use linear algebra.

### Determinant

The determinant is how much each grid cell's area (or volume) gets scaled by
the map. If $\det A = 0$, the area collapses to zero: the map squishes space
down into a lower dimension, and some direction is lost.

## 2. Vector spaces

Nothing above required vectors to be arrows. The derivative is linear:
$(f + g)' = f' + g'$ and $(af)' = a f'$, and so are integration and expectation.
A **vector space** is any set where linear combinations make sense, and a linear
map is any map that preserves them.

### The four subspaces, through knobs

I like to think of $Ax$ where $x$ is a set of knobs, one per column of $A$.

- **Dimension**: how many independent directions a space (or subspace) has,
  i.e. the number of vectors in a basis for it.
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
