# Theory

N-way Canonical Partial Least Squares (N-CPLS) extracts supervised latent components from
a predictor tensor
$X \in \mathbb{R}^{n \times p_1 \times \cdots \times p_d}$ and a primary-response matrix
$Y_{\mathrm{prim}} \in \mathbb{R}^{n \times q}$. Depending on the structure of
$Y_{\mathrm{prim}}$, the method can be used for regression, discriminant analysis, or a
combination of both. Optional additional responses
$Y_{\mathrm{add}} \in \mathbb{R}^{n \times r}$ may contribute to component extraction,
but only $Y_{\mathrm{prim}}$ is deflated during fitting and predicted afterward.

The method was introduced by Liland et al. (2022). It is designed for settings in which
the predictor space is large relative to the number of samples and predictor variables may
show substantial collinearity, so that coefficient estimation by ordinary regression
methods would be unstable or impossible.

The paper formulates N-CPLS as a multiway extension of canonical PLS. Below, the method
is explained mainly in terms of the mode-1 unfolding of the predictor tensor, because
that two-dimensional representation is easier to visualize and usually easier to follow
than the full multiway notation. The matrix in which each sample tensor is stacked into
one row is

```math
X_{(1)} \in \mathbb{R}^{n \times p},
\qquad
p = \prod_{j=1}^d p_j,
```

If the multilinear branch of the algorithm is enabled (default), the collapsed
predictor-side weight object obtained from the unfolded calculations is not kept as an
independent weight for every unfolded predictor coordinate. Instead, it is approximated by
one weight vector for each predictor mode. For a $d$-way predictor tensor, this means that
mode-specific vectors
$w^{(1)} \in \mathbb{R}^{p_1}, \ldots, w^{(d)} \in \mathbb{R}^{p_d}$ are estimated, and
the weight assigned to predictor coordinate $(j_1, \ldots, j_d)$ is then reconstructed
from their outer product:

```math
W_{j_1,\ldots,j_d} \approx w^{(1)}_{j_1} w^{(2)}_{j_2} \cdots w^{(d)}_{j_d},
```

up to an overall scaling factor. Thus, the multilinear branch replaces a separate
coefficient for every possible predictor coordinate by a structured rank-1 approximation
assembled from mode-wise weights. If the unfolded branch is used instead, that
compression step is skipped and the coordinate-wise weight object is kept directly.

## Main Algorithmic Steps

Each component is built in four steps: inference of response-specific predictor
directions, selection of the optimal linear combination of these directions by CCA,
optional compression of the resulting predictor-side weight object to a multilinear
rank-1 form, and extraction of an orthogonal score vector. In the description below,
$X$ denotes the preprocessed predictor tensor and $Y$ denotes the current deflated
working copy of the preprocessed primary-response matrix. At the start of fitting,
$Y = Y_{\mathrm{prim}}$. Under the default settings, this means centered $X$ and centered
$Y_{\mathrm{prim}}$, while $Y_{\mathrm{add}}$, if present, is used as supplied.

### 1. First supervised compression

The first step constructs one candidate predictor direction for each response column in
the combined response matrix
$Y_{\mathrm{comb}} = Y$ or $Y_{\mathrm{comb}} = [Y \; Y_{\mathrm{add}}]$, depending on
whether additional responses are present. These candidate directions are collected in the
matrix of candidate loading weights
$W_{0,(1)}$, obtained by multiplying the transposed unfolded predictor matrix with
$Y_{\mathrm{comb}}$:

```math
W_{0,(1)} = X_{(1)}^\top Y_{\mathrm{comb}}.
```

The resulting matrix $W_{0,(1)}$ has one row for every column of the unfolded predictor
matrix $X_{(1)}$ and one column for every column of $Y_{\mathrm{comb}}$. Its
$(j, k)$ entry is

```math
w_{0,jk} = \sum_{i=1}^n x_{ij} y_{\mathrm{comb},ik},
```

or the corresponding weighted sum if observation weights are used. Under the default
centering of $X$ and $Y_{\mathrm{prim}}$, these entries are proportional to
cross-covariances. More generally, they summarize how predictor coordinate $j$ and
response column $k$ vary together across the samples:

- large positive values mean that samples with large positive
  $y_{\mathrm{comb},ik}$ tend also to have
  large positive predictor values at coordinate $j$,
- large negative values mean the relation tends to go in the opposite direction,
- values near zero mean that this predictor coordinate contributes little to that
  response-specific direction.

The $k$th column of $W_{0,(1)}$ therefore contains one coefficient for every column of
$X_{(1)}$ and can be read as a response-specific direction in unfolded predictor space.
Projecting the unfolded predictor matrix onto these candidate directions gives the
candidate scores

```math
Z_0 = X_{(1)} W_{0,(1)}.
```

The resulting candidate score matrix $Z_0$ has one row per sample and one column per
response column in $Y_{\mathrm{comb}}$. Entry $(i, k)$ is the score of sample $i$ on the
candidate predictor direction associated with response column $k$. $Z_0$ can thus beee seen 
as a compressed, response-guided representation of $X_{(1)}$ in which for each sample and 
each response column, all unfolded predictor coordinates are summarized by a single value.

### 2. Canonical combination

In the second step, canonical correlation analysis (CCA) is applied to $Z_0$ and the
current primary-response matrix $Y$. This yields a dominant canonical weight vector
$c$ on the $Z_0$ side. The vector $c$ tells us how to combine the columns of $Z_0$ so
that the resulting linear combination is maximally correlated with a corresponding linear
combination of the columns of $Y$.

```math
c
=
\text{first left canonical weight vector from }
\mathrm{CCA}(Z_0, Y),
\qquad
```

The weight vector $c$ stores one coefficient for every column of $Z_0$. These coefficients 
can be combined with the candidate directions into one single predictor direction:

```math
W_{(1)} = W_{0,(1)} c.
```

So each predictor coordinate receives a single combined weight obtained by adding its
response-specific candidate weights, weighted by the CCA coefficients in $c$.

Thus, combining the candidate score columns by $c$ is equivalent to projecting
$X_{(1)}$ onto one combined predictor direction $W_{(1)}$.

### 3. Optional multilinear compression

Up to this point the algorithm has produced one unconstrained predictor-side weight
object $W$. If the predictors are unfolded into one long variable axis, then $W$ has

```math
\prod_{j=1}^d p_j
```

free weights. The multilinear branch replaces this by a much smaller set of mode-specific
vectors with only

```math
\sum_{j=1}^d p_j
```

free weights.

If `multilinear = false`, the package uses $W$ directly. This is the unfolded branch, and
it is equivalent to stopping after the CPLS-style calculation and keeping the full
predictor-side direction.

If `multilinear = true`, the package refolds $W$ to predictor shape and approximates it by
a rank-1 multilinear object

```math
W^\circ = w^{(1)} \circ w^{(2)} \circ \cdots \circ w^{(d)}.
```

The exact approximation depends on the number of predictor modes: if $d = 1$, $W$ is
just normalized; if $d = 2$, the leading rank-1 SVD approximation is used; and if
$d \ge 3$, a one-component PARAFAC model is fitted.

After this factorization, the package optionally orthogonalizes the mode vectors on
previous mode vectors, normalizes them, and then recombines them into the outer-product
tensor $W^\circ$.

Two points are important here.

First, the extracted mode vectors are not the objects used directly for the score
calculation. They are recombined into the single predictor-side weight tensor $W^\circ$,
and the score vector is computed from that combined tensor. In other words, the
multilinear factors are an interpretable parameterization of the component, not separate
component scores.

Second, this rank-1 restriction is both the main advantage and the main limitation of the
multilinear branch. It gives one loading vector per mode, which is attractive for
interpretation and often acts as useful regularization. But it also means that only
separable predictor-side directions can be represented exactly. If the truly predictive
direction is not well approximated by an outer product, the unfolded branch can be more
flexible.

### 4. Score, loadings, and deflation

The actual component score is obtained by projecting each sample onto the final
predictor-side weight object. In the multilinear branch this object is $W^\circ$; in the
unfolded branch it is simply $W$. Using $W^\circ$ to denote the final object passed to the
score calculation, the projection is, in unfolded notation, just a matrix-vector product:

```math
t_{\mathrm{raw}} = X_{(1)} \operatorname{vec}(W^\circ).
```

The package then orthogonalizes `t_raw` on earlier score vectors and normalizes it to
unit length. With `T_{1:a-1}` denoting the previously extracted scores,

```math
t
=
t_{\mathrm{raw}} - T_{1:a-1}(T_{1:a-1}^\top t_{\mathrm{raw}}),
\qquad
t = \frac{t}{\|t\|}.
```

Because `t` is normalized, the loading calculations in the code do not need an explicit
denominator `t^\top t`:

```math
P = X_{(1)}^\top t,
\qquad
q = Y^\top t,
\qquad
Y \leftarrow Y - t q^\top.
```

Only `Y` is deflated. `X` stays fixed throughout fitting. This is one of the main
computational differences from classical N-PLS.

### 5. After all components

After the component loop, the package forms score-projection tensors `R` and cumulative
regression coefficients `B`. In unfolded notation the main relation is

```math
R = W_A (P_A^\top W_A)^{-1},
\qquad
B = \operatorname{cumsum}(R \odot Q^\top).
```

Here `W_A`, `P_A`, and `Q` denote the component-wise stacks of predictor weights,
predictor loadings, and response loadings. Predictions are obtained from the preprocessed
predictors and the stored cumulative coefficient array, and the mean of $Y_{\mathrm{prim}}$ is added
back afterwards.

## Orthogonalization Explained

The paper mainly discusses two orthogonalization ideas. The package contains three
orthogonalization-related operations in total, and separating them makes the algorithm
much easier to understand.

Because the stored score matrix $T$ is orthonormal, orthogonalization is just projection
subtraction. For a current vector or matrix $x$,

```math
x_{\perp} = x - T(T^\top x).
```

This removes the part of $x$ that lies in the span of the previous score vectors and
keeps only the perpendicular remainder.

### Candidate scores Z₀ on previous scores

This branch is only used when `Yadd` is present. Before CCA, the package removes from
each column of $Z_0$ the part that already lies in the span of previous score vectors:

```math
Z_0 \leftarrow Z_0 - T_{1:a-1}(T_{1:a-1}^\top Z_0).
```

Intuitively, `Yadd` enlarges the supervised search space. Without this projection, the
algorithm can keep rediscovering the same score direction through the auxiliary response
columns. The manuscript writes the more general projector with
$(T^\top T)^{-1}$. The package can omit that factor because the stored score matrix $T$
is already orthonormal.

### Final score t on previous scores

This step is always used. Even if `Yadd` is absent, the raw score direction obtained from
$W$ or $W^\circ$ may still contain parts of earlier components. Orthogonalizing
$t_{\mathrm{raw}}$ and then normalizing it ensures

```math
T^\top T = I.
```

This is why later projections can use the simple formula `T * (T' * x)` instead of a
general least-squares projector. It is also what makes the stored component scores easy
to interpret as separate latent axes.

### Mode weights w⁽ʲ⁾ on previous mode weights

This step is optional and only exists in the multilinear branch. If
`orthogonalize_mode_weights = true`, each mode vector is projected away from the
previously stored mode vectors in the same mode:

```math
w_a^{(j)}
\leftarrow
w_a^{(j)} - W_{1:a-1}^{(j)}\!\left(W_{1:a-1}^{(j)\top} w_a^{(j)}\right),
\qquad
w_a^{(j)} \leftarrow \frac{w_a^{(j)}}{\|w_a^{(j)}\|}.
```

Because the final multilinear weight object is an outer product, orthogonalizing the
mode vectors makes the resulting outer-product tensors orthogonal as well. This is useful
when you want loading plots that are directly comparable across components. It is also a
stricter model constraint, so it can reduce predictive performance when the true
component shapes overlap within a mode.

By default this package does **not** impose that restriction, so predictor-side weight
tensors are generally not orthogonal unless `orthogonalize_mode_weights = true`.

## Observation Weights

When `obs_weights` are supplied, three parts of the fit change.

First, preprocessing of $X$ and $Y_{\mathrm{prim}}$ uses weighted means and weighted standard
deviations along the sample mode. Second, candidate loading weights become

```math
W_{0,(1)} = X_{(1)}^\top D_w Y_{\mathrm{comb}},
\qquad
D_w = \operatorname{diag}(w_1,\ldots,w_n).
```

Third, the CCA step uses row-scaled matrices $D_w^{1/2} Z_0$ and $D_w^{1/2} Y$. This
matches the usual covariance-weighting convention: the weights enter linearly in the
cross-products and as square roots in the CCA row scaling.

## Additional Responses (`Yadd`)


The auxiliary response block $Y_{\mathrm{add}}$ is for sample-level information that is available during fitting and is related to the same latent structure as 
$Y_{\mathrm{prim}}$, but is not itself a prediction target. A typical use case is a 
low-noise proxy measurement, metadata, or an auxiliary assay available only for the 
calibration samples.

When `Yadd` is present, the following code branches are activated.

1\. The fitting loop forms `Ycomb = hcat(Y, Yadd)`.

2\. `candidate_loading_weights` and `candidate_scores` use `Ycomb`, so the auxiliary
   columns enlarge the supervised search space.

3\. The candidate-score orthogonalization step for $Z_0$ is turned on.

4\. CCA is still performed against the current deflated $Y$, not against `Ycomb`.

5\. The response loading $q$, the deflation step, the regression coefficients, and
   `predict` all use only `Yprim`.

6\. New samples do not need `Yadd`, because the fitted model stores only predictor-side
   objects and primary-response regression coefficients.


This means that $Y_{\mathrm{add}}$ can therefore make the first few components more parsimonious when $Y_{\mathrm{prim}}$ is noisy but aligned auxiliary information exists. 
In this package, however, `Yadd` is not centered automatically. Under the default 
centered-`X` workflow, constant column offsets in `Yadd` usually do not change the latent directions because they vanish in the product $X_{(1)}^\top Y_{\mathrm{add}}$. If $X$ is 
not centered, or if you want a specific preprocessing convention for the auxiliary block, center `Yadd` manually before calling [`fit`](@ref).
