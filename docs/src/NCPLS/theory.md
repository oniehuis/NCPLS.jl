# Theory

N-way Canonical Partial Least Squares (N-CPLS) extracts supervised latent components from
a predictor array
$X \in \mathbb{R}^{n \times p_1 \times \cdots \times p_d}$ and a primary-response matrix
$Y_{\mathrm{prim}} \in \mathbb{R}^{n \times q}$. Optional additional responses
$Y_{\mathrm{add}} \in \mathbb{R}^{n \times r}$ can help choose the components, but only
`Yprim` is predicted.

The paper by Liland et al. presents the method mainly in tensor notation. For
understanding the algorithm, it is often easier to first unfold the predictor array along
the sample mode. Let

```math
X_{(1)} \in \mathbb{R}^{n \times p},
\qquad
p = \prod_{j=1}^d p_j,
```

be the matrix obtained by stacking each sample tensor into one row. In that view, each
component step becomes an ordinary matrix calculation, followed optionally by a
multilinear rank-1 approximation.

**Implementation note.** The package preprocesses `X` and `Yprim` according to the model
settings. By default, `Yprim` is centered and its mean is added back during prediction.
When `X` is also centered along the sample mode, this centering of `Yprim` usually does
not change the extracted latent directions; it mainly ensures that the response intercept
is handled automatically. `Yadd`, in contrast, is only converted to `Float64` and
validated for size. It is not centered or scaled automatically.

Below, `Y` denotes the current deflated working copy of the preprocessed primary-response
matrix. At the start of fitting, `Y` equals the preprocessed `Yprim`.

## Basic Introduction

Each component is built from four ideas: create response-specific predictor directions,
combine them with CCA, optionally compress them to a multilinear rank-1 form, and extract
an orthogonal score vector.

### 1. First supervised compression

If `Yadd === nothing`, the algorithm works only with `Y`. Otherwise it builds the combined
response block

```math
Y_{\mathrm{comb}} =
\begin{cases}
Y, & \text{if } Y_{\mathrm{add}} \text{ is absent}, \\
[Y \;\; Y_{\mathrm{add}}], & \text{if } Y_{\mathrm{add}} \text{ is present},
\end{cases}
\qquad
Y_{\mathrm{comb}} \in \mathbb{R}^{n \times m},
```

with $m = q$ or $m = q + r$, respectively.

The first compression computes the candidate loading weights

```math
W_{0,(1)} = X_{(1)}^\top Y_{\mathrm{comb}}.
```

`W_{0,(1)}` has `p` rows and `m` columns. After refolding, it becomes a tensor `W0` with
shape `p1 x p2 x ... x pd x m`, so there is one predictor-shaped slice for each response
column.

For a fixed response column $k$, the corresponding slice is

```math
W_{0,:,\ldots,:,k}
=
\sum_{i=1}^n y_{\mathrm{comb},ik}\, X_{i,:,:,\ldots,:}.
```

This is the key point behind the "first compression": it does not yet produce the final
component. It produces one predictor-side direction per response column. When `X` and the
corresponding response column are centered, each entry is proportional to a
cross-covariance between one predictor position and that response. The factor
$1/(n-1)$ is omitted because only the direction matters. So this object is not a full
covariance matrix. It is a predictor-shaped direction for each response column.

The same formula also explains why constant response offsets are often irrelevant for
component extraction. If `X` is centered along the sample mode, then

```math
X_{(1)}^\top \mathbf{1} = 0.
```

Therefore adding a constant offset to any column of `Ycomb` does not change
`W_{0,(1)}`. Under the default centered-`X` workflow, constant shifts in `Yprim` or
`Yadd` typically do not change the latent directions. For `Yprim`, they still matter for
prediction because the mean response must be restored afterwards.

If there is only one response column, there is only one such slice. After the later
collapse by the canonical vector `c`, the result is simply one predictor-side
vector/matrix/tensor for the current component.

A useful shape diagram is

```text
X_(1):   n x p
Y_comb:  n x m

W0_(1) = X_(1)' Y_comb      -> p x m
Z0     = X_(1)  W0_(1)      -> n x m
CCA(Z0, Y) -> c             -> m x 1
W_(1)  = W0_(1) c           -> p x 1
```

### 2. Canonical combination

Projecting `X` onto the response-specific directions gives the candidate scores

```math
Z_0 = X_{(1)} W_{0,(1)}.
```

`Z0` has one column per response column in `Ycomb`. CCA is then run between `Z0` and the
current primary-response matrix `Y`. The dominant left canonical weight vector tells us
how to combine the columns of `Z0` so that the resulting linear combination is maximally
correlated with `Y`.

```math
c
=
\text{first left canonical weight vector from }
\mathrm{CCA}(Z_0, Y),
\qquad
W_{(1)} = W_{0,(1)} c.
```

This is why `c` is a vector: it only needs one coefficient per column of `Z0`. The
product `W0 * c` collapses the response dimension and returns one predictor-side weight
object `W` for the current component.

### 3. Optional multilinear compression

Up to this point the algorithm has produced one unconstrained predictor-side weight
object `W`. If the predictors are unfolded into one long variable axis, then `W` has

```math
\prod_{j=1}^d p_j
```

free weights. The multilinear branch replaces this by a much smaller set of mode-specific
vectors with only

```math
\sum_{j=1}^d p_j
```

free weights.

If `multilinear = false`, the package uses `W` directly. This is the unfolded branch, and
it is equivalent to stopping after the CPLS-style calculation and keeping the full
predictor-side direction.

If `multilinear = true`, the package refolds `W` to predictor shape and approximates it by
a rank-1 multilinear object

```math
W^\circ = w^{(1)} \circ w^{(2)} \circ \cdots \circ w^{(d)}.
```

The exact approximation depends on the number of predictor modes.

1. If `d = 1`, `W` is just normalized.
2. If `d = 2`, the leading rank-1 SVD approximation is used.
3. If `d >= 3`, a one-component PARAFAC model is fitted.

After this factorization, the package optionally orthogonalizes the mode vectors on
previous mode vectors, normalizes them, and then recombines them into the outer-product
tensor `W^\circ`.

Two points are important here.

First, the extracted mode vectors are not the objects used directly for the score
calculation. They are recombined into the single predictor-side weight tensor `W^\circ`,
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
predictor-side weight object. In the multilinear branch this object is `W^\circ`; in the
unfolded branch it is simply `W`. Using `W^\circ` to denote the final object passed to the
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
predictors and the stored cumulative coefficient array, and the mean of `Yprim` is added
back afterwards.

## Orthogonalization Explained

The paper mainly discusses two orthogonalization ideas. The package contains three
orthogonalization-related operations in total, and separating them makes the algorithm
much easier to understand.

Because the stored score matrix `T` is orthonormal, orthogonalization is just projection
subtraction. For a current vector or matrix `x`,

```math
x_{\perp} = x - T(T^\top x).
```

This removes the part of `x` that lies in the span of the previous score vectors and
keeps only the perpendicular remainder.

### Candidate scores `Z0` on previous scores

This branch is only used when `Yadd` is present. Before CCA, the package removes from
each column of `Z0` the part that already lies in the span of previous score vectors:

```math
Z_0 \leftarrow Z_0 - T_{1:a-1}(T_{1:a-1}^\top Z_0).
```

Intuitively, `Yadd` enlarges the supervised search space. Without this projection, the
algorithm can keep rediscovering the same score direction through the auxiliary response
columns. The manuscript writes the more general projector with
$(T^\top T)^{-1}$. The package can omit that factor because the stored score matrix `T`
is already orthonormal.

### Final score `t` on previous scores

This step is always used. Even if `Yadd` is absent, the raw score direction obtained from
`W` or `W^\circ` may still contain parts of earlier components. Orthogonalizing
`t_raw` and then normalizing it ensures

```math
T^\top T = I.
```

This is why later projections can use the simple formula `T * (T' * x)` instead of a
general least-squares projector. It is also what makes the stored component scores easy
to interpret as separate latent axes.

### Mode weights `w^(j)` on previous mode weights

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

First, preprocessing of `X` and `Yprim` uses weighted means and weighted standard
deviations along the sample mode. Second, candidate loading weights become

```math
W_{0,(1)} = X_{(1)}^\top D_w Y_{\mathrm{comb}},
\qquad
D_w = \operatorname{diag}(w_1,\ldots,w_n).
```

Third, the CCA step uses row-scaled matrices `D_w^{1/2} Z0` and `D_w^{1/2} Y`. This
matches the usual covariance-weighting convention: the weights enter linearly in the
cross-products and as square roots in the CCA row scaling.

## Additional Responses (`Yadd`)

`Yadd` is for sample-level information that is available during fitting and is related to
the same latent structure as `Yprim`, but is not itself a prediction target. A typical
use case is a low-noise proxy measurement, metadata, or an auxiliary assay available only
for the calibration samples.

When `Yadd` is present, the following code branches are activated.

1. The fitting loop forms `Ycomb = hcat(Y, Yadd)`.
2. `candidate_loading_weights` and `candidate_scores` use `Ycomb`, so the auxiliary
   columns enlarge the supervised search space.
3. The candidate-score orthogonalization step for `Z0` is turned on.
4. CCA is still performed against the current deflated `Y`, not against `Ycomb`.
5. The response loading `q`, the deflation step, the regression coefficients, and
   `predict` all use only `Yprim`.
6. New samples do not need `Yadd`, because the fitted model stores only predictor-side
   objects and primary-response regression coefficients.

`Yadd` can therefore make the first few components more parsimonious when `Yprim` is
noisy but aligned auxiliary information exists. In this package, however, `Yadd` is not
centered automatically. Under the default centered-`X` workflow, constant column offsets
in `Yadd` usually do not change the latent directions because they vanish in the product
`X_{(1)}^\top Y_{\mathrm{add}}`. If `X` is not centered, or if you want a specific
preprocessing convention for the auxiliary block, center `Yadd` manually before calling
[`fit`](@ref).
