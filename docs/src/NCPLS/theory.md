# Theory

N-way Canonical Partial Least Squares (N-CPLS) is a supervised latent-variable method for
regression and classification tailored to multiway (tensor-valued) predictor data. It
extends the canonical partial least squares (CPLS) framework to predictors with more than
two modes, enabling direct modelling of multidimensional arrays rather than requiring prior
unfolding.

The aim of N-CPLS is to extract latent components that summarise the predictor tensor
$\mathcal{X} \in \mathbb{R}^{n \times p_1 \times \cdots \times p_d}$, where $n$ is the
number of samples and $p_1, \dots, p_d$ are the sizes of the variable modes, such that
these components are maximally informative with respect to a multivariate response matrix
$Y \in \mathbb{R}^{n \times q}$. As in CPLS, the extraction of components is guided
by a correlation-based criterion obtained via canonical correlation analysis (CCA), rather
than the covariance maximisation used in classical PLS. This makes N-CPLS applicable to
both regression and classification problems within a unified framework.

In addition to modelling one or more primary responses, N-CPLS allows the inclusion of
auxiliary response variables that are used during component extraction but are not
necessarily targets for prediction. These additional responses can guide the estimation
of latent components towards relevant structure in the data, potentially improving model
parsimony and predictive performance. As a latent-variable method, N-CPLS is particularly
well suited for settings where the number of predictors greatly exceeds the number of
samples ($p \gg n$) and where strong collinearity is present among predictors. By
projecting the data onto a low-dimensional latent space, the method mitigates issues
related to high dimensionality and multicollinearity.

A key distinction from classical CPLS is how the predictor weights are parameterised. In
the multilinear formulation of N-CPLS, the weight structure is constrained to reflect the
multiway nature of the data. Instead of estimating a single unconstrained weight vector
over all predictor variables, the method estimates a set of mode-specific weight vectors
$w^{(1)}, \dots, w^{(d)}$, one for each predictor dimension. The overall loading weight
tensor is then constructed as their outer product,
$\mathcal{W} = w^{(1)} \circ \cdots \circ w^{(d)}$.

This multilinear constraint substantially reduces the number of free parameters—from the
product of all mode dimensions to their sum—yielding a more parsimonious model. While this
restriction may limit flexibility compared to an unfolded approach, it arises from
representing the loading weights as an outer product of mode-specific vectors, which allows
inspection of variable contributions within each mode (e.g., temporal and spectral),
thereby facilitating interpretation of how each dimension contributes to the extracted
components. Moreover, this reduced parametrisation acts as an implicit regularisation,
which can improve generalisation and reduce sensitivity to overfitting, provided that the
multilinear assumption is reasonably satisfied.

## Basic procedure

N-CPLS for multiway predictors can be understood as a two-stage supervised projection 
method. Let the predictor data be a tensor 
$\mathcal{X} \in \mathbb{R}^{n \times p_1 \times \cdots \times p_d}$, where the first mode 
indexes samples and the remaining $d$ modes describe structured predictor dimensions such 
as time, ions, wavelength, image rows, or image columns. Let 
$Y_{\mathrm{prim}} \in \mathbb{R}^{n \times q}$ denote the primary responses to be 
predicted, and optionally let $Y_{\mathrm{add}} \in \mathbb{R}^{n \times r}$ denote 
additional responses that are available during training and should influence the extracted 
latent structure without necessarily being prediction targets.

As in CPLS, N-CPLS first constructs a supervised intermediate representation from 
$\mathcal{X}$ and the combined response block $[Y_{\mathrm{prim}} ;; Y_{\mathrm{add}}]$. 
This is done by computing response-specific directions in predictor space via contraction 
of $\mathcal{X}$ with the response block along the sample mode. In the matrix case, this 
corresponds to forming $W_0 = X^\top [Y_{\mathrm{prim}} ;; Y_{\mathrm{add}}],$ where each 
column of $W_0$ represents a predictor direction associated with one response variable.

The predictor data are then projected onto these response-specific directions, yielding a 
candidate score matrix $Z_0 = X W_0 \in \mathbb{R}^{n \times (q+r)}.$ Each column of $Z_0$ 
summarises the predictor tensor in a direction that reflects variation relevant to a 
particular response variable. In this way, the original high-dimensional predictor data are 
reduced to a small set of supervised candidate components, one per response.

In the second stage, these candidate components are combined into a single latent variable 
that is optimally aligned with the primary responses. This is achieved by performing 
canonical correlation analysis (CCA) between $Z_0$ and $Y_{\mathrm{prim}}$, yielding a 
vector of canonical weights $c \in \mathbb{R}^{q+r}$. These weights define how the 
candidate components should be linearly combined to maximise their correlation with the 
primary responses.

The final predictor-side loading weights are then obtained by mapping this combination back 
to the predictor space as $\mathcal{W} = \mathcal{W}0 \times c,$ which collapses the 
response dimension by forming a weighted combination of the response-specific directions. 
In the multilinear formulation of N-CPLS,this weight tensor is subsequently approximated by 
an outer product of mode-specific vectors, 
$\mathcal{W}\circ = w^{(1)} \circ \cdots \circ w^{(d)},$ yielding a separable 
representation of the component. Importantly, subsequent computations are not performed on 
the individual mode-specific vectors $w^{(1)}, \dots, w^{(d)}$, but on the combined loading 
weight tensor $\mathcal{W}\circ$. 

The component score vector is obtained by projecting the predictor tensor onto this tensor, 
$t = \mathcal{X} ,\circledast_d, \mathcal{W}\circ,$ where the contraction is taken over all 
predictor modes, resulting in a score vector $t \in \mathbb{R}^n$. The predictor loadings 
are computed as $\mathcal{P} = \mathcal{X}^\top t / (t^\top t),$ and the response loadings 
for the primary responses as $q = Y_{\mathrm{prim}}^\top t / (t^\top t).$

To ensure that subsequent components capture new information, N-CPLS employs 
orthogonalisation in a conditional manner. When additional responses are included, the 
candidate score matrix $Z_0$ is orthogonalised with respect to previously extracted score 
vectors before the canonical correlation step. This prevents information introduced 
through $Y_{\mathrm{add}}$ from being repeatedly captured across components. When no 
additional responses are used, this orthogonalisation step is omitted.

In the multilinear formulation, an additional optional orthogonalisation can be applied to 
the mode-specific weight vectors $w^{(j)}$ across components. For each mode $j$ and 
component $a$, this corresponds to removing the projection of $w^{(j)}a$ onto the subspace 
spanned by the previously estimated mode-$j$ weight vectors, 
$w^{(j)}a \leftarrow w^{(j)}a - W^{(j)}{1:a-1}(W^{(j)\top}{1:a-1} W^{(j)}{1:a-1})^{-1} W^{(j)\top}_{1:a-1} w^{(j)}_a$, 
followed by normalisation. This enforces orthogonality of the loading vectors within each 
mode.

Because the multilinear loading tensor is formed as an outer product, 
$\mathcal{W}_{\circ,a} = w^{(1)}_a \circ \cdots \circ w^{(d)}_a$, orthogonalising each 
$w^{(j)}_a$ implies that the resulting component loading tensors become orthogonal in the 
unfolded predictor space. This yields loading vectors per mode that are directly 
comparable across components, which simplifies interpretation and visualisation.

However, this step imposes an additional constraint on the solution: each new component is 
restricted to be orthogonal to previous ones within every mode separately, not only in the 
combined predictor space. As a result, the admissible set of loading tensors is reduced, 
which can prevent the model from capturing directions that would otherwise improve fit or 
prediction. Consequently, while this orthogonalisation can enhance interpretability, it may 
also reduce predictive performance when the true structure is not well aligned with these 
per-mode orthogonality constraints.

After the score vector $t$ has been computed, the algorithm proceeds with response 
deflation. Unlike classical N-PLS, N-CPLS avoids repeated deflation of the full predictor 
tensor and instead relies on orthogonalisation and response deflation to ensure that 
subsequent components capture new variation.

The algorithm then repeats the same sequence of steps—construction of candidate directions, 
canonical combination, multilinear approximation, and score computation—until the desired 
number of components has been extracted.

Finally, the regression coefficients are formed by combining the component-wise 
contributions of the loading weight tensors and response loadings. Rather than a simple 
summation, this is implemented as element-wise multiplication along the component 
dimension, followed by cumulative summation across components. Denoting by 
$\mathcal{W}_{\circ}$ the stack of component-wise loading weight tensors and by $Q$ the 
corresponding matrix of response loadings, the regression coefficients $\mathcal{B}$ are 
obtained through this component-wise accumulation.

This formulation allows predictions for all components and all responses to be computed 
simultaneously. In particular, predictions are given by
$\hat{Y} = \mathcal{X} ,\circledast_d, \mathcal{B} + \mathbf{1}\bar{y}^\top,$ where 
$\mathcal{B}$ contains the accumulated regression coefficients across components. The 
resulting prediction array has dimensions corresponding to samples, components, and 
responses, enabling evaluation of model performance as a function of the number of 
extracted components.

## Additional responses

## Observational weights

