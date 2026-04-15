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
This is done by computing, for each response variable, a corresponding direction in 
predictor space via contraction of $\mathcal{X}$ with the response block along the sample 
mode. In the matrix case, this corresponds to forming 
$W_0 = X^\top [Y_{\mathrm{prim}} ;; Y_{\mathrm{add}}]$, where each column of $W_0$ 
represents a predictor direction associated with one response variable.

The predictor data are then projected onto these response-specific directions, yielding a 
candidate score matrix $Z_0 \in \mathbb{R}^{n \times (q+r)}$. Each column of $Z_0$ 
summarises the predictor tensor in a direction that reflects variation relevant to a 
particular response variable. In this way, the original high-dimensional predictor data are 
reduced to a small set of supervised candidate components, one per response.

In the second stage, these candidate components are combined into a single latent variable 
that is optimally aligned with the primary responses. This is achieved by performing 
canonical correlation analysis (CCA) between $Z_0$ and $Y_{\mathrm{prim}}$, yielding a 
vector of canonical weights $c$. These weights define how the candidate components should 
be linearly combined to maximise their correlation with the primary responses.

The final predictor-side loading weights are then obtained by mapping this combination back 
to the predictor space as $\mathcal{W} = \mathcal{W}_0 \times c$, where the response 
dimension is collapsed. In the multilinear formulation of N-CPLS, this weight tensor is 
subsequently approximated by an outer product of mode-specific vectors,
$\mathcal{W} \approx w^{(1)} \circ \cdots \circ w^{(d)}$,
yielding a separable representation of the component. The corresponding score vector is 
obtained by projecting $\mathcal{X}$ onto this multilinear weight tensor.

