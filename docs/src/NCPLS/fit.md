# Fit

`fit` now infers the task directly from the supplied response object. There is no
separate `analysis_mode` switch anymore.

The three main input conventions are:

1. `fit(m, X, y::AbstractVector{<:Real})` for univariate numeric regression.
2. `fit(m, X, Y::AbstractMatrix{<:Real})` for a user-defined response block.
3. `fit(m, X, labels::AbstractCategoricalArray)` for class labels, internally converted
   to a one-hot response matrix.

For custom matrix responses, `sampleclasses` and `responselabels` can also define which
response columns carry class scores:

- if none of the unique `sampleclasses` appear in `responselabels`, the labels are kept
  as sample metadata only,
- if only some appear, `fit` throws an error,
- if all appear, each label must occur exactly once and the matched response columns must
  form a one-hot block that agrees row-wise with `sampleclasses`.

This means that mixed response blocks such as `[class indicators | continuous traits]`
are supported at fit time without any extra model field.

## API

```@docs
NCPLS.fit
```
