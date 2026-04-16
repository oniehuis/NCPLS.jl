# Projection and Prediction

`predict` always returns the full numeric response block. It does not convert class-score
columns to labels automatically.

For classification-capable fits, NCPLS provides two decoding layers on top of
`predict`:

- `onehot(mf, X)` returns one-hot predictions for the inferred class-response block,
- `sampleclasses(mf, X)` maps those class scores back to labels,
- `predictclasses(mf, X)` is a convenience alias for the prediction overload of
  `sampleclasses`.

For mixed response blocks such as `[class indicators | continuous traits]`, the class
helpers use only the inferred class columns, while `predict` still returns all columns.

## API

```@docs
NCPLS.onehot
NCPLS.predict
NCPLS.predictclasses
NCPLS.project
```
