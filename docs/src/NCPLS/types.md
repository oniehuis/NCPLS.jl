# Types

Two stored label vectors are easy to confuse:

- `sampleclasses(mf)` lives on the sample axis and stores one label per training sample,
- `responselabels(mf)` lives on the response axis and stores one label per response
  column.

These vectors interact only when NCPLS tries to infer a class-response block from a
custom response matrix. If the labels can be matched consistently to a one-hot sub-block
of `Yprim`, classification helpers use that sub-block. Otherwise `sampleclasses(mf)`
remains metadata for grouping and plotting.

## API

```@docs
NCPLS.NCPLSModel
NCPLS.PredictorAxis
NCPLS.AbstractNCPLSFit
NCPLS.NCPLSFitLight
NCPLS.coef
NCPLS.xmean
NCPLS.xstd
NCPLS.ymean
NCPLS.NCPLSFit
NCPLS.fitted
NCPLS.ncomponents
NCPLS.predictoraxes
NCPLS.residuals
NCPLS.responselabels
NCPLS.sampleclasses
NCPLS.samplelabels
NCPLS.xscores
```
