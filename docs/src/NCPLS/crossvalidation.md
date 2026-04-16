# Cross Validation

The dedicated discriminant-analysis helpers `cvda`, `permda`, and `outlierscan` are
still restricted to categorical labels or pure one-hot response matrices. They do not
accept mixed response blocks that also contain continuous trait columns.

If you fit a mixed response model such as `[class indicators | continuous traits]` and
want cross-validation, use `nestedcv` or `nestedcvperm` directly with callbacks that
score and decode only the class sub-block you care about.

## API

```@docs
NCPLS.cvda
NCPLS.cvreg
NCPLS.nestedcv
NCPLS.nestedcvperm
NCPLS.outlierscan
NCPLS.permda
NCPLS.permreg
NCPLS.pvalue
```
