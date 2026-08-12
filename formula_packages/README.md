# Legacy formula-package backup

This project-local folder is retained only for compatibility with earlier
development copies. V1.0.0 stores active user formulas outside the
installation under `%LOCALAPPDATA%\PEA-PGNN\V1.0.0\FormulaData`. The shipped
`custom/` folder is deliberately empty and is not used for new saves.

The expression language is intentionally restricted. Formula packages cannot
import Python modules, open files, start processes, access the network, call
object attributes, or execute arbitrary Python statements.

Available variables:

`t`, `t0`, `RH`, `T`, `h0`, `VtoS`, `ks`, `wb`, `fc28`, `cement`, `water`,
`aggregate`, and `Ec28`.

Available functions:

`abs`, `sqrt`, `exp`, `log`, `log10`, `log1p`, `tanh`, `minimum`, `maximum`,
`clip`, and `where`.

Custom formulas are comparison curves only.  The trained PEA-PGNN still uses
its frozen B3/GL2000/ACI prior features.  Making a new formula part of the
neural model requires rebuilding the feature contract and retraining.
