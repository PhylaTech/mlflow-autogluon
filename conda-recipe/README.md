# conda-forge recipe

This directory holds the conda recipe used to publish `mlflow-autogluon` to the
[conda-forge](https://conda-forge.org/) channel, so it can be installed with
conda/mamba and referenced from `environment.yml` files:

```yaml
# environment.yml
channels:
  - conda-forge
dependencies:
  - mlflow-autogluon
```

## Design notes

- **`noarch: python`**: the package is pure Python, so a single build serves
  every platform and Python version.
- **Minimal runtime dependency**: only `mlflow >=2.15`. The AutoGluon predictor
  packages are imported lazily inside functions, so `import mlflow_autogluon`
  succeeds without them; users install `autogluon.tabular` / `.timeseries` /
  `.multimodal` as needed. This keeps the conda-forge dependency graph small and
  avoids pinning the (large, fast-moving) AutoGluon stack.
- The `sha256` and `version` correspond to the PyPI sdist and are bumped
  automatically by the conda-forge autotick bot once the feedstock exists.

## Initial submission (one time)

conda-forge packages are created by adding this recipe to
[conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes):

1. Fork `conda-forge/staged-recipes`.
2. Copy `meta.yaml` to `recipes/mlflow-autogluon/meta.yaml` in that fork.
3. Open a PR to `conda-forge/staged-recipes`.
4. A conda-forge reviewer merges it; a bot then creates the
   `mlflow-autogluon-feedstock` repository and the package appears on the
   conda-forge channel within an hour or so.

## Ongoing maintenance

After the feedstock exists, this in-repo copy is a reference only: the
authoritative recipe lives in the feedstock, and the autotick bot opens a PR
there for each new PyPI release. Keep this copy in sync when the recipe
structure (not just the version) changes.
