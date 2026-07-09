# Contributing

Issues and pull requests are welcome at
[PhylaTech/mlflow-autogluon](https://github.com/PhylaTech/mlflow-autogluon).

## Development environment

=== "pixi (recommended)"

    The repo ships a `pixi.toml` with two environments: the **default**
    environment (tabular predictor plus test, lint, and docs tooling; fast to
    install) and the **full** environment (adds timeseries and multimodal for
    CI-parity coverage runs).

    ```bash
    pixi install          # default environment
    pixi install -e full  # everything, needed for coverage / test-all
    ```

    All common operations are pixi tasks:

    | Task | Environment | What it does |
    | --- | --- | --- |
    | `pixi run test` | default | tabular test suite |
    | `pixi run lint` | default | `ruff check` |
    | `pixi run docs` | default | live docs server (`mkdocs serve`) |
    | `pixi run docs-build` | default | strict docs build |
    | `pixi run -e full test-all` | full | complete test suite |
    | `pixi run -e full test-timeseries` | full | timeseries tests only |
    | `pixi run -e full test-multimodal` | full | multimodal tests only |
    | `pixi run -e full coverage` | full | CI-parity gate, 100 percent required |

=== "conda / mamba"

    The repo also ships an `environment.yml` (includes all predictor extras):

    ```bash
    mamba env create -f environment.yml
    mamba run -n mlflow-autogluon pip install -e .

    # Tests (fast: the suite trains tiny DUMMY-model predictors)
    mamba run -n mlflow-autogluon pytest

    # Coverage (CI enforces 100 percent with all predictor extras installed)
    mamba run -n mlflow-autogluon pytest --cov=mlflow_autogluon --cov-fail-under=100

    # Lint
    mamba run -n mlflow-autogluon ruff check mlflow_autogluon tests
    ```

=== "pip"

    ```bash
    pip install -e .[dev]
    pytest
    ```

CI runs the suite on Python 3.10 to 3.12 against the latest MLflow, plus a job pinned
to `mlflow<3` to guard the oldest supported line and a coverage job with all
predictor extras that enforces 100 percent.

## Documentation

Docs are MkDocs Material, deployed to GitHub Pages on every push to `main`:

```bash
mamba run -n mlflow-autogluon mkdocs serve
```

## Commit messages and releases

Commits to `main` follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `chore:`, with `!` or a `BREAKING CHANGE:` footer for
breaking changes). [release-please](https://github.com/googleapis/release-please)
watches `main`, maintains a running release PR with the version bump and
`CHANGELOG.md`, and merging that PR tags the release. Publishing the GitHub
release triggers the `Publish to PyPI` workflow, which uploads the package via
PyPI trusted publishing. Versions live in `pyproject.toml` and
`mlflow_autogluon/__init__.py`; both are managed by release-please, never bump
them by hand.

## Roadmap

- ClearML integration (tracked separately)
