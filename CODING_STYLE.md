# Coding Style

This document defines the enforced coding style for `ambench` release cleanup work.

It is based on the Isaac Lab contribution guide's coding-style section and adapted to this repository's current toolchain and structure.

## Baseline

- Follow the Google Python Style Guide as the default baseline.
- Follow PEP 8 for code layout, comments, and readability.
- Use Python type hints following PEP 484 and modern built-in generics from PEP 585 where applicable.
- Use Google-style docstrings when a docstring is needed.

## File Structure

Within a Python module, prefer this order:

1. imports
2. constants
3. public functions
4. public classes
5. private functions
6. private classes

Keep files readable and predictable. Do not mix unrelated helpers, runtime logic, and one-off debugging code in the same module.

## Imports

- Keep imports sorted by the formatter and import-sorting tools.
- Prefer explicit imports over wildcard imports.
- Keep import side effects to a minimum.
- Avoid circular imports.

If a circular import is hard to avoid:

- first move shared types or helpers to a lower-level module
- only tolerate circularity at a clear package boundary
- do not introduce local import hacks unless there is a concrete runtime reason

## Type Hints

- Add type hints to new or edited public functions, methods, and important internal helpers.
- Prefer accurate return types over broad `Any`.
- Use built-in collection generics such as `list[str]` and `dict[str, int]` when supported.
- Do not duplicate signature type information in docstrings.

## Docstrings And Comments

- Write docstrings when behavior, assumptions, arguments, return values, side effects, or units are not obvious from the signature and body.
- Use Google-style docstrings.
- Keep docstrings factual and concise.
- Use comments to explain intent, invariants, coordinate-frame assumptions, or non-obvious decisions.
- Do not add comments that simply restate the code.

## Naming And Readability

- Prefer descriptive names over abbreviations unless the domain already makes the abbreviation standard.
- Keep task, controller, robot, and policy responsibilities clearly separated.
- Do not introduce new abstractions unless they remove clear duplication or simplify maintenance.

## Repository-Specific Rules

- Prefer Isaac Lab APIs and project patterns over direct Isaac Sim APIs unless lower-level behavior is required.
- Match nearby implementations before introducing new patterns.
- Keep controller math reusable and separate from task reward/reset logic.
- Keep data collection, replay, and script-only workflows out of environment classes.
- Do not hard-code machine-specific absolute paths.
- Do not leave development-only debug code, commented-out blocks, or AI-generated filler text in release-facing files.

## Tooling Enforcement

This repo currently enforces style primarily through:

- `pre-commit`
- `black`
- `flake8`
- `isort`
- `pyupgrade`
- `codespell`

Run:

```bash
pre-commit run --all-files
```

For scoped work, prefer running hooks on only the touched files first.

## Practical Standard For Release Cleanup

For release cleanup, code should be:

- behavior-preserving unless the change is explicitly a bug fix
- consistent with nearby modules
- typed where practical
- free of obvious dead code and stray debug artifacts
- formatted and lint-clean under the repo's current hooks

If the upstream Isaac Lab guidance and the current repo tooling differ, follow the stricter rule that improves readability and keeps the repo passing its configured checks.
