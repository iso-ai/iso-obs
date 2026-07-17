# Reliability ↔ iso-obs coordination contract

This document defines how the Reliability platform and the public `iso-obs`
SDK evolve independently without breaking their shared API.

## Ownership

| Surface | Authoritative repository | Owner workstream |
| --- | --- | --- |
| Hosted API, persistence, workers, and web app | `iso-ai/reliability` | Claude Code |
| Python SDK, CLI, and published schemas | `iso-ai/iso-obs` | Codex |
| HTTP and event compatibility contract | Both, through contract tests | Shared |

`iso-ai/iso-obs` is authoritative for `packages/python-sdk`, `packages/cli`,
and `packages/schemas`. Reliability may consume released versions or a pinned
Git revision, but it must not overwrite these directories in the public
repository.

> Migration note: Reliability currently contains `scripts/sync-public.sh`,
> which mirrors these packages from Reliability into `iso-obs` with
> `rsync --delete`. Do not run that script after SDK work begins here. Replace
> it with a pull/vendor workflow or remove the duplicated packages before the
> next SDK sync.

## Current compatibility baseline

- Reliability baseline: `0986482` (`main`, 2026-07-17)
- iso-obs import baseline: the SDK, CLI, and schemas present at that Reliability
  revision
- API prefix: `/api/v1`
- Authentication: `Authorization: Bearer <api_key>`
- SDK configuration:
  - `ISO_OBS_API_KEY` is required unless passed explicitly.
  - `ISO_OBS_BASE_URL` defaults to `https://api.iso-obs.com/api/v1`.

The phase-one HTTP surface is:

| Method | Path |
| --- | --- |
| `POST`, `GET` | `/projects` |
| `GET` | `/projects/{project_id}` |
| `POST`, `GET` | `/projects/{project_id}/systems` |
| `POST`, `GET` | `/systems/{system_id}/versions` |
| `POST`, `GET` | `/runs` |
| `GET` | `/runs/{run_id}` |
| `POST` | `/runs/{run_id}/events:batch` |
| `POST` | `/runs/{run_id}/complete` |
| `POST` | `/runs/{run_id}/fail` |
| `GET` | `/runs/{run_id}/events` |
| `GET` | `/runs/{run_id}/metrics` |

## Change protocol

1. The proposing workstream opens a small contract change containing examples,
   migration impact, and the intended compatibility level.
2. Add or update consumer-facing schema and HTTP contract tests before changing
   implementation behavior.
3. Additive changes may proceed without a version bump when old payloads remain
   valid. Breaking changes require a new API or trace-schema version.
4. Reliability validates against the candidate `iso-obs` revision. The SDK
   validates against a Reliability test server or captured contract fixtures.
5. Record the paired Reliability commit and `iso-obs` commit in the PRs that
   introduce the change.

## Compatibility invariants

- IDs retain their typed prefixes.
- Event timestamps remain timezone-aware UTC.
- Event order and the monotonic timestamp are preserved within a run.
- Batch ingestion is retry-safe; duplicate event IDs cannot create duplicate
  events.
- Unknown additive payload fields do not break older consumers.
- Authentication credentials are never logged, persisted in traces, or included
  in exceptions.
- A failed final event flush is recoverable from the SDK spill file.
- Every SDK request is bounded by a timeout and maps transport/API failures to
  documented SDK exceptions.

## Required checks before integration

Run in `iso-obs`:

```bash
uv lock
uv sync --all-packages --frozen
uv run ruff check .
uv run black --check .
uv run mypy .
uv run pytest
```

Run in Reliability after updating its pinned SDK/schema revision:

```bash
uv run pytest apps/api/tests
```

The workstreams may merge independently only when their changes are internal.
Any shared-contract change is complete only after both check sets pass against
the paired revisions.
