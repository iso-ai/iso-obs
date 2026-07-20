# iso-obs-cli

Command-line interface for [Reliability Studio](https://github.com/iso-ai/reliability).
Installs the `iso` command.

## Installation

```bash
pip install iso-obs-cli
```

## Usage

```bash
# Store an API key (written to ~/.config/iso-obs/config.toml, mode 600).
iso auth login --api-key <YOUR_KEY>

# Show the masked key and the API base URL in use.
iso auth whoami

# Create a project and write iso-obs.toml in the current directory.
iso project init --name "lunar-lander"

# Register a system version.
iso system register \
    --project prj_1234567890abcdef12345678 \
    --name "lander-policy" \
    --version "1.2.0" \
    --artifact-uri s3://models/lander-1.2.0.pt

# Inspect a run and its metrics.
iso run inspect run_1234567890abcdef12345678

# Inspect and audit local dataset evidence without an API key.
iso dataset ingest trajectories.jsonl \
    --plan ingestion-plan.json \
    --output dataset-bundle.json \
    --report ingestion-report.json
iso dataset inspect dataset-bundle.json
iso dataset synchronize dataset-bundle.json --plan synchronization-plan.json
iso dataset audit dataset-bundle.json --plan reliability-audit-plan.json
iso dataset split split-request.json
iso dataset compile-replay replay-request.json \
    --bundle dataset-bundle.json \
    --synchronization synchronization-report.json \
    --adapter simulation-adapter.json \
    --output failure-replay-capsule.json

# Submit any SDK report produced with report.to_json().
iso evidence submit evidence-report.json --project-id prj_1234567890abcdef12345678

# Print the CLI version.
iso version
```

## Configuration

| Source | Purpose |
| --- | --- |
| `ISO_OBS_API_KEY` | API key; overrides the stored config. |
| `ISO_OBS_BASE_URL` | API base URL; defaults to `https://reliability-studio-5cmy6.ondigitalocean.app/api/v1`. |
| `~/.config/iso-obs/config.toml` | Key stored by `iso auth login`. |

Dataset audit commands emit canonical JSON to stdout, or to a new file with
`--output`. Ingestion creates a linked bundle/report pair at the required
`--output` and `--report` paths. Existing files are never overwritten.

JSONL ingestion is available in the base installation. Install the optional
Parquet or MCAP readers with:

```bash
pip install "iso-obs[parquet]"
pip install "iso-obs[mcap]"
pip install "iso-obs[mcap-ros2]"  # includes decoded ROS 2 messages
pip install "iso-obs[rosbag2]"     # split MCAP recordings + metadata.yaml
```

Ingestion requires explicit nested field paths, timestamp units, channel
modalities, and clock identifiers. It hashes source bytes and payload content
but does not infer labels, clock alignment, or dataset split roles.

MCAP plans additionally declare topic names, expected message and schema
encodings, raw/JSON/decoded payload handling, timestamp source, and episode
windows. The same command selects the MCAP contract by its schema version:

```bash
iso dataset ingest recording.mcap \
    --plan mcap-ingestion-plan.json \
    --output dataset-bundle.json \
    --report ingestion-report.json
```

The reader validates available chunk and data-section CRCs and preserves
physical record order. Log time, publish time, and ROS header time are never
substituted for one another.

For a rosbag2 directory containing `metadata.yaml` and split MCAP files, use a
versioned `iso-obs.rosbag2-mcap-ingestion-plan.v1` wrapper around the MCAP plan:

```bash
iso dataset ingest robot-run-42/ \
    --plan rosbag2-ingestion-plan.json \
    --output dataset-bundle.json \
    --report ingestion-report.json \
    --recording-report recording-evidence.json
```

The recording evidence is required. It retains the metadata digest, ordered
file paths and digests, byte counts, declared and observed message/timing
statistics, and topic-level type/serialization checks. File order comes from
metadata, while physical order is preserved within each file. The CLI does not
glob files, globally sort messages, or deduplicate repeated split-boundary
messages. Manifest disagreement and overlapping file times produce exit code
`3` with all three artifacts intact.

`iso dataset compile-replay` joins a strict replay request with its immutable
dataset bundle, synchronization report, simulator adapter manifest, and
optional rosbag2 recording evidence. It preserves a valid capsule at
`--output`, including review and insufficient-evidence results. Exit code `3`
means either reconstruction is approximate or the failure claim requires
review; inspect the independent `fidelity` and
`failure_claim_qualification` fields. Exit code `5` means replay construction
is blocked. Cross-artifact digest or identity mismatches are invalid input and
produce exit code `2` without an output capsule.

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Operation succeeded; the scientific claim is supported within scope. |
| `1` | API or network failure. |
| `2` | Usage, local input, or authentication failure. |
| `3` | A valid report requires review. |
| `4` | The split audit demonstrates contamination. |
| `5` | The report abstains because evidence is insufficient. |

Nonzero scientific outcomes still emit the full report. Automation should
retain that artifact rather than relying on the exit code alone.

## Evidence report submission

`iso evidence submit <report.json>` accepts SDK reports that use either the
top-level `report_schema_version` or `schema_version` convention. It validates
the report as an unambiguous JSON object, serializes it using the SDK's
canonical JSON rules, computes the corresponding `sha256:` content digest, and
posts the integrity envelope to `/evidence/reports` with the configured bearer
API key.

Use `--project-id` to associate the report with a project. The command reports
whether the platform created the content-addressed record (`201`) or returned
the existing record (`200`). Conflicting versions, duplicate JSON keys,
non-finite numbers, and malformed reports are rejected locally before a network
request.
