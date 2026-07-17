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

# Print the CLI version.
iso version
```

## Configuration

| Source | Purpose |
| --- | --- |
| `ISO_OBS_API_KEY` | API key; overrides the stored config. |
| `ISO_OBS_BASE_URL` | API base URL; defaults to `https://api.iso-obs.com/api/v1`. |
| `~/.config/iso-obs/config.toml` | Key stored by `iso auth login`. |

Exit codes: `1` for API/network failures, `2` for missing or rejected
credentials.
