# Security Policy

Iso AI takes the security of Reliability Studio seriously. We appreciate the
efforts of security researchers and users who report vulnerabilities
responsibly.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues,
discussions, or pull requests.**

Instead, report them privately by email to **security@iso-obs.com**.

To help us triage and resolve the issue quickly, please include as much of the
following as you can:

- A description of the vulnerability and its potential impact.
- The affected component (SDK, CLI, API, web app, worker/service, or
  infrastructure) and version.
- Step-by-step instructions to reproduce the issue, including any proof-of-
  concept code.
- Any relevant logs, configurations, or environment details.

If you would like to encrypt your report or exchange sensitive material, say so
in your initial email and we will coordinate a secure channel.

## Response Expectations

- **Acknowledgement**: We aim to acknowledge your report within **2 business
  days**.
- **Assessment**: We will provide an initial assessment and severity
  classification within **5 business days**.
- **Updates**: We will keep you informed of our progress as we investigate and
  work on a fix.
- **Resolution**: We will notify you when the issue is resolved and coordinate
  public disclosure timing with you where appropriate.

We ask that you give us a reasonable amount of time to address the issue before
any public disclosure, and that you avoid accessing or modifying other users'
data during your research.

## Supported Versions

Security fixes are applied to the latest released minor version of each
published package. We recommend always running the most recent release.

| Component            | Package            | Supported            |
| -------------------- | ------------------ | -------------------- |
| Python SDK           | `iso-obs`          | Latest minor release |
| CLI                  | `iso-obs-cli`      | Latest minor release |
| Shared schemas       | `iso-obs-schemas`  | Latest minor release |
| API and services     | Hosted             | Current deployment   |

Older versions may not receive security updates. If you depend on an
unsupported version, please upgrade.

## Thank You

Responsible disclosure helps keep the entire community safe. We are grateful for
your help in keeping Reliability Studio and its users secure.
