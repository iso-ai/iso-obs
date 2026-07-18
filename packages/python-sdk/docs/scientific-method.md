# Scientific method and claim discipline

Reliability evidence is valuable only when its claim is no stronger than its
design. The SDK encodes that principle in report contracts and dispositions.

## Abstention is a valid result

Insufficient, conflicting, or out-of-scope evidence produces an unresolved,
review-required, or unsupported disposition. The SDK does not silently turn
missing evidence into success.

## Preserve uncertainty

Point estimates can hide both sampling uncertainty and identification
ambiguity. Reports therefore carry confidence intervals, prediction sets,
identified sets, or unresolved geometric regions as appropriate. Decisions
should use those objects, not only their centers.

## Declare scope

A claim is conditional on its model, software, environment, population,
perturbation, and evidence-use declarations. Content digests make that full
identity stable. Changing a material condition creates a new claim rather
than mutating the old one.

## Separate discovery from confirmation

Exploratory evidence can generate phenotypes, transition candidates, and
failure hypotheses. It must not certify the hypothesis it suggested.
Promotion and release decisions should use independently designated
confirmatory evidence.

## Control multiplicity

Searching many metrics, perturbations, boundaries, or subgroups increases the
chance of a false discovery. Causal campaigns use simultaneous inference or
an explicit multiplicity policy so the family of claims—not each isolated
test—retains the declared error rate.

## Do not let averages erase critical failures

Transportability and release decisions can be noncompensatory. A strong
common-condition result cannot offset failure in a rare, safety-critical
stratum. Declare local limits before analysis and preserve the limiting
stratum in the report.

## Treat simulation evidence conditionally

Simulation supports claims only inside the validated operating envelope.
Use domain randomization and perturbations to explore sensitivity, but use
physics-of-failure knowledge to define meaningful axes and failure
mechanisms. Validate simulator-to-real discrepancy per critical stratum and
never extrapolate beyond the anchor evidence.

## Keep learned outputs separate from truth

Neural encoders and multimodal detectors can surface mixed modes, ambiguous
transitions, and candidate failures at scale. Their predictions should enter
the evidence ledger as suggestions with model and calibration lineage.
Independent adjudication is what turns a suggestion into a trusted label.

## Make reproduction an identity property

Seeds alone are not reproducibility. Preserve system and environment
versions, input and artifact digests, timing assumptions, perturbations,
acceptance criteria, and report schema versions. A content-addressed replay
capsule can then become a regression test without losing the original claim
boundary.
