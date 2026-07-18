"""Tests for scoped, non-compensatory release assurance."""

from __future__ import annotations

from dataclasses import replace

import pytest

from iso_obs.assurance import (
    AssuranceDefeater,
    AssuranceDisposition,
    AssurancePolicy,
    AssuranceReasonCode,
    ChallengeEvidence,
    ChallengeKind,
    ChallengeOutcome,
    DefeaterSeverity,
    DefeaterStatus,
    RegressionEvidenceLink,
    ReleaseAssuranceDossier,
    ReleaseScope,
    evaluate_release_assurance,
)
from iso_obs.evidence import NamedDigest, sha256_digest
from iso_obs.execution import SimulationCampaignReport
from iso_obs.promotion import (
    PromotionRequirement,
    PromotionStatus,
    RegressionPromotionReview,
)
from iso_obs.regression import RegressionGateReport


def digest(value: str) -> str:
    """Build a valid digest for test content."""
    return sha256_digest(value)


def release_scope() -> ReleaseScope:
    """Build a content-addressed release scope."""
    return ReleaseScope(
        scope_id="warehouse-navigation",
        scope_version="1",
        intended_use="Low-speed autonomous navigation in mapped warehouses.",
        system_artifacts=(
            NamedDigest("planner", digest("planner-v8")),
            NamedDigest("policy", digest("policy-v18")),
        ),
        operational_domain_digest=digest("warehouse-odd"),
        validity_envelope_digest=digest("validated-envelope"),
        risk_acceptance_policy_digest=digest("risk-policy"),
        limitations=("Outdoor operation is excluded.",),
    )


def policy(
    fingerprints: tuple[str, ...] = (digest("failure-a"), digest("failure-b")),
    *,
    minimum_challenges: int = 1,
    required_kinds: tuple[ChallengeKind, ...] = (),
) -> AssurancePolicy:
    """Build an assurance policy for known failures."""
    return AssurancePolicy(
        policy_id="warehouse-release-policy",
        policy_version="1",
        required_failure_fingerprints=fingerprints,
        minimum_independent_challenges=minimum_challenges,
        required_independent_challenge_kinds=required_kinds,
    )


def promotion(
    fingerprint: str,
    index: int,
    *,
    ready: bool = True,
) -> RegressionPromotionReview:
    """Build a promotion review for one regression pack."""
    return RegressionPromotionReview(
        request_content_digest=digest(f"request-{index}"),
        source_failure_content_digest=digest(f"failure-content-{index}"),
        source_failure_fingerprint=fingerprint,
        regression_pack_digest=digest(f"pack-{index}"),
        status=(
            PromotionStatus.RELEASE_GATE_CANDIDATE_READY
            if ready
            else PromotionStatus.EVIDENCE_INSUFFICIENT
        ),
        requirement_results=(),
        unmet_requirements=(),
        limitations=("Promotion does not approve release.",),
    )


def campaign(
    promoted: RegressionPromotionReview,
    index: int,
    *,
    release_eligible: bool = True,
    gate_passed: bool | None = True,
) -> SimulationCampaignReport:
    """Build a completed or intentionally deficient campaign report."""
    gate = (
        None
        if gate_passed is None
        else RegressionGateReport(
            pack_id=f"pack-{index}",
            pack_version="1",
            observation_count=50,
            criterion_results=(),
            passed=gate_passed,
            limitations=("Gate applies only within the simulation envelope.",),
        )
    )
    return SimulationCampaignReport(
        plan_content_digest=digest(f"plan-{index}"),
        pack_content_digest=promoted.regression_pack_digest,
        result_count=50,
        work_item_evaluations=(),
        issues=(),
        execution_complete=True,
        release_evidence_eligible=release_eligible,
        regression_gate_report=gate,
        limitations=("Physical-world validity remains separate.",),
    )


def challenge(
    scope: ReleaseScope,
    evidence_id: str = "physical-validation",
    *,
    kind: ChallengeKind = ChallengeKind.PHYSICAL_VALIDATION,
    outcome: ChallengeOutcome = ChallengeOutcome.PASSED,
    independent: bool = True,
) -> ChallengeEvidence:
    """Build scoped challenge evidence."""
    return ChallengeEvidence(
        evidence_id=evidence_id,
        kind=kind,
        outcome=outcome,
        artifact_digest=digest(evidence_id),
        release_scope_digest=scope.content_digest(),
        independent_from_development=independent,
        limitations=("Sampled only declared warehouse layouts.",),
    )


def defeater(
    scope: ReleaseScope,
    severity: DefeaterSeverity,
    *,
    status: DefeaterStatus = DefeaterStatus.OPEN,
    defeater_id: str = "unmodeled-dust",
) -> AssuranceDefeater:
    """Build one scoped assurance defeater."""
    return AssuranceDefeater(
        defeater_id=defeater_id,
        description="Dust may degrade the depth sensor.",
        severity=severity,
        status=status,
        release_scope_digest=scope.content_digest(),
        evidence_digest=digest(f"evidence-{defeater_id}"),
        resolution_digest=(
            digest(f"resolution-{defeater_id}")
            if status is DefeaterStatus.RESOLVED
            else None
        ),
    )


def evidence_chain(
    *,
    promotion_ready: tuple[bool, bool] = (True, True),
    campaign_eligible: tuple[bool, bool] = (True, True),
    gate_passed: tuple[bool | None, bool | None] = (True, True),
):
    """Build two known-failure promotion and campaign chains."""
    fingerprints = (digest("failure-a"), digest("failure-b"))
    promotions = tuple(
        promotion(fingerprint, index, ready=promotion_ready[index])
        for index, fingerprint in enumerate(fingerprints)
    )
    campaigns = tuple(
        campaign(
            promotions[index],
            index,
            release_eligible=campaign_eligible[index],
            gate_passed=gate_passed[index],
        )
        for index in range(2)
    )
    links = tuple(
        RegressionEvidenceLink(
            failure_fingerprint=fingerprints[index],
            promotion_review_digest=promotions[index].content_digest(),
            campaign_report_digest=campaigns[index].content_digest(),
        )
        for index in range(2)
    )
    return fingerprints, promotions, campaigns, links


def dossier(
    *,
    scope: ReleaseScope | None = None,
    assurance_policy: AssurancePolicy | None = None,
    challenges: tuple[ChallengeEvidence, ...] | None = None,
    defeaters: tuple[AssuranceDefeater, ...] = (),
    chain=None,
):
    """Build a complete dossier and its linked concrete artifacts."""
    resolved_scope = scope or release_scope()
    resolved_chain = chain or evidence_chain()
    fingerprints, promotions, campaigns, links = resolved_chain
    resolved_policy = assurance_policy or policy(fingerprints)
    resolved_challenges = challenges or (challenge(resolved_scope),)
    artifact = ReleaseAssuranceDossier(
        dossier_id="policy-v18-release",
        dossier_version="1",
        release_scope=resolved_scope,
        policy=resolved_policy,
        regression_evidence=links,
        challenge_evidence=resolved_challenges,
        defeaters=defeaters,
        limitations=("Human release authority remains accountable.",),
    )
    return artifact, promotions, campaigns


def evaluate(built):
    """Evaluate a dossier fixture tuple."""
    artifact, promotions, campaigns = built
    return evaluate_release_assurance(
        artifact,
        promotion_reviews=promotions,
        campaign_reports=campaigns,
    )


def test_complete_dossier_is_supported_only_within_scope() -> None:
    report = evaluate(dossier())

    assert report.disposition is AssuranceDisposition.SUPPORTED_WITHIN_SCOPE
    assert report.blocking_reasons == ()
    assert report.restriction_reasons == ()
    assert report.independent_passing_challenge_count == 1
    assert len(report.regression_results) == 2
    assert all(item.regression_gate_passed for item in report.regression_results)


def test_failed_regression_gate_cannot_be_averaged_away() -> None:
    chain = evidence_chain(gate_passed=(True, False))
    scope = release_scope()
    challenges = tuple(
        challenge(
            scope,
            f"challenge-{index}",
            kind=kind,
        )
        for index, kind in enumerate(ChallengeKind)
    )

    report = evaluate(dossier(scope=scope, challenges=challenges, chain=chain))

    assert report.disposition is AssuranceDisposition.BLOCKED
    assert AssuranceReasonCode.REGRESSION_GATE_FAILED in report.blocking_reasons
    assert report.independent_passing_challenge_count == len(ChallengeKind)


def test_unqualified_promotion_blocks_assurance() -> None:
    report = evaluate(dossier(chain=evidence_chain(promotion_ready=(False, True))))

    assert report.disposition is AssuranceDisposition.BLOCKED
    assert AssuranceReasonCode.PROMOTION_NOT_QUALIFIED in report.blocking_reasons


def test_internally_inconsistent_ready_promotion_is_not_qualified() -> None:
    artifact, promotions, campaigns = dossier()
    changed_promotion = replace(
        promotions[0],
        unmet_requirements=(PromotionRequirement.ONE_MINIMAL_COUNTEREXAMPLE,),
    )
    changed_dossier = replace(
        artifact,
        regression_evidence=tuple(
            (
                replace(
                    item,
                    promotion_review_digest=changed_promotion.content_digest(),
                )
                if item.promotion_review_digest == promotions[0].content_digest()
                else item
            )
            for item in artifact.regression_evidence
        ),
    )

    report = evaluate_release_assurance(
        changed_dossier,
        promotion_reviews=(changed_promotion, promotions[1]),
        campaign_reports=campaigns,
    )

    assert report.disposition is AssuranceDisposition.BLOCKED
    assert AssuranceReasonCode.PROMOTION_NOT_QUALIFIED in report.blocking_reasons


@pytest.mark.parametrize(
    ("eligible", "gate", "reason"),
    [
        (False, True, AssuranceReasonCode.CAMPAIGN_NOT_RELEASE_ELIGIBLE),
        (True, None, AssuranceReasonCode.REGRESSION_GATE_MISSING),
    ],
)
def test_ineligible_or_missing_campaign_evidence_blocks(
    eligible: bool,
    gate: bool | None,
    reason: AssuranceReasonCode,
) -> None:
    report = evaluate(
        dossier(
            chain=evidence_chain(
                campaign_eligible=(eligible, True),
                gate_passed=(gate, True),
            )
        )
    )

    assert report.disposition is AssuranceDisposition.BLOCKED
    assert reason in report.blocking_reasons


def test_failed_challenge_blocks_even_when_regressions_pass() -> None:
    scope = release_scope()
    report = evaluate(
        dossier(
            scope=scope,
            challenges=(
                challenge(scope),
                challenge(
                    scope,
                    "failed-red-team",
                    kind=ChallengeKind.RED_TEAM,
                    outcome=ChallengeOutcome.FAILED,
                ),
            ),
        )
    )

    assert report.disposition is AssuranceDisposition.BLOCKED
    assert AssuranceReasonCode.CHALLENGE_FAILED in report.blocking_reasons


def test_independent_challenge_count_and_kind_are_non_compensatory() -> None:
    scope = release_scope()
    assurance_policy = policy(
        minimum_challenges=2,
        required_kinds=(ChallengeKind.RED_TEAM,),
    )
    challenges = (
        challenge(scope, "physical-1"),
        challenge(scope, "physical-2"),
    )

    report = evaluate(
        dossier(
            scope=scope,
            assurance_policy=assurance_policy,
            challenges=challenges,
        )
    )

    assert report.independent_passing_challenge_count == 2
    assert (
        AssuranceReasonCode.INDEPENDENT_CHALLENGE_INSUFFICIENT
        in report.blocking_reasons
    )


def test_development_team_evidence_does_not_count_as_independent() -> None:
    scope = release_scope()

    report = evaluate(
        dossier(
            scope=scope,
            challenges=(challenge(scope, independent=False),),
        )
    )

    assert report.independent_passing_challenge_count == 0
    assert report.disposition is AssuranceDisposition.BLOCKED


def test_inconclusive_challenge_restricts_otherwise_supported_scope() -> None:
    scope = release_scope()
    challenges = (
        challenge(scope),
        challenge(
            scope,
            "cross-backend",
            kind=ChallengeKind.CROSS_BACKEND_VALIDATION,
            outcome=ChallengeOutcome.INCONCLUSIVE,
        ),
    )

    report = evaluate(dossier(scope=scope, challenges=challenges))

    assert report.disposition is AssuranceDisposition.RESTRICTED
    assert AssuranceReasonCode.CHALLENGE_INCONCLUSIVE in report.restriction_reasons


def test_material_open_defeater_restricts_and_critical_open_blocks() -> None:
    scope = release_scope()
    material_report = evaluate(
        dossier(
            scope=scope,
            defeaters=(defeater(scope, DefeaterSeverity.MATERIAL),),
        )
    )
    critical_report = evaluate(
        dossier(
            scope=scope,
            defeaters=(defeater(scope, DefeaterSeverity.CRITICAL),),
        )
    )

    assert material_report.disposition is AssuranceDisposition.RESTRICTED
    assert (
        AssuranceReasonCode.MATERIAL_DEFEATER_OPEN
        in material_report.restriction_reasons
    )
    assert critical_report.disposition is AssuranceDisposition.BLOCKED
    assert (
        AssuranceReasonCode.CRITICAL_DEFEATER_OPEN in critical_report.blocking_reasons
    )


def test_resolved_critical_and_open_monitor_defeaters_do_not_block() -> None:
    scope = release_scope()
    report = evaluate(
        dossier(
            scope=scope,
            defeaters=(
                defeater(
                    scope,
                    DefeaterSeverity.CRITICAL,
                    status=DefeaterStatus.RESOLVED,
                    defeater_id="resolved-critical",
                ),
                defeater(
                    scope,
                    DefeaterSeverity.MONITOR,
                    defeater_id="monitored-drift",
                ),
            ),
        )
    )

    assert report.disposition is AssuranceDisposition.SUPPORTED_WITHIN_SCOPE
    assert report.open_defeater_ids == ("monitored-drift",)


def test_dossier_rejects_challenge_or_defeater_from_another_scope() -> None:
    scope = release_scope()
    foreign_digest = digest("foreign-scope")

    with pytest.raises(ValueError, match="challenge.*scope"):
        dossier(
            scope=scope,
            challenges=(
                replace(
                    challenge(scope),
                    release_scope_digest=foreign_digest,
                ),
            ),
        )
    with pytest.raises(ValueError, match="defeater.*scope"):
        dossier(
            scope=scope,
            defeaters=(
                replace(
                    defeater(scope, DefeaterSeverity.MATERIAL),
                    release_scope_digest=foreign_digest,
                ),
            ),
        )


def test_policy_fingerprint_coverage_must_be_exact() -> None:
    fingerprints, _, _, _ = evidence_chain()
    missing_policy = policy((fingerprints[0],))

    with pytest.raises(ValueError, match="exactly cover"):
        dossier(assurance_policy=missing_policy)


def test_supplied_artifacts_must_exactly_match_dossier_links() -> None:
    artifact, promotions, campaigns = dossier()

    with pytest.raises(ValueError, match="promotion reviews must exactly"):
        evaluate_release_assurance(
            artifact,
            promotion_reviews=promotions[:1],
            campaign_reports=campaigns,
        )
    with pytest.raises(ValueError, match="duplicate artifact digests"):
        evaluate_release_assurance(
            artifact,
            promotion_reviews=promotions,
            campaign_reports=campaigns + (campaigns[0],),
        )


def test_crossed_failure_or_pack_lineage_is_rejected() -> None:
    artifact, promotions, campaigns = dossier()

    changed_promotion = replace(
        promotions[0],
        source_failure_fingerprint=digest("crossed"),
    )
    promotion_dossier = replace(
        artifact,
        regression_evidence=tuple(
            (
                replace(
                    item,
                    promotion_review_digest=changed_promotion.content_digest(),
                )
                if item.promotion_review_digest == promotions[0].content_digest()
                else item
            )
            for item in artifact.regression_evidence
        ),
    )
    with pytest.raises(ValueError, match="failure fingerprint"):
        evaluate_release_assurance(
            promotion_dossier,
            promotion_reviews=(changed_promotion, promotions[1]),
            campaign_reports=campaigns,
        )
    changed_campaign = replace(
        campaigns[0],
        pack_content_digest=digest("different-pack"),
    )
    changed_dossier = replace(
        artifact,
        regression_evidence=tuple(
            (
                replace(
                    item,
                    campaign_report_digest=changed_campaign.content_digest(),
                )
                if item.campaign_report_digest == campaigns[0].content_digest()
                else item
            )
            for item in artifact.regression_evidence
        ),
    )
    with pytest.raises(ValueError, match="pack does not match"):
        evaluate_release_assurance(
            changed_dossier,
            promotion_reviews=promotions,
            campaign_reports=(changed_campaign, campaigns[1]),
        )


def test_dossier_and_report_are_order_invariant() -> None:
    artifact, promotions, campaigns = dossier()
    reordered = replace(
        artifact,
        regression_evidence=tuple(reversed(artifact.regression_evidence)),
        challenge_evidence=tuple(reversed(artifact.challenge_evidence)),
        limitations=tuple(reversed(artifact.limitations)),
    )

    first = evaluate_release_assurance(
        artifact,
        promotion_reviews=promotions,
        campaign_reports=campaigns,
    )
    second = evaluate_release_assurance(
        reordered,
        promotion_reviews=tuple(reversed(promotions)),
        campaign_reports=tuple(reversed(campaigns)),
    )

    assert artifact == reordered
    assert artifact.content_digest() == reordered.content_digest()
    assert first == second
    assert first.content_digest() == second.content_digest()


def test_report_preserves_epistemic_and_authority_boundaries() -> None:
    report = evaluate(dossier())
    limitations = " ".join(report.limitations)

    assert "not authorization to deploy" in limitations
    assert "exact system artifacts" in limitations
    assert "sim-to-real uncertainty" in limitations
    assert "cannot be averaged away" in limitations
    assert "Human release authority remains accountable." in limitations
    assert "Outdoor operation is excluded." in limitations


def test_scope_policy_and_defeater_validation() -> None:
    scope = release_scope()

    with pytest.raises(ValueError, match="non-negative integer"):
        replace(policy(), minimum_independent_challenges=-1)
    with pytest.raises(ValueError, match="resolution digest"):
        replace(
            defeater(scope, DefeaterSeverity.CRITICAL),
            status=DefeaterStatus.RESOLVED,
        )
    with pytest.raises(ValueError, match="must not declare"):
        replace(
            defeater(scope, DefeaterSeverity.CRITICAL),
            resolution_digest=digest("premature"),
        )
