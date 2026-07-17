"""Round-trip and consistency tests for the iso-obs-schemas contract.

These tests exercise the guarantees other packages rely on:

* every core model and the trace event serialize to JSON and re-parse to an
  equal object (no lossy fields, no non-deterministic serialization);
* the shared enums contain their spec'd members;
* the id validators reject mismatched prefixes;
* a spec-example ``constraint.violated`` event validates and its typed payload
  parses.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from iso_obs_schemas import (
    BaseEvent,
    ConstraintViolationPayload,
    Environment,
    EnvironmentVersion,
    EvaluationSuite,
    EventType,
    FailureCategory,
    Gate,
    GateRequirement,
    Invariant,
    MetricValue,
    MetricValueKind,
    PerturbationFamily,
    PerturbationSpec,
    Project,
    ReliabilityReport,
    Run,
    RunStatus,
    Scenario,
    ScenarioVersion,
    Severity,
    System,
    SystemType,
    SystemVersion,
    Workspace,
    generate_id,
)
from iso_obs_schemas.ids import IdPrefix
from iso_obs_schemas.metrics import ComparisonOperator
from iso_obs_schemas.reports import Finding


def _new(prefix: IdPrefix) -> str:
    """Return a fresh valid id for the given prefix."""
    return generate_id(prefix)


def _core_instances() -> list[BaseModel]:
    """Construct one instance of each core model with valid ids."""
    ws = _new(IdPrefix.WORKSPACE)
    prj = _new(IdPrefix.PROJECT)
    sys_id = _new(IdPrefix.SYSTEM)
    env_id = _new(IdPrefix.ENVIRONMENT)
    scn_id = _new(IdPrefix.SCENARIO)
    return [
        Workspace(id=ws, name="Acme", slug="acme"),
        Project(id=prj, workspace_id=ws, name="Robotics", slug="robotics"),
        System(
            id=sys_id,
            project_id=prj,
            name="Grasp policy",
            system_type=SystemType.POLICY,
        ),
        SystemVersion(
            id=_new(IdPrefix.SYSTEM_VERSION), system_id=sys_id, version="1.2.0"
        ),
        Environment(id=env_id, project_id=prj, name="MuJoCo arm"),
        EnvironmentVersion(
            id=_new(IdPrefix.ENVIRONMENT_VERSION),
            environment_id=env_id,
            version="2024.1",
        ),
        Scenario(id=scn_id, environment_id=env_id, name="Cluttered bin"),
        ScenarioVersion(
            id=_new(IdPrefix.SCENARIO_VERSION),
            scenario_id=scn_id,
            version="3",
            perturbations=[
                PerturbationSpec(
                    family=PerturbationFamily.OBSERVATION_NOISE,
                    name="gaussian",
                    parameters={"sigma": 0.05},
                    start_step=0,
                    end_step=100,
                )
            ],
        ),
        EvaluationSuite(
            id=_new(IdPrefix.EVALUATION_SUITE),
            project_id=prj,
            name="Nightly",
            scenario_version_ids=[_new(IdPrefix.SCENARIO_VERSION)],
            seeds=[1, 2, 3],
        ),
        Run(
            id=_new(IdPrefix.RUN),
            workspace_id=ws,
            project_id=prj,
            suite_execution_id=_new(IdPrefix.EVALUATION_SUITE),
            system_version_id=_new(IdPrefix.SYSTEM_VERSION),
            environment_version_id=_new(IdPrefix.ENVIRONMENT_VERSION),
            scenario_version_id=_new(IdPrefix.SCENARIO_VERSION),
            seed=7,
            status=RunStatus.COMPLETED,
        ),
    ]


@pytest.mark.parametrize("instance", _core_instances(), ids=lambda m: type(m).__name__)
def test_core_models_round_trip(instance: BaseModel) -> None:
    """Each core model survives a JSON dump/re-parse unchanged."""
    dumped = instance.model_dump_json()
    reparsed = type(instance).model_validate_json(dumped)
    assert reparsed == instance


def _sample_event() -> BaseEvent:
    """Build a spec-example ``constraint.violated`` trace event."""
    payload = ConstraintViolationPayload(
        constraint="joint_torque_limit",
        observed_value=42.5,
        threshold=40.0,
        severity=Severity.CRITICAL,
        first_occurrence=True,
    )
    return BaseEvent(
        event_id=_new(IdPrefix.EVENT),
        workspace_id=_new(IdPrefix.WORKSPACE),
        project_id=_new(IdPrefix.PROJECT),
        suite_id=_new(IdPrefix.EVALUATION_SUITE),
        run_id=_new(IdPrefix.RUN),
        system_id=_new(IdPrefix.SYSTEM),
        system_version=_new(IdPrefix.SYSTEM_VERSION),
        environment_id=_new(IdPrefix.ENVIRONMENT),
        environment_version=_new(IdPrefix.ENVIRONMENT_VERSION),
        scenario_id=_new(IdPrefix.SCENARIO),
        monotonic_ns=1_234_567_890,
        step=128,
        event_type=EventType.CONSTRAINT_VIOLATED,
        source="sdk",
        payload=payload.model_dump(),
    )


def test_event_round_trip() -> None:
    """A trace event survives a JSON dump/re-parse unchanged."""
    event = _sample_event()
    reparsed = BaseEvent.model_validate_json(event.model_dump_json())
    assert reparsed == event


def test_constraint_violation_event_matches_spec_example() -> None:
    """The sample constraint.violated event carries a valid typed payload."""
    event = _sample_event()
    assert event.event_type is EventType.CONSTRAINT_VIOLATED
    payload = ConstraintViolationPayload.model_validate(event.payload)
    assert payload.constraint == "joint_torque_limit"
    assert payload.observed_value > payload.threshold
    assert payload.severity is Severity.CRITICAL


def test_enum_membership() -> None:
    """Spec'd enum members are present and carry their dotted/string values."""
    assert EventType.CONSTRAINT_VIOLATED.value == "constraint.violated"
    assert EventType.RUN_STARTED.value == "run.started"
    assert RunStatus.COMPLETED in RunStatus
    assert "unsafe_action" in {c.value for c in FailureCategory}
    assert set(Severity) == {Severity.INFO, Severity.WARNING, Severity.CRITICAL}


def test_id_prefix_validation_rejects_wrong_prefix() -> None:
    """A workspace id in a project field is rejected at parse time."""
    with pytest.raises(ValidationError):
        Project(
            id=generate_id(IdPrefix.WORKSPACE),  # wrong prefix for a project id
            workspace_id=generate_id(IdPrefix.WORKSPACE),
            name="Bad",
            slug="bad",
        )


def test_metric_value_variant_enforced() -> None:
    """A scalar metric without a scalar value is rejected."""
    ok = MetricValue(key="success_rate", kind=MetricValueKind.SCALAR, scalar_value=0.9)
    assert ok.passed is None
    with pytest.raises(ValidationError):
        MetricValue(key="success_rate", kind=MetricValueKind.SCALAR)


def test_gate_and_report_round_trip() -> None:
    """Gate and reliability report models round-trip through JSON."""
    gate = Gate(
        name="ship-gate",
        requirements=[
            GateRequirement(
                metric="success_rate", operator=ComparisonOperator.GTE, value=0.95
            )
        ],
    )
    assert Gate.model_validate_json(gate.model_dump_json()) == gate

    invariant = Invariant(name="stay-upright", expression="pole_angle < 0.2")
    assert Invariant.model_validate_json(invariant.model_dump_json()) == invariant

    report = ReliabilityReport(
        id=_new(IdPrefix.REPORT),
        workspace_id=_new(IdPrefix.WORKSPACE),
        project_id=_new(IdPrefix.PROJECT),
        suite_execution_id=_new(IdPrefix.EVALUATION_SUITE),
        title="Nightly reliability",
        findings=[
            Finding(
                statement="Policy violates torque limits under observation noise.",
                supporting_run_ids=[_new(IdPrefix.RUN)],
                confidence=0.8,
            )
        ],
    )
    assert ReliabilityReport.model_validate_json(report.model_dump_json()) == report
