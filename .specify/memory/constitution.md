<!--
Sync Impact Report
==================
Version change: 1.0.0 → 1.1.0
Modified principles: None renamed or removed
Added sections:
  - Development Phases & Data Strategy (new — local scaffolding → CHPC deployment workflow,
    smoke test dataset under datasets/, Kaggle dataset progression)
Modified sections:
  - Infrastructure & Environment Constraints — updated Data Source entry to reflect two-phase
    data strategy (local smoke test → Kaggle full validation)
  - Development Workflow & Quality Gates — noted smoke-test exemption for DSC gate
Removed sections: None
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ aligned
  - .specify/templates/spec-template.md ✅ aligned
  - .specify/templates/tasks-template.md ✅ aligned
  - .specify/templates/commands/ — directory not present, N/A
Deferred TODOs: None
-->

# WMH-Spark Constitution

## Core Principles

### I. Distributed-First Architecture (NON-NEGOTIABLE)

All data transformation, feature extraction, and ML inference MUST execute as distributed PySpark
operations across worker nodes. Driver-side loops over voxels or subjects are strictly prohibited.
The canonical data model is a flattened 2D PySpark DataFrame where each row represents one
(X, Y, Z) voxel — the 3D→2D transformation is the architectural boundary that enables
distribution. Any operation that cannot be expressed as a DataFrame transformation MUST be
escalated for architectural review before implementation begins.

**Rationale**: The entire project goal is to replace the sequential MATLAB architecture. Any
reversion to sequential patterns negates the scalability objective and defeats the purpose of
the migration.

### II. Containerized Pre-Processing

The skull-stripping step (HD-BET) MUST run inside a Docker container. No bare-metal or
environment-specific tool dependencies are permitted outside a container boundary. Docker image
tags MUST be pinned (no `latest`) to guarantee identical output across runs and worker nodes.
Container execution MUST be orchestrated by the pipeline, not invoked manually.

**Rationale**: Brain extraction results vary with tool version. Containerization ensures that
all subjects across all nodes are processed identically, preventing confounding variables from
entering the classifier.

### III. Memory-Safe Partitioning (NON-NEGOTIABLE)

Every PySpark shuffle or join operation MUST specify an explicit partition count tuned to the
16 GB RAM cap per worker node. Partition strategy (count, key column, repartition vs. coalesce)
MUST be documented at each call site where it is set. Uncontrolled shuffles that risk OOM errors
constitute a blocking defect and MUST be resolved before the feature is considered complete.
Driver-node operations MUST be limited to coordination tasks; no large dataset materializations
on the driver are permitted.

**Rationale**: The cluster has strict memory limits (16 GB per worker, 32 GB driver). OOM errors
during shuffle are the most common failure mode for voxel-scale neuroimaging DataFrames. This
principle makes memory safety a first-class design constraint, not an afterthought.

### IV. Accuracy-Gated Output

No pipeline output on the full Kaggle validation dataset is considered valid unless the automated
evaluation module reports a Dice Similarity Coefficient (DSC) > 0.85 against expert Silver
Standard masks. The evaluation step is mandatory and MUST run automatically at the end of every
full pipeline execution — it is not an optional or manual post-hoc step. Any code path that
bypasses evaluation is prohibited.

**Smoke-test exception**: During local scaffolding runs against the `datasets/` smoke-test data,
the DSC gate does not apply because no paired expert masks are available. Smoke runs MUST instead
assert pipeline structural correctness: output DataFrame shape, binary label range, and benchmark
log generation.

**Rationale**: The project replaces a validated clinical tool (MATLAB UBO Detector). Without a
quantitative accuracy gate on the full dataset, there is no evidence the distributed
reimplementation produces clinically meaningful results.

### V. Scalability & Benchmarking (NON-NEGOTIABLE)

Every end-to-end pipeline run MUST log: per-subject processing time, cluster throughput
(subjects processed per hour), and the scalability factor as worker count scales from 1 to 4
nodes. Benchmark results MUST be stored in a structured format (JSON or CSV) for automated
comparison. Performance against the legacy MATLAB serial baseline MUST be reported. A run
that does not produce benchmark output is considered incomplete.

**Rationale**: Proving near-linear scalability and reduced latency vs. MATLAB is an explicit
project deliverable. Without instrumented benchmarking, the scalability claim cannot be
substantiated.

### VI. Test-Driven Correctness

Unit tests MUST cover: feature extraction math (T1 intensity, FLAIR intensity, T1/FLAIR ratio,
spatial prior mapping), classification output shape and binary label constraints, post-processing
volume threshold logic, and DSC calculation. Integration tests MUST verify the full subject
pipeline on at least one real NIfTI scan pair (T1 + FLAIR + mask). Tests MUST be written and
confirmed failing before implementation begins (Red-Green-Refactor). The test suite MUST be
executable in a single command without manual setup.

**Rationale**: Neuroimaging pipelines fail silently — incorrect voxel math produces plausible
but wrong outputs. Test-driven development catches arithmetic errors before they propagate to
the classifier and corrupt accuracy results.

## Infrastructure & Environment Constraints

The pipeline ultimately targets the University of Utah CHPC simulated cloud environment. The
fixed production cluster configuration is:

- **Driver Node**: 1 node, 32 GB RAM — orchestration only, no large data materialization
- **Worker Nodes**: 4 nodes, 16 GB RAM each — all distributed computation
- **Tech Stack**: Python 3.9, PySpark 3.5, Docker, Spark MLlib
- **Data Sources**: Two-phase (see Development Phases & Data Strategy below):
  - Phase 1 (smoke): Local `datasets/` directory — T1_RMS.nii.gz, FLAIR.nii.gz,
    T2_HippocampalSubfields.nii.gz, T2_caipi.nii.gz
  - Phase 2 (full validation): Kaggle WMH Segmentation Challenge dataset (paired T1, FLAIR,
    expert masks in NIfTI format), deployed to CHPC storage volume
- **Storage**: Network-mounted volume on CHPC; data MUST be read with partition-aware strategies
  to prevent driver bottlenecks during I/O

No deviation from the Python 3.9 / PySpark 3.5 stack is permitted without an amendment to this
constitution. All Docker images used in containerized steps MUST be compatible with the CHPC
node OS environment.

## Development Phases & Data Strategy

Development follows a local-first, CHPC-deploy model with two explicit phases:

### Phase 1 — Local Scaffolding (Smoke Tests)

All initial code scaffolding and structural validation occurs on the local development machine.

- **Smoke test dataset**: `datasets/` in the repository root. Contains:
  - `T1_RMS.nii.gz` — T1-weighted scan (Root Mean Square)
  - `FLAIR.nii.gz` — FLAIR scan
  - `T2_HippocampalSubfields.nii.gz` — T2 hippocampal subfields scan
  - `T2_caipi.nii.gz` — T2 CAIPIRINHA scan
- **Purpose**: Verify pipeline structure, I/O correctness, DataFrame shape, and feature
  extraction logic on real NIfTI files before any CHPC deployment.
- **Output review**: Pipeline run outputs (logs, benchmark JSON, partial predictions) MUST be
  inspectable locally to confirm the scaffold is functioning before the code is pushed.
- **No expert masks in smoke data**: DSC evaluation is skipped; structural assertions apply
  instead (see Principle IV smoke-test exception).

### Phase 2 — CHPC Deployment (Full Validation)

Once the local scaffold is sufficiently complete and smoke tests pass:

1. Push the branch to GitHub.
2. Clone on CHPC and configure the network-mounted data volume.
3. Run the full pipeline against the Kaggle WMH dataset on the 4-worker cluster.
4. DSC > 0.85 gate and scalability benchmarks MUST be satisfied.

**Transition criteria**: The move from Phase 1 to Phase 2 occurs when all core pipeline modules
(I/O, pre-processing, feature extraction, classification, post-processing, evaluation) have
passing unit tests and at least one successful smoke-test end-to-end run locally.

### Data Progression

```
Local datasets/  →  GitHub push  →  CHPC clone  →  Kaggle WMH dataset
(smoke tests)       (scaffold        (full run        (production
                     complete)        on cluster)       validation)
```

## Development Workflow & Quality Gates

**Branch strategy**: All features are developed on named branches (`feature/<name>`). No direct
commits to `master` without a passing test suite.

**Mandatory quality gates** (all MUST pass before a feature is merged):

1. Unit test suite passes with zero failures
2. At least one successful end-to-end smoke run against local `datasets/` (Phase 1) or
   integration test against a real NIfTI subject (Phase 2)
3. Partition strategy is documented at all shuffle/join call sites
4. Evaluation module reports DSC on validation subject(s) — Phase 2 only; smoke-test exemption
   applies during Phase 1
5. Benchmark log is generated and stored

**Code review** MUST verify compliance with all 6 Core Principles. Any PR that introduces a
driver-side voxel loop, an untagged Docker image, an undocumented partition strategy, or skips
the evaluation step (in Phase 2) is non-compliant and MUST be rejected.

**Complexity justification**: Any architectural decision that deviates from the standard
DataFrame + MLlib + Docker stack MUST be documented in the plan's Complexity Tracking table
with an explicit rationale for why the simpler approach is insufficient.

## Governance

This constitution supersedes all other development practices for the WMH-Spark project.
Amendments require: (1) written description of the change and motivation, (2) version bump
per semantic versioning rules (MAJOR for principle removals/redefinitions, MINOR for new
principles or sections, PATCH for clarifications), (3) update to all dependent templates and
this Sync Impact Report.

**Versioning policy**:
- MAJOR: Removing or fundamentally redefining an existing principle
- MINOR: Adding a new principle, section, or materially expanding guidance
- PATCH: Wording clarifications, typo fixes, non-semantic refinements

All PRs and code reviews MUST verify compliance with the six Core Principles. Violations found
during review constitute blocking defects. Refer to `CLAUDE.md` and the current plan artifact
for runtime development guidance.

**Version**: 1.1.0 | **Ratified**: 2026-04-25 | **Last Amended**: 2026-04-25
