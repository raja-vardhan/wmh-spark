# Architecture and design rationale

This document explains *why* the pipeline is shaped the way it is. The code
is the implementation; this is the reasoning behind it.

## Why subject-level parallelism, not voxel-level

A naive Spark port of an MRI pipeline turns each voxel into a row of a
DataFrame. For a 1mm MNI volume that's ~7 million rows per subject, ~700
million rows per 100-subject cohort, and every row carries a feature
vector. Spark serializes every row through Pickle/Arrow, shuffles them
across the network for joins or aggregations, and reconstructs them on
the receiving worker.

The cost of all that machinery is wasted because the operation we actually
want to perform — extracting features from a 3D image — has no shuffle
need. Every voxel's feature is computed from its own neighborhood; nothing
crosses subject boundaries until training time.

The right pattern is **one subject = one row**. The row carries a path,
not the volume. A worker pulls the volume from shared storage, runs the
NumPy pipeline locally, and writes results to shared storage. Spark
schedules the work, restarts failed tasks, and tracks lineage — which is
exactly what we want from it. We pay no serialization cost on volumes
because they never travel through Spark's wire protocol.

## Where Spark's DataFrame *does* earn its keep

Training. Pooling subsampled voxels from 100+ subjects into one
classifier requires shuffling samples across workers, which is exactly
the workload Spark MLlib is built for. We use a DataFrame here, train an
MLlib RandomForestClassifier, and the distributed shuffle pays for
itself because the alternative is collecting all samples to the driver.

This split — RDD-of-subjects everywhere except the training stage — is
the architectural insight that makes Spark a good fit for this workload
rather than a bad fit.

## Why we keep a sklearn mirror of the trained RF

MLlib's `RandomForestClassificationModel.transform()` requires building a
DataFrame for every prediction call. At single-subject inference time
this costs more than the prediction itself. We re-fit a sklearn
`RandomForestClassifier` on the same training data and broadcast that to
workers. Both models see identical inputs, so accuracy is statistically
equivalent; the sklearn mirror is purely an ergonomic optimization for
the inference path.

For very large cohorts where the pooled training data exceeds driver
memory, this re-fit isn't viable. The fallback path
(`predict_rf_spark_distributed`) uses MLlib's transform directly.

## Why skull-stripping is a separate pre-stage

HD-BET on CPU takes ~5 minutes per volume. SynthStrip takes ~30 seconds.
If we benchmark "Spark vs MATLAB" with HD-BET inside the timed loop, we
are mostly measuring HD-BET, not Spark. We separate stripping into its
own stage with its own timing, so the headline benchmark measures the
actual MATLAB-replacement work: feature extraction, training, inference,
and evaluation.

## Why we evaluate against expert ground truth, not UBO

The original proposal used UBO output as the "Silver Standard." This
creates a circular validation: a perfect reproduction of UBO would score
100% even if UBO is itself flawed. The Kaggle and WMH Challenge datasets
both ship expert-annotated masks. We use those as primary reference and
report UBO agreement as a secondary metric. This way:

- If we beat UBO against expert ground truth, that's a contribution.
- If we match UBO against expert ground truth, that's reproduction with
  a faster runtime, which is also a contribution.
- If both we and UBO score poorly, we're not falsely claiming success.

## Why three classifiers, not two

k-NN vs Random Forest is a forced comparison: k-NN is bad on this data
and Random Forest is good, but the gap mostly reflects the algorithm's
fit to tabular features rather than anything specific about WMH. Adding
XGBoost gives us a meaningful third point: it's the modern default for
tabular data and is what a reviewer would ask why we didn't include.

If RF beats k-NN but loses to XGBoost, that's a more honest finding than
"RF beats k-NN." If RF wins all three, the story is stronger because we
showed it under fair comparison.

## Why we cap voxels per subject in sampling

WMH burden varies enormously across subjects: a young control may have
~100 lesion voxels; a CADASIL patient may have 10,000+. Without a cap,
training samples are dominated by a handful of high-burden subjects, and
the model overfits their specific texture. Capping per-subject
contributions ensures every subject in the training cohort exerts
roughly equal influence on the classifier.

## Why driver-only k-NN

k-NN inference requires the entire training set at every prediction
call. There is no distributed inference path that beats single-node for
a fixed model size. Putting k-NN on Spark would mean broadcasting the
training set to every worker — which works but doesn't accelerate
anything compared to single-node sklearn with multi-threaded prediction.
We're explicit about this: k-NN is a baseline for accuracy comparison,
not a scalability candidate. The whole point of the project is that
k-NN can't scale; we're not pretending otherwise.

## Why no dropping registration warps to the driver

Volumes never leave their owning worker except as paths. Registration
results are paths to NIfTI files in shared storage; the driver receives
only metadata (path + timing + success flag). This keeps driver memory
bounded regardless of cohort size and is a hard requirement for scaling
to UK Biobank-class cohorts (50,000+ subjects).

## Failure handling

Every stage wraps its work in `try/except` and returns a result object
with `success: bool`. A bad subject (missing file, registration failure,
NaN in features) doesn't kill the job — it just doesn't contribute to
the next stage. Stage drivers filter on `result.success` before passing
work downstream. This is critical at cohort scale because the
probability of *some* subject failing approaches 1 as cohort size grows.

## Reproducibility manifest

- All seeds in `config.py`, dumped alongside outputs as `resolved_config.yaml`.
- Train/val/test splits committed to manifest.parquet, derived from a
  fixed seed and a fixed subject ordering.
- Dataset SHA-256 checksums emitted by `io_utils.write_checksum_manifest`.
- Pinned Python and Spark versions in `requirements.txt`.
- A reviewer can rerun any experiment by checking out the commit hash,
  recreating the venv, and running `spark-submit scripts/run_pipeline.py`.

## What we deliberately did *not* do

- We did not parallelize feature extraction *within* a subject. NumPy's
  vectorized operations on a single volume are already faster than the
  serialization cost of shipping voxel chunks across workers. Adding
  intra-subject parallelism would slow the pipeline down.
- We did not use Pandas UDFs for feature extraction. They look
  attractive but force a row-by-row iteration that loses the SIMD
  vectorization advantage of NumPy on full volumes.
- We did not implement custom MLlib estimators. The bundled
  `RandomForestClassifier` is well-tested and good enough for the
  evaluation criteria; writing our own would buy nothing.
- We did not benchmark on cohorts smaller than 50 subjects. JVM startup
  + Spark scheduling overhead dominate at that scale and produce
  misleading scaling curves.
