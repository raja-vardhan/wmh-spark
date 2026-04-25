# CHPC environment setup

This document covers bringing up the Spark cluster on the University of
Utah CHPC. It assumes you have a CHPC account, scratch directory access,
and an allocation on a partition that can grant 5 nodes (1 master + 4
workers) for the durations you need.

## One-time setup

### 1. Project directory

Clone the repo into your home directory:

```bash
cd $HOME
git clone <repo-url> wmh-spark
cd wmh-spark
```

### 2. Python environment

CHPC's `python/3.10` module is sufficient. Build a venv inside the project
directory so it persists across sessions:

```bash
module load python/3.10
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Pinned versions in `requirements.txt` are critical -- PySpark 3.5.x
requires Java 8/11/17 specifically, and CHPC's default Java is sometimes
mismatched. Verify with:

```bash
java -version   # expect 17.x
python -c "import pyspark; print(pyspark.__version__)"
```

### 3. Singularity images for skull stripping

Build SynthStrip locally (or on a node that has `singularity build`):

```bash
mkdir -p $HOME/sif
singularity build $HOME/sif/synthstrip.sif docker://freesurfer/synthstrip:1.6
```

For HD-BET on GPU partitions, use the existing image registry:

```bash
singularity build $HOME/sif/hdbet.sif docker://mic-dkfz/hd-bet:latest
```

### 4. MNI template

Place the MNI152 1mm brain template in a stable location:

```bash
mkdir -p $HOME/templates
# from FSL's data directory or downloaded from https://nist.mni.mcgill.ca
cp /uufs/chpc.utah.edu/sys/installdir/fsl/std/MNI152_T1_1mm_brain.nii.gz \
    $HOME/templates/
```

Update `configs/chpc_4node.yaml` if your paths differ.

### 5. Data

Stage your data into `$SCRATCH`:

```bash
mkdir -p $SCRATCH/wmh/data
# rsync or cp your subject directories here
python scripts/make_manifest.py \
    --data-dir $SCRATCH/wmh/data \
    --out      $SCRATCH/wmh/manifest.parquet
```

## Per-job submission

Edit the SLURM script if your account or partition differs:

```bash
sed -i 's/soc-kp/your-account/' scripts/slurm/spark_4node.sbatch
```

Then submit:

```bash
sbatch scripts/slurm/spark_4node.sbatch
squeue -u $USER     # watch the job
```

Logs appear in `logs/wmh-<jobid>.out` and `logs/wmh-<jobid>.err`.

## Web UI access

The Spark UI runs on port 4040 of the driver node. To view it:

```bash
# in another terminal, set up SSH tunneling
ssh -L 4040:<driver-node>:4040 <chpc-login>
# then open http://localhost:4040 in your browser
```

The driver node is the first node in the `SLURM_JOB_NODELIST`; you can
get it from `squeue -u $USER -o "%N"`.

## Troubleshooting

**Job fails immediately with "Java not found":** the `module load jdk/17.0.7`
line in the sbatch script likely needs adjustment for your CHPC account's
default modules. Run `module spider jdk` to find the available JDK and
update the sbatch.

**Workers don't register with master:** firewall rules can block port
7077 between nodes. Check by running `nc -zv <master-host> 7077` from a
worker.

**Out of memory during training:** the pooled training set may be too
large for the driver. Reduce `sampling.max_voxels_per_subject` in
`configs/chpc_4node.yaml` or switch to `predict_rf_spark_distributed`
in `pipeline.py`'s training stage.

**SynthStrip fails inside Singularity:** ensure your `$SCRATCH` is bind-
mounted into the container. Add `--bind $SCRATCH:$SCRATCH` to the
singularity exec call in `stripping.py` if it isn't already.
