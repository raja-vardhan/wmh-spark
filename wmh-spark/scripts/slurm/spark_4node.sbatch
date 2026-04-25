#!/bin/bash
#SBATCH --job-name=wmh-spark-4node
#SBATCH --account=soc-kp                  # change to your CHPC account
#SBATCH --partition=soc-kp
#SBATCH --nodes=5                         # 1 master/driver + 4 workers
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/wmh-%j.out
#SBATCH --error=logs/wmh-%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Module loads. Pin every version. CHPC defaults change without warning.
# ---------------------------------------------------------------------------
module purge
module load jdk/17.0.7
module load python/3.10
module load spark/3.5.1
module load singularity/3.11

# ---------------------------------------------------------------------------
# Project env
# ---------------------------------------------------------------------------
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/wmh-spark}
source $PROJECT_ROOT/.venv/bin/activate
export PYTHONPATH=$PROJECT_ROOT/src:$PYTHONPATH

# ---------------------------------------------------------------------------
# Spark standalone cluster bring-up
# ---------------------------------------------------------------------------
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
SPARK_MASTER_HOST=${NODES[0]}
WORKER_HOSTS=("${NODES[@]:1}")

export SPARK_MASTER_HOST
export SPARK_MASTER_PORT=7077
export SPARK_LOG_DIR=$PROJECT_ROOT/logs/spark-$SLURM_JOB_ID
export SPARK_WORKER_DIR=$SCRATCH/spark-work-$SLURM_JOB_ID
mkdir -p $SPARK_LOG_DIR $SPARK_WORKER_DIR

echo "Starting Spark master on $SPARK_MASTER_HOST"
ssh $SPARK_MASTER_HOST "$SPARK_HOME/sbin/start-master.sh"

for w in "${WORKER_HOSTS[@]}"; do
    echo "Starting Spark worker on $w"
    ssh $w "$SPARK_HOME/sbin/start-worker.sh \
        spark://$SPARK_MASTER_HOST:7077 \
        --cores 4 --memory 28g \
        --work-dir $SPARK_WORKER_DIR"
done

# Give workers time to register with the master before submitting.
sleep 15

# ---------------------------------------------------------------------------
# Submit the pipeline
# ---------------------------------------------------------------------------
spark-submit \
    --master spark://$SPARK_MASTER_HOST:7077 \
    --deploy-mode client \
    --driver-memory 16g \
    --executor-memory 28g \
    --executor-cores 4 \
    --total-executor-cores 16 \
    --conf spark.local.dir=$SCRATCH/spark-tmp \
    --py-files $PROJECT_ROOT/src/wmh_spark.zip \
    $PROJECT_ROOT/scripts/run_pipeline.py \
    --config $PROJECT_ROOT/configs/chpc_4node.yaml

# ---------------------------------------------------------------------------
# Tear-down
# ---------------------------------------------------------------------------
for w in "${WORKER_HOSTS[@]}"; do
    ssh $w "$SPARK_HOME/sbin/stop-worker.sh" || true
done
ssh $SPARK_MASTER_HOST "$SPARK_HOME/sbin/stop-master.sh" || true

echo "Done."
