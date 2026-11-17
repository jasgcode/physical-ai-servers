#!/bin/bash
# Long-lived cosmos_edge_server + Tailscale Funnel as a Slurm batch job. Wraps the
# same scripts/establish_server.sh used for interactive runs.
#
# Prerequisite (one-time): pixi run setup, scripts/install_tailscale.sh, and
# ~/.tailscale-bfaq-authkey on disk.
#
# Submit from the project root. Slurm resolves the relative --output path
# against the SUBMIT directory, so logs/ must exist before submitting:
#   mkdir -p logs && sbatch slurm/run.sbatch
#   tail -f logs/slurm-<jobid>.out
#
# More than one GPU: raise --gres and submit with COSMOS_EDGE_REPLICAS=<n> (max 3;
# sbatch forwards the submission environment into the job by default).
# Stop the server: scancel <jobid> (scripts/launch.sh traps SIGTERM).
#
# NOTE: replace the account/partition placeholders and confirm the gres syntax
# and walltime against your allocation before the first submit.

#SBATCH --job-name=cosmos-edge-server
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# Slurm copies the batch script to a spool dir, so BASH_SOURCE does not point
# into the project; SLURM_SUBMIT_DIR does.
PROJECT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT"
mkdir -p logs

export PATH="${HOME}/.local/bin:${HOME}/bin:${HOME}/.pixi/bin:${PATH}"

echo "[sbatch] PROJECT=$PROJECT HOST=$(hostname -s) JOB=${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=name --format=csv,noheader || true

exec pixi run establish
