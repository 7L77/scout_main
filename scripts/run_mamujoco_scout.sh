#!/usr/bin/env bash
# Run the OMIGA MA-MuJoCo experiments with the current unified main.py entry.
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_dir"

SCENARIOS=${SCENARIOS:-"2ant 3hopper 6halfcheetah"}
DATASETS=${DATASETS:-"Medium"}
SEED_START=${SEED_START:-0}
SEED_END=${SEED_END:-0}
OFFLINE_STEPS=${OFFLINE_STEPS:-500000}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
LOG_INTERVAL=${LOG_INTERVAL:-5000}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
NUM_EVAL_EPISODES=${NUM_EVAL_EPISODES:-10}
DATA_DIR=${DATA_DIR:-"$repo_dir/data"}
SAVE_DIR=${SAVE_DIR:-"$repo_dir/exp"}
PROJECT_NAME=${PROJECT_NAME:-"SCOUT-MAMuJoCo"}
VGF_LR=${VGF_LR:-0.02}
DRY_RUN=${DRY_RUN:-0}
PYTHON=${PYTHON:-${PYTHON_BIN:-python}}

export CUDA_VISIBLE_DEVICES=${GPU:-0}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=${WANDB_MODE:-offline}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH:-"$HOME/.mujoco/mujoco210"}
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Appendix B, Table 5: train steps and the test-step column to read in eval.csv.
# main.py evaluates all test budgets: 0, 1, 3, 5, 10.
for scenario in $SCENARIOS; do
  for dataset in $DATASETS; do
    case "$scenario:$dataset" in
      2ant:Expert|2ant:Medium)                  train_steps=2; test_steps=0 ;;
      2ant:Medium-Expert|2ant:Medium-Replay)    train_steps=0; test_steps=0 ;;
      3hopper:Expert)                         train_steps=2; test_steps=10 ;;
      3hopper:Medium-Expert|3hopper:Medium)     train_steps=2; test_steps=3 ;;
      3hopper:Medium-Replay)                  train_steps=3; test_steps=0 ;;
      6halfcheetah:Expert)                    train_steps=2; test_steps=3 ;;
      6halfcheetah:Medium-Expert)             train_steps=3; test_steps=0 ;;
      6halfcheetah:Medium)                    train_steps=0; test_steps=1 ;;
      6halfcheetah:Medium-Replay)             train_steps=3; test_steps=1 ;;
      *) printf 'Unsupported scenario/dataset: %s/%s\n' "$scenario" "$dataset" >&2; exit 2 ;;
    esac

    for seed in $(seq "$SEED_START" "$SEED_END"); do
      printf '\nSCOUT scenario=%s dataset=%s seed=%s train_steps=%s report_test_steps=%s\n' \
        "$scenario" "$dataset" "$seed" "$train_steps" "$test_steps"
      command=(
        "$PYTHON" -u main.py
        --agent_name=scout --env=mamujoco --source=omiga
        "--scenario=$scenario" "--dataset=$dataset" "--seed=$seed"
        "--project_name=$PROJECT_NAME" "--run_group=${scenario}_${dataset}"
        "--data_dir=$DATA_DIR" "--save_dir=$SAVE_DIR"
        "--offline_steps=$OFFLINE_STEPS" --online_steps=0
        --batch_size=32 --sequence_length=20
        "--log_interval=$LOG_INTERVAL" "--eval_interval=$EVAL_INTERVAL"
        "--save_interval=$SAVE_INTERVAL" "--num_eval_episodes=$NUM_EVAL_EPISODES"
        --num_eval_workers=1
        "--agent.train_vgf_steps=$train_steps" "--agent.vgf_lr=$VGF_LR"
        --agent.vgf_alpha=1.0 --agent.vgf_particles=10 --agent.flow_steps=10
      )
      if [[ "$DRY_RUN" == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
      else
        "${command[@]}"
      fi
    done
  done
done
