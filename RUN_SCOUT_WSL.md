# SCOUT MA-MuJoCo in WSL

Use the separate `GPUscout` Conda environment. The existing `GPUfacmac` and
`~/venvs/mgof-gpu` have legacy MuJoCo but do not have the required JAX stack.
SCOUT reuses the MuJoCo 2.1 runtime at `~/.mujoco/mujoco210`.

## Enter the environment

From Windows PowerShell:

```powershell
wsl -d Ubuntu-20.04
```

Inside WSL:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate GPUscout
cd /mnt/e/PRO/MAERL/NEW_CODE/scout-main
```

For a fresh machine, create and install the environment first:

```bash
conda create -n GPUscout python=3.10 -y
conda activate GPUscout
python -m pip install -r requirements/scout-wsl.txt
```

The exact installed versions on this machine are also saved in
`requirements/scout-wsl.lock.txt`.

## Check GPU and environments without downloading datasets

```bash
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python scripts/check_mamujoco_scout.py
```

The check collects a few random transitions, performs one gradient update, and
checks all five refinement budgets in each environment. These are setup checks,
not experimental results.

## Run the three Medium experiments

```bash
bash scripts/run_mamujoco_scout.sh
```

The default is GPU 0, seed 0, 500,000 offline updates, no online fine-tuning,
10 evaluation episodes, and evaluations/checkpoints every 100,000 updates.
Experiments run sequentially.

| Task | Scenario flag | Agent partition | Medium train steps | Medium report test steps |
| --- | --- | --- | --- | --- |
| Ant-v2 | `2ant` | `2x4` | 2 | 0 |
| Hopper-v2 | `3hopper` | `3x1` | 2 | 3 |
| HalfCheetah-v2 | `6halfcheetah` | `6x1` | 0 | 1 |

The refinement step size is 0.02, particles 10, temperature 1.0, and flow steps
10, from Appendix B Tables 4 and 5. The software versions differ from the
paper's A40/JAX 0.4.30 stack to support this WSL setup.

Run just one task:

```bash
SCENARIOS=2ant bash scripts/run_mamujoco_scout.sh
SCENARIOS=3hopper bash scripts/run_mamujoco_scout.sh
SCENARIOS=6halfcheetah bash scripts/run_mamujoco_scout.sh
```

Run all four qualities with eight seeds (96 sequential runs):

```bash
DATASETS="Expert Medium-Expert Medium Medium-Replay" SEED_END=7 \
  bash scripts/run_mamujoco_scout.sh
```

Print commands without starting training or downloading data:

```bash
DRY_RUN=1 bash scripts/run_mamujoco_scout.sh
```

Use a different Python executable:

```bash
PYTHON_BIN="$HOME/miniconda3/envs/GPUscout/bin/python" \
  bash scripts/run_mamujoco_scout.sh
```

## Data and outputs

The script uses the public OMIGA datasets. On this WSL host, Hugging Face direct
access failed, so the script defaults to `HF_ENDPOINT=https://hf-mirror.com`.
The mirror's dataset listing and a ZIP range request were checked. Override
the endpoint if needed:

```bash
HF_ENDPOINT=https://huggingface.co bash scripts/run_mamujoco_scout.sh
```

At the first real run, each task ZIP is downloaded and extracted to:

```text
data/omiga/mamujoco/2ant.vlt/
data/omiga/mamujoco/3hopper.vlt/
data/omiga/mamujoco/6halfcheetah.vlt/
```

The three ZIPs total approximately 2.79 GB (decimal), before extraction.
Each contains all dataset qualities. Existing nonempty vaults are reused.
You can choose a data location with `DATA_DIR=/absolute/path`.

Results are saved under:

```text
exp/SCOUT-MAMuJoCo/<scenario>_<dataset>/sd000_<timestamp>/
  flags.json
  train.csv
  eval.csv
  params_100000.pkl
  ...
```

The entry point evaluates test refinement steps 0, 1, 3, 5, 10 separately.
For example, use `evaluation/mean_episode_return_steps_3` in `eval.csv` for the
Medium Hopper setting. Returns in this file are raw episode returns; the
entry point does not implement the paper's normalized score conversion.

WandB defaults to offline mode; no account login is required. Set
`WANDB_MODE=online` if you want connected logging.

## Compatibility changes

The local setup makes `main.py` honor `--data_dir` and `WANDB_MODE`, accesses
the unwrapped MuJoCo environment for private state methods, implements environment
cleanup, and marks evaluation refinement steps as static for JAX compilation.
Checkpoint writes use the existing `save_agent` / `restore_agent` helpers.
These changes address startup and evaluation errors found during setup checks.
