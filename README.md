# RL Blocking Manager: Run Instructions

This README is for the RL training/evaluation script in `scripts/ppo_block_manager.py`.

The repo `scripts/` directory currently contains:
- `ppo_block_manager.py`
- `test_blocking_demo.py`
- `test_two_agent.py`

## 1) Go to the repo root

```bash
cd ~/f1tenth_blocking_rl
```

## 2) Activate the virtual environment

Based on your current setup, use the Conda environment:

```bash
conda activate f1gym39
```

If Conda is not initialized in that terminal, run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate f1gym39
mkdir -p logs/rl_runs
```
## 3) Training command

This command trains PPO and stores all outputs in a dedicated run folder under `logs/rl_runs/`.

```bash
python scripts/ppo_block_manager.py train   --config maps/config_example_map.yaml   --random-spawn   --spawn-gap-min 1.0   --spawn-gap-max 3.0   --ego-lateral-offset-rand 0.1   --opp-lateral-offset-rand 0.5   --spawn-yaw-rand 0.1   --opp-rrt-replan-every 5   --steps 1000   --total-timesteps 400000   --output-dir logs/rl_runs/ppo_block_debug   --device cpu
```

### What gets written into the training log folder

Inside `logs/rl_runs/$RUN_NAME/`, the script writes:

- `monitor.csv`
- `epoch_metrics.csv`
- `checkpoints/`
- `tb/`
- `ppo_block_final.zip`

## 4) View TensorBoard logs

```bash
tensorboard --logdir logs/rl_runs
```

Then open the local TensorBoard URL it prints in the terminal.

## 5) Evaluation command

This evaluates a trained PPO model and writes evaluation logs to a separate folder.

```bash
MODEL_RUN=REPLACE_WITH_YOUR_TRAIN_RUN_FOLDER
EVAL_NAME=${MODEL_RUN}_eval
mkdir -p logs/rl_runs/$EVAL_NAME

python scripts/ppo_block_manager.py eval   --config maps/config_example_map.yaml   --random-spawn   --spawn-gap-min 1.0   --spawn-gap-max 2.5   --ego-lateral-offset-rand 0.05   --opp-lateral-offset-rand 0.20   --spawn-yaw-rand 0.05   --opp-rrt-replan-every 5   --steps 1000   --model-path logs/rl_runs/$MODEL_RUN/ppo_block_final.zip   --output-dir logs/rl_runs/$EVAL_NAME
```

## 6) What gets written during evaluation

Inside `logs/rl_runs/$EVAL_NAME/`, the script writes:

- `eval_metrics.csv`
- `eval_summary.json`

## 7) Quick start example

```bash
cd ~/f1tenth_blocking_rl
source ~/miniconda3/etc/profile.d/conda.sh
conda activate f1gym39
pip install stable-baselines3 "gymnasium>=0.29" tensorboard

RUN_NAME=ppo_block_$(date +%Y%m%d_%H%M%S)
mkdir -p logs/rl_runs/$RUN_NAME

python scripts/ppo_block_manager.py train   --config maps/config_example_map.yaml   --random-spawn   --spawn-gap-min 1.0   --spawn-gap-max 2.5   --ego-lateral-offset-rand 0.05   --opp-lateral-offset-rand 0.20   --spawn-yaw-rand 0.05   --opp-rrt-replan-every 5   --steps 1000   --total-timesteps 200000   --output-dir logs/rl_runs/$RUN_NAME
```

## 8) Recommended folder layout for runs

```text
f1tenth_blocking_rl/
├── logs/
│   └── rl_runs/
│       ├── ppo_block_YYYYMMDD_HHMMSS/
│       │   ├── monitor.csv
│       │   ├── epoch_metrics.csv
│       │   ├── checkpoints/
│       │   ├── tb/
│       │   └── ppo_block_final.zip
│       └── ppo_block_YYYYMMDD_HHMMSS_eval/
│           ├── eval_metrics.csv
│           └── eval_summary.json
└── scripts/
    └── ppo_block_manager.py
```

## 9) Notes

- Run these commands from the repo root: `~/f1tenth_blocking_rl`
- The training script is under `scripts/ppo_block_manager.py`
- The environment uses random spawn when `--random-spawn` is passed
- The follower uses the RRT* overtaking baseline during RL training/evaluation
