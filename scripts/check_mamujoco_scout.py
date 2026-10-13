"""Check GPU execution, MuJoCo, an update, and test-time refinement without datasets."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from agents.scout import ScoutAgent, get_config
from envs.environments import get_environment
from utils.replay_buffers import FlashbaxReplayBuffer


def check_scenario(scenario):
    env = get_environment("omiga", "mamujoco", scenario, seed=0)
    try:
        observations, infos = env.reset()
        buffer = FlashbaxReplayBuffer(sequence_length=2, batch_size=2, max_size=16)
        rng = np.random.default_rng(0)
        for _ in range(8):
            actions = {
                name: rng.uniform(-0.5, 0.5, env.num_actions).astype(np.float32)
                for name in env.agents
            }
            next_obs, rewards, terminals, truncations, next_infos = env.step(actions)
            buffer.add(observations, actions, rewards, terminals, truncations,
                       {"state": infos["state"]})
            observations, infos = next_obs, next_infos
            if all(terminals.values()) or all(truncations.values()):
                observations, infos = env.reset()

        batch = buffer.sample()
        config = get_config().to_dict()
        config.update(train_vgf_steps=0 if scenario == "6halfcheetah" else 2,
                      vgf_lr=0.02, total_steps=1)
        agent = ScoutAgent.create(
            0, jnp.asarray(batch["observations"]), jnp.asarray(batch["actions"]),
            list(env.agents), config,
        )
        agent, metrics = agent.update(batch, 1)
        metrics = {key: float(value) for key, value in metrics.items()}
        assert all(np.isfinite(value) for value in metrics.values()), metrics
        for steps in (0, 1, 3, 5, 10):
            actions = agent.sample_actions(observations, jax.random.PRNGKey(0),
                                           eval_vgf_steps=steps)
            arrays = {key: np.asarray(value) for key, value in actions.items()}
            assert all(value.shape == (env.num_actions,) and np.isfinite(value).all()
                       for value in arrays.values()), arrays
            env.step(arrays)
        print(f"PASS {scenario}: GPU update + refinement steps 0/1/3/5/10 + MuJoCo step",
              flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="+",
                        default=["2ant", "3hopper", "6halfcheetah"])
    args = parser.parse_args()
    print("JAX", jax.__version__, "devices:", jax.devices(), flush=True)
    assert jax.default_backend() == "gpu", "JAX is not using the GPU"
    for scenario in args.scenarios:
        check_scenario(scenario)
    print("All checks passed. No offline dataset was downloaded and no experiment was trained.",
          flush=True)
