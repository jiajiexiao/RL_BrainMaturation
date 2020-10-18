from rlbrainmaturation.tasks.task import Task, Instruction
from rlbrainmaturation.tasks.odr import ODR
from rlbrainmaturation.tasks.gap import Gap
from rlbrainmaturation.tasks.odr_distract import ODRDistract
from rlbrainmaturation.envs.environment import Environment

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.a3c import A2CTrainer
from ray.tune.registry import register_env
import numpy as np


def main() -> None:
    ray.init()
    np.random.seed(0)

    # instructions = {
    #     0: [Instruction(time=0, x=5, y=5)],
    #     1: [Instruction(time=1, x=5, y=5), Instruction(time=1, x=1, y=5)],
    #     2: [Instruction(time=2, x=5, y=5, rng=np.random.default_rng())],
    # }
    # task = Task(
    #     target_x=1,
    #     target_y=5,
    #     instructions=instructions,
    #     tot_frames=4,
    #     width=42,
    #     height=42,
    # )

    # task = ODR(target_x=1, target_y=5, width=42, height=42)
    # task = Gap(target_x=1, target_y=5, width=42, height=42)
    task = ODRDistract(target_x=1, target_y=5, width=42, height=42)

    def env_creator(env_config):
        return Environment(env_config)  # return an env instance

    register_env("my_env", env_creator)

    # trainer_config = DEFAULT_CONFIG.copy()
    # trainer_config["num_workers"] = 1
    # trainer_config["train_batch_size"] = 20  # 100
    # trainer_config["sgd_minibatch_size"] = 15  # 32
    # trainer_config["num_sgd_iter"] = 50

    trainer = PPOTrainer(
        env="my_env",
        config={
            "env_config": {"task": task},
            "framework": "torch",
            "num_workers": 1,
            "train_batch_size": 10,
            "sgd_minibatch_size": 5,
            "num_sgd_iter": 10,
            # "model": {
            #     # Whether to wrap the model with an LSTM.
            #     "use_lstm": True,
            #     # Max seq len for training the LSTM, defaults to 20.
            #     "max_seq_len": task.tot_frames - 1,
            #     # # Size of the LSTM cell.
            #     "lstm_cell_size": task.tot_frames - 1,
            #     # # Whether to feed a_{t-1}, r_{t-1} to LSTM.
            #     # # "lstm_use_prev_action_reward": False,
            # },
        },
    )

    trainer = A2CTrainer(
        env="my_env",
        config={
            "env_config": {"task": task},
            "framework": "torch",
            "num_workers": 1,
            "train_batch_size": 10,
            # "model": {
            #     # Whether to wrap the model with an LSTM.
            #     "use_lstm": True,
            #     # Max seq len for training the LSTM, defaults to 20.
            #     "max_seq_len": task.tot_frames - 1,
            #     # # Size of the LSTM cell.
            #     "lstm_cell_size": task.tot_frames - 1,
            #     # # Whether to feed a_{t-1}, r_{t-1} to LSTM.
            #     # # "lstm_use_prev_action_reward": False,
            # },
        },
    )

    # trainer = DQNTrainer(
    #     env="my_env",
    #     config={
    #         "env_config": {"task": task},
    #         "framework": "torch",
    #         "num_workers": 1,
    #         "train_batch_size": 10,
    #         # "model": {
    #         #     # Whether to wrap the model with an LSTM.
    #         #     "use_lstm": True,
    #         #     # Max seq len for training the LSTM, defaults to 20.
    #         #     "max_seq_len": task.tot_frames - 1,
    #         #     # # Size of the LSTM cell.
    #         #     "lstm_cell_size": task.tot_frames - 1,
    #         #     # # Whether to feed a_{t-1}, r_{t-1} to LSTM.
    #         #     # # "lstm_use_prev_action_reward": False,
    #         # },
    #     },
    # )

    env = Environment(env_config={"task": task})

    for i in range(200):
        print(f"Training iteration {i}...")
        trainer.train()

        done = False
        cumulative_reward = 0.0
        observation = env.reset()

        while not done:
            action = trainer.compute_action(observation)

            observation, reward, done, results = env.step(action)
            print(f"Time: {env.time}. Action: {action}")
            cumulative_reward += reward
        print(
            f"Last step reward: {reward: .3e}; Cumulative reward: {cumulative_reward:.3e}"
        )


if __name__ == "__main__":
    main()
