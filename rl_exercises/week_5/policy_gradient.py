from typing import Any, Dict, List, Tuple

import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rliable import library as rly
from rliable import metrics, plot_utils


class SampleEfficencyPlotter:
    def __init__(self, frames):
        self.scores_dict = {}
        self.frames: np.array = frames
        self.algorithms = []
        sns.set(style="whitegrid")

    def add_algorithm(
        self, algorithm: str = "dummy", num_runs=1, num_games=1, num_frames=10
    ) -> None:
        self.algorithms.append(algorithm)
        self.scores_dict[algorithm] = np.zeros((num_runs, num_games, num_frames))

    def add_Value(
        self, key: str, value: float, run: int, game: int, frame: int
    ) -> None:
        self.scores_dict[key][run - 1, game - 1, frame - 1] = value

    def plot(self) -> None:
        print(self.scores_dict)
        iqm = lambda scores: np.array(
            [
                metrics.aggregate_iqm(scores[..., frame])
                for frame in range(scores.shape[-1])
            ]
        )

        # Dann Intervalle berechnen
        iqm_scores, iqm_cis = rly.get_interval_estimates(
            self.scores_dict, iqm, reps=50000
        )

        plot_utils.plot_sample_efficiency_curve(
            self.frames - 1,
            iqm_scores,
            iqm_cis,
            algorithms=self.algorithms,
            xlabel=r"Number of Frames",
            ylabel="Mean Return",
        )
        plt.legend(loc="best")
        plt.show()


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed random number generators for reproducibility.

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment to seed.
    seed : int, optional
        Seed value for NumPy, PyTorch, and environment (default is 0).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


# TODO: Complete the Policy implementatoin
class Policy(nn.Module):
    """
    Multi-layer perceptron mapping states to action probabilities.

    Implements a linear feed-forward network with one hidden layer and softmax output.

    Parameters
    ----------
    state_space : gym.spaces.Box
        Observation space defining the dimensionality of inputs.
    action_space : gym.spaces.Discrete
        Action space defining number of output classes.
    hidden_size : int, optional
        Number of units in the hidden layer (default is 128).
    """

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Discrete,
        hidden_size: int = 128,
    ):
        """
        Initialize the policy network.

        Parameters
        ----------
        state_space : gym.spaces.Box
            Observation space of the environment.
        action_space : gym.spaces.Discrete
            Action space of the environment.
        hidden_size : int, optional
            Number of hidden units. Defaults to 128.
        """
        super().__init__()
        self.state_dim = int(np.prod(state_space.shape))
        self.n_actions = action_space.n

        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, self.n_actions)

        # TODO: Define two linear layers: self.fc1 and self.fc2
        # self.fc1 should map from self.state_dim to hidden_size
        # self.fc2 should map from hidden_size to self.n_actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute action probabilities for given state(s).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (state_dim,) or (batch_size, state_dim).

        Returns
        -------
        torch.Tensor
            Softmax probabilities over actions, shape (batch_size, n_actions).
        """

        # TODO: Apply fc1 followed by ReLU (Flatten input if needed)
        # TODO: Apply fc2 to get logits
        # TODO: Return softmax over logits along the last dimension

        # x = self.ln(x)  # <- Normalisierung vor dem Netz

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        probs = nn.functional.softmax(x, dim=-1)

        return probs


class REINFORCEAgent(AbstractAgent):
    """
    REINFORCE agent performing on-policy Monte Carlo policy gradient updates.

    Wraps an MLP policy network and optimizer, providing train, predict, save, load, and evaluate methods.

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment for interaction.
    lr : float, optional
        Learning rate for optimizer (default is 1e-2).
    gamma : float, optional
        Discount factor for returns (default is 0.99).
    seed : int, optional
        Random seed for reproducibility (default is 0).
    """

    def __init__(
        self,
        env: gym.Env,
        lr: float = 1e-2,
        gamma: float = 0.99,
        seed: int = 0,
        hidden_size: int = 128,
    ) -> None:
        """
        Initialize the REINFORCE agent.

        Args:
            env (gym.Env): Environment for training.
            lr (float, optional): Learning rate. Defaults to 1e-2.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            seed (int, optional): Random seed. Defaults to 0.
        """

        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.policy = Policy(env.observation_space, env.action_space, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.total_episodes = 0
        self.algorithms = []
        self.mean_returns_dict = {}

    def predict_action(
        self, state: np.ndarray, info: Dict[str, Any] = {}, evaluate: bool = False
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action according to the current policy.

        In training mode, samples stochastically and returns log probability.
        In evaluation mode, returns the argmax action deterministically.

        Parameters
        ----------
        state : np.ndarray
            Current observation from the environment.
        info : dict, optional
            Additional info (unused here, default is empty).
        evaluate : bool, optional
            If True, use deterministic policy (default is False).

        Returns
        -------
        action : int
            Selected action index.
        info_out : dict
            Contains 'log_prob' if in training mode; empty if evaluating.
        """
        # TODO: Pass state through the policy network to get action probabilities
        # If evaluate is True, return the action with highest probability
        # Otherwise, sample from the action distribution and return the log-probability as a key in the dictionary (Hint: use torch.distributions.Categorical)

        state_tensor = torch.tensor(state, dtype=torch.float32)

        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        probs = self.policy.forward(state_tensor)

        if evaluate:
            action = probs.argmax(dim=1)
            return action.item(), {}
        else:
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            return action.item(), {"log_prob": log_prob.squeeze()}

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute discounted reward-to-go for each timestep.

        Parameters
        ----------
        rewards : list of float
            Sequence of rewards for one episode.

        Returns
        -------
        torch.Tensor
            Discounted returns tensor of shape (len(rewards),).
        """

        # TODO: Initialize running return R = 0
        # TODO: Iterate over rewards and compute the return-to-go:
        #       - Update R = r + gamma * R
        #       - Insert R at the beginning of the returns list
        # TODO: Convert the list of returns to a torch.Tensor and return

        _R = 0
        returns = []
        for r in reversed(rewards):
            _R = r + self.gamma * _R
            returns.insert(0, _R)

        return torch.tensor(returns, dtype=torch.float32)

    def update_agent(
        self,
        training_batch: List[
            Tuple[np.ndarray, int, float, np.ndarray, bool, Dict[str, Any]]
        ],
    ) -> float:
        """
        Perform a policy-gradient update using one full episode.

        Parameters
        ----------
        training_batch : list of tuples
            Each tuple is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            Scalar loss value after update.
        """
        # unpack log_probs & rewards
        log_probs = [t[5]["log_prob"] for t in training_batch]
        rewards = [t[2] for t in training_batch]

        # compute discounted returns
        returns_t = self.compute_returns(rewards)

        # TODO: Normalize returns with mean and standard deviation,
        # and add 1e-8 to the denominator to avoid division by zero

        mean_returns = returns_t.mean()
        std_returns = returns_t.std(unbiased=False)
        norm_returns = (returns_t - mean_returns) / (std_returns + 1e-8)

        # norm_returns = returns_t

        lp_tensor = torch.stack(log_probs)
        loss = -torch.sum(lp_tensor * norm_returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save(self, path: str) -> None:
        """
        Save policy network and optimizer state to a checkpoint.

        Parameters
        ----------
        path : str
            File path to save checkpoint.
        """
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load policy network and optimizer state from checkpoint.

        Parameters
        ----------
        path : str
            File path of checkpoint to load.
        """
        ckpt = torch.load(path)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    def evaluate(
        self,
        eval_env: gym.Env,
        num_episodes: int = 10,
        max_steps=500,
    ) -> Tuple[float, float]:
        """
        Evaluate policy over multiple episodes.

        Parameters
        ----------
        eval_env : gym.Env
            Environment for evaluation.
        num_episodes : int, optional
            Number of episodes to run (default is 10).

        Returns
        -------
        mean_return : float
            Average episode return.
        std_return : float
            Standard deviation of returns.
        """
        self.policy.eval()
        returns: List[float] = []  # noqa: F841
        # TODO: rollout num_episodes in eval_env and aggregate undiscounted returns across episodes

        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            done = False
            total_return = 0.0
            steps = 0

            while not done and steps < max_steps:
                steps += 1
                action, _ = self.predict_action(
                    state, evaluate=True
                )  # keine Exploration
                state, reward, done, truncated, _ = eval_env.step(action)
                total_return += reward
                if done or truncated:
                    break

            returns.append(total_return)

        self.policy.train()  # Set back to training mode

        # TODO: Return the mean and std of the returns across episodes

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        return mean_return, std_return

    def train(
        self,
        num_episodes: int,
        plotter: SampleEfficencyPlotter,
        eval_interval: int = 10,
        eval_episodes: int = 5,
        max_steps: int = 500,
        key: str = "dummy",
        run: int = 1,
        eval_interval_steps: int = 5000,
        num_steps: int = 100000,
    ) -> None:
        """def train(
        self,
        num_episodes: int,
        eval_interval: int = 10,
        eval_episodes: int = 5,
        max_steps: int = 500,
        key: String = ""
        Train the agent on-policy for a number of episodes.

        Parameters
        ----------
        num_episodes : int
            Total number of training episodes.
        eval_interval : int, optional
            Frequency of evaluation prints (default is 10).
        """
        eval_env = gym.make(self.env.spec.id)  # fresh copy for eval
        i = 0
        k = 0
        total_steps = 0

        while total_steps < num_steps + 1:
            state, _ = self.env.reset()
            done = False
            batch: List[Tuple[Any, ...]] = []
            steps = 0

            while not done and steps < max_steps:
                steps += 1
                action, info = self.predict_action(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                batch.append((state, action, float(reward), next_state, done, info))
                state = next_state

            total_steps += steps
            loss = self.update_agent(batch)
            total_return = sum(r for _, _, r, *_ in batch)
            self.total_episodes += 1

            if total_steps >= (k + 1) * 1000:
                print(
                    f"[Train] Step {total_steps:3d} Return {total_return:5.1f} Loss {loss:.3f}"
                )
                k += 1

            # if ep % eval_interval == 0:
            #    mean_ret, std_ret = self.evaluate(eval_env, num_episodes=eval_episodes, max_steps=500)
            #    print(f"[Eval ] Ep {ep:3d} AvgReturn {mean_ret:5.1f} ± {std_ret:4.1f}")

            if total_steps >= (i + 1) * eval_interval_steps:
                mean_ret, std_ret = self.evaluate(
                    eval_env, num_episodes=eval_episodes, max_steps=500
                )
                print(
                    f"[Eval ] Step {total_steps:3d} AvgReturn {mean_ret:5.1f} ± {std_ret:4.1f}"
                )
                plotter.add_Value(key=key, value=mean_ret, game=1, run=run, frame=i)
                i += 1

        print("Training complete.")


def render_agent(agent, env, num_episodes=1):
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = agent.predict_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            env.render()  # <- hier wird gerendert!
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep + 1} return: {total_reward}")


@hydra.main(
    config_path="../configs/agent/", config_name="reinforce", version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training with Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration with fields:
          env:
            name: str        # Gym environment id
          seed: int
          agent:
            lr: float
            gamma: float
            hidden_size: int
          train:
            episodes: int
            eval_interval: int
            eval_episodes: int
    """
    # Initialize environment and seed
    # env = gym.make(cfg.env.name, render_mode="human")
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # Instantiate agent with hyperparameters from config
    agent = None

    eval_steps = cfg.train.eval_steps
    steps = cfg.train.steps
    frames = np.zeros(int(steps / eval_steps))
    print(frames)
    k = 0
    for i in range(1, steps + 1):
        if i % eval_steps == 0:
            frames[k] = i
            k += 1
    print(frames)

    # use episodes instead of frames: frames = np.array([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]) - 1
    plot = SampleEfficencyPlotter(frames - 1)

    # Train agent

    runs = 5
    key = "128"
    plot.add_algorithm(
        algorithm=key, num_runs=runs, num_games=1, num_frames=len(frames)
    )
    for r in range(1, runs + 1):
        if agent is not None:
            del agent
        set_seed(env, r)
        agent = REINFORCEAgent(
            env=env,
            lr=cfg.agent.lr,
            gamma=cfg.agent.gamma,
            seed=r,
            hidden_size=cfg.agent.hidden_size,
        )
        agent.train(
            num_episodes=cfg.train.episodes,
            eval_interval=cfg.train.eval_interval,
            eval_episodes=cfg.train.eval_episodes,
            max_steps=cfg.train.max_steps,
            key=key,
            plotter=plot,
            run=r,
            eval_interval_steps=cfg.train.eval_steps,
            num_steps=cfg.train.steps,
        )
    del agent
    agent = REINFORCEAgent(
        env=env,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
    )
    runs = 5
    key = "64"
    plot.add_algorithm(
        algorithm=key, num_runs=runs, num_games=1, num_frames=len(frames)
    )
    for r in range(1, runs + 1):
        if agent is not None:
            del agent
        set_seed(env, r)
        agent = REINFORCEAgent(
            env=env,
            lr=cfg.agent.lr,
            gamma=cfg.agent.gamma,
            seed=r,
            hidden_size=64,
        )
        agent.train(
            num_episodes=cfg.train.episodes,
            eval_interval=cfg.train.eval_interval,
            eval_episodes=cfg.train.eval_episodes,
            max_steps=cfg.train.max_steps,
            key=key,
            plotter=plot,
            run=r,
            eval_interval_steps=cfg.train.eval_steps,
            num_steps=cfg.train.steps,
        )

    runs = 5
    key = "256"
    plot.add_algorithm(
        algorithm=key, num_runs=runs, num_games=1, num_frames=len(frames)
    )
    for r in range(1, runs + 1):
        if agent is not None:
            del agent
        set_seed(env, r)
        agent = REINFORCEAgent(
            env=env,
            lr=cfg.agent.lr,
            gamma=cfg.agent.gamma,
            seed=r,
            hidden_size=256,
        )
        agent.train(
            num_episodes=cfg.train.episodes,
            eval_interval=cfg.train.eval_interval,
            eval_episodes=cfg.train.eval_episodes,
            max_steps=cfg.train.max_steps,
            key=key,
            plotter=plot,
            run=r,
            eval_interval_steps=cfg.train.eval_steps,
            num_steps=cfg.train.steps,
        )

    plot.plot()

    env = gym.make(cfg.env.name, render_mode="human")
    render_agent(agent, env)


if __name__ == "__main__":
    main()
