# Documentation

This week, we completed Level 1 and Level 2.

## Level 1
We finished the implementation and successfully tested it on the CartPole environment.

## Level 2

To visualize the performance of our agents, we used the Sample Efficiency Curve from RLiable.

We implemented a class called SampleEfficiencyPlotter, which stores the scores_dict as an attribute. New values are added via the add_Value() function, and the results are plotted using the plot() function.

To enable a more accurate comparison of sample complexity between REINFORCE and DQN, and to evaluate the effect of different trajectory lengths, we chose to use total frames on the x-axis instead of episodes. This required changes to the train() function so that evaluation occurs based on the number of frames rather than completed episodes.

Each Sample Efficiency Curve represents five runs on the CartPole environment, with evaluations performed every 5,000 frames. The x-axis spans a total of 100,000 frames, resulting in a scores_dict["algorithm"] shape of (5 × 1 × 20).

### Question 1
- How does the trajectory length affect training stability and convergence?

The graph "trajectory_length.png" compares two training runs: one with a maximum of 100 steps per episode and the default setting of 500 steps per episode.

Both models were evaluated using 500-step episodes, ensuring a fair performance comparison.

The graph shows that training with shorter trajectories (100 steps) leads to greater instability and fails to converge to the maximum reward of 500, unlike the training with longer episodes. However like you can see on the last reward, the longer trajectory is also not stable.

### Question 2
- What is the impact of network architecture and learning rate?

Learning Rate (Figure: learning_rate.png)
The graph compares two learning rates: 1e-2 (blue) and 2e-3 (orange).
Both learning rates lead to convergence at the maximum return of 500. The higher learning rate (1e-2) converges more quickly but with less stability, while the lower learning rate (2e-3) converges more slowly but consistently.

Network Architecture (Figure: Hidden_Layer.png)
This plot compares three hidden layer sizes: 64, 128, and 256 units.
The configuration with 128 units (blue) performs best overall. It reaches the maximum return quickly and consistently.
The 64-unit network (orange) converges, but shows more instability and lower returns in later stages.
The 256-unit network (green) converges slowly and exhibits higher variance, possibly due to overfitting.

### Question 3
- How does the sample complexity of REINFORCE compare to DQN?
The graph "DQN_200ksteps" shows the mean rewards in the Cartpole environment from our DQN implementation last week. Although DQN eventually converges to an average reward of 500, it requires many more training steps to reach this level. Comparing this to our REINFORCE results from this week, it can be observed that Deep Q-Learning is much more stable, even if REINFORCE might learn faster initially.



