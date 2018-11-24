# Report

## Learning Algorithm

### Architecture

The problem was solved with different kinds of networks and architectures to see what works best.

A Multi Agent DDPG (Deep Deterministic Policy Gradients) seems to work very well for this particular problem.

Different configurations of weight sharing was attempted to see what works best. Example,

- Sharing critics
- Sharing actors and critics
- Sharing actors and critics and the replay buffer

Sharing local actor network, local critic nework and the replay buffer across the agents produced the best results.

DDPG is a variation of actor critic methods. MADDPG adapts agent training for multiple agents in either collaborative or competitive configurations, or even in a combination of the two with the help of some reward engineering. See [OpenAI team_spirit](https://blog.openai.com/openai-five/) for an example of how this is done!

#### Notes about Actor Critic Agents

Policy networks (actors), which uses advantage (relative returns) to reinforce actions are in general faster learners and has
low variance, however they accumulate bias over time.

Value networks (critics) on the other hand use bias (error) to arrive at better state value estimations and results in low bias,
however they are slow learners and has high variance.

Actor-critic agents combines a policy network (actor) with high learning rate and a value network (critic) with solid baseline
to help each other and learn more effectively. This leads the overall model to better estimate state values, gain more confidence in
its own actions and reduce surprises in general (which means they tend to take better actions).

### Hyperparameters and Implementation Notes

- Overall network architecture: Multi Agent Deep Deterministic Policy Gradients
- Optimizer: Adam with a learning rate of 1e-3 for both actor and critic
- Discount: 0.99
- Replay buffer size: 1e5
- Batch size: 128

A better LR helped the network to converge faster. The code is very similar to DDPG, but with the important difference that the networks are shared among multiple agents.

Long replay buffer was needed to help the networks bootstrap.

Sharing both actor and critic networks and the replay buffer helped the network converge much much faster than other configurations.

An additional state of (-1 or 1) indicating which agent, was added to help the shared networks recognize what agent is in context.

## Plot of Rewards

Mean episode rewards are plotted at the end of training

![](./score.png)

## Ideas for Future Work

### Explore

- Try out more multi-agent environments and algorithm variations
- Find more real life situations where MADDPG could be applied
- Continue to research better LR tuning for RL (can we use variable LR schedules, etc.)
- Adapt the two agent MADDPG implementation to n agents
