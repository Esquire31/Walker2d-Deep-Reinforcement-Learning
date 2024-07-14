# Walker2d-Deep-Reinforcement-Learning

## Project Description
This project demonstrates the implementation of a policy-gradient-based method called REINFORCE from scratch, a fundamental Deep Reinforcement Learning (DRL) algorithm. The primary goal is to train a robot to learn how to walk using the Walker2d environment from Gymnasium (formerly OpenAI Gym).

## Background Theory
### Reinforcement Learning (RL)
Reinforcement Learning (RL) is a subfield of machine learning where an agent learns to perform tasks in a given environment by taking actions and receiving rewards for these actions. The agent's objective is to learn a policy that maximizes the cumulative reward over time.

### Deep Reinforcement Learning (DRL)
Deep Reinforcement Learning (DRL) combines reinforcement learning with deep learning, allowing agents to operate in complex and dynamic environments that traditional RL algorithms can't handle. In DRL, the agent's policy is represented by a deep neural network, which is trained to optimize the cumulative reward. DRL has achieved remarkable results in various fields, including robotics, video games, and natural language processing.

### DRL Algorithms
DRL algorithms can be categorized into two main families:
- **Value-based methods**: Focus on learning the values of the agent's actions in each state and then selecting actions that maximize this value. Example: Deep Q-Networks (DQNs).
- **Policy-based methods**: Directly learn the policy that determines the action to be taken in each state, without explicitly learning the value function. Example: REINFORCE.

Policy-based methods can handle high-dimensional or continuous action spaces more effectively, making them suitable for tasks such as robotic control. However, they often suffer from high variance when updating the policies, leading to less stable training.

### Actor-Critic Methods
These methods combine value-based and policy-based methods by using one neural network (the actor) to choose actions and another (the critic) to evaluate these actions. Example: Deep Deterministic Policy Gradient (DDPG).

### The REINFORCE Algorithm
REINFORCE is a Monte Carlo policy gradient method that updates the policy parameters to maximize the expected return.
REINFORCE updates the policy by computing the gradient of the expected return with respect to the policy parameters and performing gradient ascent.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deep-reinforcement-learning-walker.git
   cd deep-reinforcement-learning-walker
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To train the agent and visualize the results, run the following command:
```bash
python train/main.py
```

## Project Structure
```
├── README.md              # Project description and instructions
├── requirements.txt       # Project dependencies
└── train                  # Directory containing training scripts
    ├── agent.py               # Implementation of the REINFORCE agent
    ├── main.py                # Main script to train the agent
    ├── mujoco_env.py          # Script to test the environment
    ├── policy_network.py      # Neural network definition for the policy
    └── trainer.py             # Training loop for the REINFORCE algorithm
```

## Training Details
- **Environment**: Walker2d-v4 (Gymnasium)
- **Episodes**: 5000
- **Network Architecture**: Two hidden layers with 512 neurons each
- **Learning Rate**: 1e-5
- **Discount Factor (gamma)**: 0.99

## Results
The learning curve showing the episode returns over 5000 episodes is saved as `learning_curve.png`.

### Example of the Initial Stage of Learning
![App Screenshot](https://github.com/Esquire31/Walker2d-Deep-Reinforcement-Learning/blob/main/Examples/mujoco%202024-07-13%2012-22-44.gif)

### Example of the Curve
![App Screenshot](https://github.com/Esquire31/Walker2d-Deep-Reinforcement-Learning/blob/main/Examples/learning_curve.png)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

