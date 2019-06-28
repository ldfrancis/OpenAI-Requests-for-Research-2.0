# OpenAI-Requests-for-Research-2.0
This repository will contain my implementations of solutions to the challenges described in the openAI's  [request for research 2.0](https://openai.com/blog/requests-for-research-2/)

## Warmups
I've resolved to attempt the warm-up exercises. These exercises, although, having been solved, would serve as a stepping stone into solving the more complex challenges as they would provide some moral support to annihilate the difficulties the implementer would encounter while struggling with the more difficult challenges.

- [ ] **XOR_LSTM:** ⭐

The aim of this challenge is to train an LSTM to solve the XOR problem. This problem involves determining the parity of a sequence of bits. 2 approaches are to be tested.
1. train the LSTM on a generated dataset of random 100,000 binary strings of length 50 and check the performance.
2. train the LSTM on a generated dataset of random 100,000 binary strings with a randomly chosen length between 1 and 50 and check its performance.


- [ ] **CLASSIC_SNAKE:** ⭐

In this challenge, one is to come up with an implementation of the classic snake game as a Gym environment and then solve the environment using a reinforcement learning algorithm.

## Challenges

- [ ] **SLITHERIN’:** ⭐⭐

This challenge involves Implementing and solving a multiplayer clone of the classic Snake game as a Gym environment.

The environment is specified to be one that has a large field with multiple snakes with each snake being able to eat randomly appearing fruits while avoiding any form of collision with all snakes (including itself) and the walls. The game ends when all snakes are dead.

The environment is to be solved using self-play and the learned behaviour of the agent is to be inspected to see if it tries to completely search for food, try to attack other snakes or just avoid collision.
