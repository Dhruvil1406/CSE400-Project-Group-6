# CSE 400: Fundamentals of Probability in Computing

## Problem Statement
To maximize the probability of a vehicle arriving at the destination before a calculated deadline by selecting the best route based on traffic congestion (Route Optimisation Problem).

## Problem Description
In real-world urban transportation networks, travel times are inherently uncertain due to dynamic traffic conditions, signal delays, congestion variability, and stochastic vehicle interactions. Traditional deterministic shortest-path algorithms (such as the A* Algorithm) assume fixed travel times on road segments and therefore fail to capture this uncertainty. As a result, they often produce routes that are optimal only in expectation but unreliable in practice.

To address this limitation, our project models travel time as a stochastic process by incorporating traffic variability and congestion effects. Instead of relying on a fixed route decision, we use a Monte Carlo simulation-based approach to evaluate route performance across multiple possible scenarios.

The objective is not just to minimize expected travel time, but to maximize the probability of arriving before a given deadline, ensuring more reliable routing decisions under uncertainty. While the route itself is initially computed using A*, its evaluation is enhanced using probabilistic modeling to better reflect realistic scenarios.

Travel time for each road segment is modeled as a random variable (lognormal distribution), and deadlines are also treated probabilistically. The system uses Monte Carlo simulation to evaluate the selected route across many possible traffic conditions, generating a distribution of outcomes instead of a single estimate.

This approach enables the computation of reliability metrics such as:
- Probability of on-time arrival  
- Expected travel time  
- Variance in travel time  

As a result, the objective shifts from simply minimizing travel time to selecting routes that are more reliable and risk-aware, making the solution more effective in real-world, uncertain transportation environments.
