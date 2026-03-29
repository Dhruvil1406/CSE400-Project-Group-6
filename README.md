# CSE 400: Fundamentals of Probability in Computing
Problem Statement : To maximize the probability of vehicle arriving at the destination before calculated time by selecting best route based on traffic congestion (Route Optimisation Problem)

Problem Description : In real-world urban transportation networks, travel times are inherently uncertain due to dynamic traffic conditions, signal delays, congestion variability, and stochastic vehicle interactions. Traditional deterministic shortest-path algorithms assume fixed travel times on road segments and therefore fail to capture this uncertainty, often producing routes that are optimal only in expectation but unreliable in practice.

This project focuses on solving a route optimization problem under uncertainty by maximizing the probability that a vehicle reaches its destination within a given deadline. Unlike traditional approaches that rely on deterministic shortest-path algorithms like A*, this system accounts for real-world variability in traffic conditions such as congestion, delays, and unpredictable travel patterns. While the route itself is initially computed using A*, its evaluation is enhanced using probabilistic modeling to better reflect realistic scenarios.


