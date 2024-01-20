# Gossip Spread Inside a Social Network Simulation

This simulation was inspired by the random-walk based epidemiological model. The idea is to simulate the spread of gossip inside a social network, built using NumPy. Feel free to copy, share, or modify the code for your own purposes!

## TLDR!

You can use the [test_gossip.py](test_gossip.py) file to simulate specific scenarios. Play around with the parameters! If you want to change the core logic, look into [gossip_network.py](gossip_network.py).

## Motivation

I was looking for a fun mathematical puzzle, and I remembered one of my undergrad courses on which I worked with a similar model for simulating a classic epidemic. I then thought: could this be used to simulate something less scary, and more keen to the general public? And the answer is yes! Everyone has seen a gossip spread around at least once. Which is why I dedicated some time to thinking:

1. What are the elements and dynamics of gossip?
2. How does gossip evolve over time?
3. How could I use these insights to build a simulation?

## Components

First of all, for gossip to exist, we need people. They can be simulated using particles, both in a 2D or 3D space. For simplicity, I stuck with 2D.

People aren't simply spreading gossips around just because; they usually do this within their social circle, which is why to simulate this, we would need a social network. This can be built simply using an adjacency matrix, let's call it $A$.

The $A_{i,j}$ entry of the matrix is equal to 1 if $i$ has adjacency with $j$ and 0 otherwise. In real-world scenarios, we could observe cases where this matrix is not symmetrical, since human relationships are complicated. I decided to assume symmetry for simplicity. In fact, the code expects it to be symmetric and will raise exceptions if not the case.

Now, we can incorporate the social aspect into the simulation by only allowing particles with adjacencies to infect each other.

Gossips evolve over time, thus the willingness of spreading them also does. This behavior can easily be simulated using an exponential decrease function: $$e^{-\lambda t}$$, where $\lambda$ is the spread constant, interpreted as how "spicy" or "dull" the gossip is. Values close to zero make for a spicy gossip; the closer to zero, the spicier the gossip.

This time is not the same for all particles, as it must depend on when exactly did they hear about the gossip. In other words, only because your best friend learned about the gossip as soon as possible, doesn't mean everyone would too. The time is relative for each subject.

## How to Use

The bare essentials to use this simulation are:

1. The social network instance. Must be of type np.ndarray and symmetric.
2. The particles instance, which itself contains some more optional parameters.

Additionally, you may provide some or all of the following:

```python
infection_radius: float
movement_radius: float
infected_particles: np.ndarray
spread_rate: float
spread_probabilities: np.ndarray
iterations: int
current_iteration: int
infection_time: np.ndarray
title: str
```
Once you instantiate your Gossip instance, you can either manually control the iterations and use the functions I provided, or whatever your favorite graphics library is, or you can actually animate the whole process using the
```python
Gossip.animate_system_evolution()
```
member function.
## Example Usage

To run a basic simulation, execute the following command:

```bash
python test_gossip.py
```

This file contains some pre fabricated scenarios with easily updatable parameters. 
## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
## License

This project is licensed under the  [MIT License](https://mit-license.org/).


