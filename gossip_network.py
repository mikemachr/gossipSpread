import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.animation import FuncAnimation

class SocialNetwork:
    """Used to control the social interactions between the particles."""
    def __init__(self, network: np.ndarray = None):
        """Initializes the social network with the given matrix if it passes validation."""
        if network is not None and self.validate_social_network(network):
            self.network = network
        else:
            self.network = None

    def validate_social_network(self, network):
        """Validates provided network."""
        if not isinstance(network, np.ndarray):
            raise ValueError(f"Network must be of type: {type(np.ndarray)}")
        m, n = network.shape
        if m != n:
            raise ValueError(f"Invalid dimensions for matrix: {m}*{n}")
        if not np.array_equal(network.T, network):
            raise ValueError("Matrix must be symmetric")

        for row in range(m):
            for col in range(n):
                if network[row, col] not in [0, 1]:
                    raise ValueError(f"Invalid entry in network at position: {row},{col}")
                if row == col and network[row, col] == 1:
                    raise ValueError(f"Entry {row},{col} cannot have an adjacency to itself")

        if network.sum() == 0:
            warnings.warn("Network contains no adjacencies")
        return True

    def __len__(self):
        """Defines the length of the network as the number of rows/columns."""
        return self.network.shape[0]

    def __getitem__(self, index):
        """Method to subscript the network."""
        return self.network[index]

class Particle:
    """Used to control all operations related to the particle's positions."""
    radius: float
    update_radius: float
    n_particles: int
    positions: np.ndarray

    def __init__(self, radius: float = 1.0, update_radius: float = 0.1, n_particles: int = 30, starting_positions=None):
        """Initializes the particle instance."""
        if starting_positions is None:
            starting_positions = np.random.normal(0, .2 * radius, size=(n_particles, 2))

        if radius is not None:
            if self.verify_particles(radius, update_radius, starting_positions):
                self.positions = starting_positions
                self.n_particles = n_particles if starting_positions is None else starting_positions.shape[0]
                self.radius = radius
                self.update_radius = update_radius
                self.update_positions()
        else:
            self.radius = None
            self.positions = None

    def __len__(self):
        return self.n_particles

    def verify_particles(self, radius: float, update_radius: float, starting_positions: np.ndarray) -> bool:
        """Verifies the provided parameters result in a valid particle instance."""
        if not isinstance(radius, float) or radius <= 0:
            raise ValueError(f"Invalid radius: {radius} provided. Must be a positive float")

        if not isinstance(update_radius, float) or update_radius <= 0:
            raise ValueError(f"Invalid update radius: {update_radius} provided. Must be a positive float")

        m, n = starting_positions.shape
        if n != 2:
            raise ValueError(f"Invalid dimensions for array: {m}*{n}. Expected m*2")
        epsilon = 1e-10
        if np.any(np.linalg.norm(starting_positions, axis=1) > radius+epsilon):
            raise ValueError("At least one invalid starting position was provided. "
                            "Check that all positions fall strictly inside radius")


        return True

    def update_positions(self):
        """Updates the positions of the particles starting from the previous position.
        Ensures they stay inside bounds."""
        n_points = len(self)

        angles = np.random.uniform(0, 2 * np.pi, n_points)
        radii = np.sqrt(np.random.uniform(0, self.update_radius**2, n_points))  
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)

        coordinates: np.ndarray = np.column_stack((x, y)) + self.positions

        distances_from_origin = np.linalg.norm(coordinates, axis=1)
        scaling_factor = np.minimum(1.0, self.radius / distances_from_origin)

        coordinates *= scaling_factor[:, np.newaxis]

        self.positions = coordinates

class Gossip:
    """Controls the operations related to the gossip for the particles.
    Manages status and spread probabilities."""

    social_network: SocialNetwork
    particles: Particle
    infection_radius: float
    movement_radius: float
    infected_particles: np.ndarray
    spread_rate: float
    spread_probabilities: np.ndarray
    iterations: int
    current_iteration: int
    infection_time: np.ndarray
    title: str

    def __init__(self, social_network: SocialNetwork = None, particles: Particle = None, iterations: int = 100,
                 infection_radius: float = 0.1, movement_radius: float = 0.1,
                 infectable_particles: int | np.ndarray = 30, spread_rate: float = 0.,title: str = None) -> None:
        """Initializes a gossip instance with provided parameters. Social_network and particles are mandatory.
        Infection radius must be positive float, infected particles may be int containing number of infected
        particles at start, or binary array containing status of particles. Spread rate must be in interval [0, inf)."""
        if particles is None or social_network is None:
            raise ValueError("Must provide social network and particles instances!")

        n_particles = len(particles)
        network_size = len(social_network)

        if network_size != n_particles:
            raise ValueError(f"Network size and number of particles don't match, network size: {network_size}, particles: {particles}")

        if isinstance(iterations, int):
            self.iterations = iterations
        else:
            raise TypeError(iterations, "Iterations must be of type int")

        if isinstance(infectable_particles, int):
            self.infected_particles = np.random.choice([0, 1], size=infectable_particles)
        else:
            self.infected_particles = infectable_particles

        if self.infected_particles.shape[0] != n_particles:
            raise ValueError(f"Cannot initialize particles status. Number of assignable particles status: {len(self.infected_particles)} "
                             f"Number of particles: {n_particles}")

        self.social_network = social_network
        self.particles = particles

        if isinstance(infection_radius, float) and infection_radius > 0:
            self.infection_radius = infection_radius
        else:
            raise ValueError(f"Invalid infection radius: {infection_radius} provided. Must be a positive float")

        if isinstance(movement_radius, float) and movement_radius > 0:
            self.movement_radius = movement_radius
        else:
            raise ValueError(f"Invalid movement radius: {movement_radius} provided. Must be a positive float")

        if not isinstance(spread_rate, float) or not 0 <= spread_rate <= 1:
            raise ValueError("Please provide a valid spread rate. Must be of type float, inside interval [0, 1]")
        else:
            self.spread_rate = spread_rate

        self.spread_probabilities = np.copy(self.infected_particles)
        infection_time = np.zeros_like(self.infected_particles, dtype=np.object_)
        infection_time[self.infected_particles == 0] = np.nan
        self.infection_time = infection_time
        self.current_iteration = 0
        self.title = title

    def update_particle_status(self):
        """Checks if particles should be infected based on current status and position relative to infected particles,
        then updates them accordingly."""
        spread = (self.spread_probabilities >= np.random.uniform(0, 1, self.particles.n_particles)) #self.* self.infected_particles
        updated_statuses = np.copy(self.infected_particles)

        for i in range(len(spread)):
            if spread[i]:
                distances = np.linalg.norm(self.particles.positions - self.particles.positions[i], axis=1)
                nearby_particles = np.array(distances <= self.infection_radius) * self.social_network[i]
                updated_statuses[nearby_particles==1] = 1.

        self.infection_time[updated_statuses != self.infected_particles] = self.current_iteration
        self.infected_particles = np.logical_or(updated_statuses,self.infected_particles)

    def update_spread_probabilities(self):
        vec_exp = np.vectorize(np.exp)
        self.spread_probabilities = np.ones(self.spread_probabilities.shape[0]) * \
                                   vec_exp(-(self.current_iteration - self.infection_time) * self.spread_rate)
        self.spread_probabilities = self.infected_particles * self.spread_probabilities

    def iteration(self):
        """Simulates the gossip network."""
        self.particles.update_positions()
        self.update_particle_status()
        self.update_spread_probabilities()
        self.current_iteration += 1

        if self.current_iteration == self.iterations:
            return 0

    def plot_iteration(self, ax):
        """Plot a single iteration of the system."""
        ax.clear()

        boundary_circle = plt.Circle((0, 0), self.particles.radius, fill=False, color='black', linestyle='dashed',
                                    linewidth=2)
        ax.add_artist(boundary_circle)

        infected_indices = np.where(self.infected_particles == 1)[0]
        ax.scatter(self.particles.positions[infected_indices, 0],
                   self.particles.positions[infected_indices, 1],
                   color='red', label='Infected', marker='o', s=100, alpha=0.8, edgecolors='black', linewidths=1.5)

        non_infected_indices = np.where(self.infected_particles == 0)[0]
        ax.scatter(self.particles.positions[non_infected_indices, 0],
                   self.particles.positions[non_infected_indices, 1],
                   color='blue', label='Non-Infected', marker='o', s=100, alpha=0.8, edgecolors='black', linewidths=1.5)

        ax.set_xlim(-self.particles.radius, self.particles.radius)
        ax.set_ylim(-self.particles.radius, self.particles.radius)
        ax.set_aspect('equal')
        ax.set_title(f'Iteration {self.current_iteration}\n{self.title}', fontsize=16)
        ax.legend(loc='upper right', fontsize=12, markerscale=1.2, edgecolor='white')

        ax.set_xlabel('X-axis', fontsize=14)
        ax.set_ylabel('Y-axis', fontsize=14)

        ax.set_facecolor('#f5f5f5')

    def animate_system_evolution(self, filename='gossip_evolution.gif'):
        """Animate the evolution of the system and save as a gif."""
        fig, ax = plt.subplots(figsize=(8, 8))

        def update(frame):
            self.iteration()
            self.plot_iteration(ax)

        anim = FuncAnimation(fig, update, frames=self.iterations, repeat=False)
        anim.save(filename, writer='pillow', fps=3)

    def visualize_single_iteration(self):
        """Visualize a single iteration of the system."""
        fig, ax = plt.subplots(figsize=(8, 8))
        self.plot_iteration(ax)
        plt.show()
