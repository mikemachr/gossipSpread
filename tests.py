import unittest
import numpy as np
from gossip_network import SocialNetwork, Particle, Gossip
from create_test_cases import generate_test_network, generate_test_network_particle_positions

class TestSocialNetwork(unittest.TestCase):
    def test_valid_social_network(self):
        matrix = np.array([[0, 1], [1, 0]])
        social_network = SocialNetwork(matrix)
        self.assertEqual(len(social_network), 2)

    def test_invalid_social_network_type(self):
        with self.assertRaises(ValueError):
            SocialNetwork([[0, 1], [1, 0]])

    def test_invalid_social_network_dimensions(self):
        with self.assertRaises(ValueError):
            SocialNetwork(np.array([[0, 1, 0], [1, 0, 1]]))

    def test_invalid_social_network_symmetry(self):
        with self.assertRaises(ValueError):
            SocialNetwork(np.array([[0, 1], [0, 0]]))

    def test_invalid_social_network_entries(self):
        with self.assertRaises(ValueError):
            SocialNetwork(np.array([[0, 2], [1, 0]]))

    def test_invalid_social_network_self_adjacency(self):
        with self.assertRaises(ValueError):
            SocialNetwork(np.array([[1, 0], [1, 0]]))

class TestParticle(unittest.TestCase):
    def test_valid_particle_initialization(self):
        positions = generate_test_network_particle_positions()
        particle = Particle(radius=1.0, update_radius=0.1, n_particles=30, starting_positions=positions)
        self.assertEqual(len(particle), 30)

    def test_invalid_particle_radius(self):
        with self.assertRaises(ValueError):
            Particle(radius=-1.0, update_radius=0.1, n_particles=30)

    def test_invalid_particle_update_radius(self):
        with self.assertRaises(ValueError):
            Particle(radius=1.0, update_radius=-0.1, n_particles=30)

    def test_invalid_particle_positions(self):
        with self.assertRaises(ValueError):
            Particle(radius=1.0, update_radius=0.1, n_particles=30, starting_positions=np.ones((30, 3)))

    def test_invalid_particle_positions_outside_radius(self):
        positions = generate_test_network_particle_positions(radius=1.0)
        positions[0] = [1.5, 0]  # Position outside the radius
        with self.assertRaises(ValueError):
            Particle(radius=1.0, update_radius=0.1, n_particles=30, starting_positions=positions)

    def test_invalid_particle_positions_equal_to_radius(self):
        positions = generate_test_network_particle_positions(radius=1.0)
        positions[0] = [1.1, 0]  # Position on the radius
        with self.assertRaises(ValueError):
            Particle(radius=1.0, update_radius=0.1, n_particles=30, starting_positions=positions)

class TestGossip(unittest.TestCase):
    def test_valid_gossip_initialization(self):
        network = generate_test_network()
        positions = generate_test_network_particle_positions()
        social_network = SocialNetwork(network)
        particles = Particle(radius=1.0, update_radius=0.1, n_particles=30, starting_positions=positions)
        gossip = Gossip(social_network=social_network, particles=particles, iterations=100)
        self.assertEqual(len(gossip.infected_particles), 30)

    def test_invalid_gossip_missing_particles(self):
        network = generate_test_network()
        with self.assertRaises(ValueError):
            Gossip(social_network=SocialNetwork(network), particles=None, iterations=100)

    def test_invalid_gossip_missing_network(self):
        positions = generate_test_network_particle_positions()
        with self.assertRaises(ValueError):
            Gossip(social_network=None, particles=Particle(radius=1.0, update_radius=0.1, n_particles=30, starting_positions=positions), iterations=100)

    def test_invalid_gossip_iterations_type(self):
        network = generate_test_network()
        positions = generate_test_network_particle_positions()
        social_network = SocialNetwork(network)
        particles = Particle(radius=1.0, update_radius=0.1, n_particles=30, starting_positions=positions)
        with self.assertRaises(TypeError):
            Gossip(social_network=social_network, particles=particles, iterations="100")

    def test_gossip_iteration(self):
        network = generate_test_network()
        positions = generate_test_network_particle_positions()
        social_network = SocialNetwork(network)
        particles = Particle(radius=1.0, update_radius=0.1, n_particles=30, starting_positions=positions)
        gossip = Gossip(social_network=social_network, particles=particles, iterations=100)

        gossip.iteration()
        self.assertEqual(gossip.current_iteration, 1)

    def test_gossip_animation(self):
        network = generate_test_network()
        positions = generate_test_network_particle_positions()
        social_network = SocialNetwork(network)
        particles = Particle(radius=1.0, update_radius=0.1, n_particles=30, starting_positions=positions)
        gossip = Gossip(social_network=social_network, particles=particles, iterations=5)

        gossip.animate_system_evolution(filename='test_animation.gif')
        # Manual check for the generated animation file

if __name__ == '__main__':
    unittest.main()
