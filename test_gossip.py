from gossip_network import *
from create_test_cases import *

def simulate_gossip_scenario(matrix_size, fill_percentage, radius, n_particles, iterations, infection_radius, update_radius, spread_rate, title, filename):
    social_network = SocialNetwork(generate_test_network(matrix_size=matrix_size, fill_percentage=fill_percentage))
    positions = generate_test_network_particle_positions(radius=radius, n_particles=n_particles)
    initial_infections = np.random.choice([0, 1], size=N, p=[1 - p, p])
    
    particles = Particle(radius=radius, update_radius=update_radius, starting_positions=positions)

    gossip = Gossip(social_network=social_network, particles=particles, iterations=iterations, infection_radius=infection_radius,
                    movement_radius=update_radius, infectable_particles=initial_infections, spread_rate=spread_rate, title=title)

    gossip.animate_system_evolution(filename=filename)

# Set parameters
R = 5.
N = 100
I = 500
IR = 0.3
UR = 0.3
p = 0.2

# Simulate scenarios
simulate_gossip_scenario(matrix_size=N, fill_percentage=1.0, radius=R, n_particles=N, iterations=I,
                         infection_radius=IR, update_radius=UR, spread_rate=0.001,
                         title='Everyone connected, spicy gossip', filename='everyone_connected_spicy_gossip.gif')

simulate_gossip_scenario(matrix_size=N, fill_percentage=1.0, radius=R, n_particles=N, iterations=I,
                         infection_radius=IR, update_radius=UR, spread_rate=0.1,
                         title='Everyone connected, dull gossip', filename='everyone_connected_dull_gossip.gif')

simulate_gossip_scenario(matrix_size=N, fill_percentage=0.5, radius=R, n_particles=N, iterations=I,
                         infection_radius=IR, update_radius=UR, spread_rate=0.001,
                         title='Half connected, spicy gossip', filename='half_connected_spicy_gossip.gif')

simulate_gossip_scenario(matrix_size=N, fill_percentage=0.5, radius=R, n_particles=N, iterations=I,
                         infection_radius=IR, update_radius=UR, spread_rate=0.1,
                         title='Half connected, dull gossip', filename='half_connected_dull_gossip.gif')


simulate_gossip_scenario(matrix_size=N, fill_percentage=0.2, radius=R, n_particles=N, iterations=I,
                         infection_radius=IR, update_radius=UR, spread_rate=0.001,
                         title='Sparsely connected, spicy gossip', filename='sparsely_connected_spicy_gossip.gif')

simulate_gossip_scenario(matrix_size=N, fill_percentage=0.2, radius=R, n_particles=N, iterations=I,
                         infection_radius=IR, update_radius=UR, spread_rate=0.1,
                         title='Sparsely connected, dull gossip', filename='sparsely_connected_dull_gossip.gif')
