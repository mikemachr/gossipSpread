import numpy as np
def generate_test_network(matrix_size=30, fill_percentage=0.4):
    """Generates a test case for the social network."""
    num_non_zero_entries = int(fill_percentage * matrix_size * (matrix_size - 1) / 2)

    indices = np.triu_indices(matrix_size, k=1)
    selected_indices = np.random.choice(range(len(indices[0])), size=num_non_zero_entries, replace=False)
    row_indices, col_indices = indices[0][selected_indices], indices[1][selected_indices]

    symmetric_matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    symmetric_matrix[row_indices, col_indices] = symmetric_matrix[col_indices, row_indices] = 1

    return symmetric_matrix

def generate_test_network_particle_positions(radius=1.0, n_particles=30):
    """Initializes particle positions within a circle."""
    angles = np.linspace(0, 2 * np.pi, n_particles)
    radii = np.sqrt(np.random.uniform(0, radius**2, n_particles))  
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    positions = np.column_stack((x, y))

    return positions
