import numpy as np


def generate_markov_samples(transition_matrix, initial_state, num_samples):
    num_states = transition_matrix.shape[-1]
    order = transition_matrix.ndim -1
    if order == 1 and type(initial_state) is int:
            pass
    elif order != len(initial_state):
        raise ValueError("The order of the transition matrix must match the order of the initial state")
    if num_states != transition_matrix.shape[-2]:
        raise ValueError("inconsistent dimensions of transition matrix")

    states = np.zeros(num_samples, dtype=np.int_)
    states[:order] = initial_state

    # Generate samples
    for i in range(order, num_samples):
        current_states = states[-order:]
        probabilities = transition_matrix[tuple(current_states)]
        states[i] = np.random.choice(range(num_states), p=probabilities)
    return states

__all__ = ["generate_markov_samples"]

if __name__ == "__main__":
    # First order chain:
    # Example transition matrix
    transition_matrix = np.array([[0.7, 0.15, 0.15], [0.4, 0.5, 0.1], [0.2, 0.3, 0.5]])
    # Example initial state
    initial_state = 0
    # Number of samples to generate
    num_samples = 100

    # Generate samples
    samples = generate_markov_samples(transition_matrix, initial_state, num_samples)

    # Print the generated samples
    print("Generated Samples:")
    print(samples)

    #Second order chain
    # Example transition matrix
    transition_matrix = np.array([[[0.2, 0.8], [0.6, 0.4]], [[0.7, 0.3], [0.5, 0.5]]])
    # Example initial states
    initial_states = [0, 0]
    # Number of samples to generate
    num_samples = 100

    # Generate samples
    samples = generate_markov_samples(transition_matrix, initial_states, num_samples)

    # Print the generated samples
    print("Generated Samples:")
    print(samples)
