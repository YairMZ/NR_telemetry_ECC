"""Compare performance of Online DUDE and EMLE on a uniform source."""
import numpy as np
from inference import OnlineDude, EMLE
from utils.performance_analysis import stats, ber
from numpy.random import default_rng
from utils.random_sources import generate_markov_samples


def compare_denoisers(source: np.ndarray, f: float, k: int, threshold: float, comparison_name: str,
                      window_length: int = 0, conf_center: int = 40, conf_slope: float = 0.35):
    """Compare performance of Online DUDE and EMLE on a uniform source.

    :param source: the source to denoise. A 2D binary array of shape (num_buffers, buffer_length)
    :param f: channel bit flip probability
    :param k: the context length to use for the online DUDE
    :param threshold: the threshold to use for the EMLE
    :param window_length: the window length to use for the EMLE
    :param conf_center: the center of the confidence function to use for the EMLE
    :param conf_slope: the slope of the confidence function to use for the EMLE
    :returns: a tuple (online_dude_stats, emle_stats, online_dude_ber, emle_ber) where online_dude_stats and emle_stats are
    tuples of the form (cm, recall, precision, f1, accuracy) of the denoising costs, and online_dude_ber and emle_ber are
    the BER of the denoised sequence.
    """
    channel_transition_matrix = np.array([[1 - f, f], [f, 1 - f]])
    loss_matrix = np.array([[0, 1], [1, 0]])
    # Online DUDE
    online_dude_estimates = np.zeros_like(source)
    for i in range(source.shape[1]):
        online_dude = OnlineDude(channel_transition_matrix, loss_matrix, k, hard_decision=True)
        online_dude_estimates[:, i], _ = online_dude.denoise_sequence(source[:, i])
    # EMLE
    emle = EMLE(f, threshold, window_length, conf_center, conf_slope, hard_decision=True, soft_input=False)
    emle_estimates, emle_costs = emle.denoise_sequence(source)
    # Compute stats
    online_dude_stats = stats(source.flatten(), online_dude_estimates.flatten())
    emle_stats = stats(source.flatten(), emle_estimates.flatten())
    # Compute BER
    online_dude_ber = ber(source.flatten(), online_dude_estimates.flatten())
    emle_ber = ber(source.flatten(), emle_estimates.flatten())

    print(f"----------Estimates-{comparison_name}----------")
    print(f"Online DUDE BER: {online_dude_ber}")
    print(f"Online DUDE flips: {np.sum(online_dude_estimates.flatten() != source.flatten())}")
    print(f"EMLE BER: {emle_ber}")
    print(f"EMLE flips: {np.sum(emle_estimates.flatten() != source.flatten())}")

    return online_dude_stats, emle_stats, online_dude_ber, emle_ber


rng = default_rng()
n = 10 ** 5
f = 0.2
model_length = 1
window_length = 0
# denoise clean uniform sequence
k = 4
source = rng.integers(0, 2, size=(n, model_length))
compare_denoisers(source, f, k, 0.36, "Clean-Sequence", window_length)

# denoise noisy uniform sequence
noise = rng.binomial(1, f, size=(n, model_length))
noisy_source = (source + noise) % 2
compare_denoisers(noisy_source, f, k, 0.36, "Uniform-Source", window_length)

# denoise a fixed source
k = 1
source = np.ones((n, model_length))
noise = rng.binomial(1, f, size=(n, model_length))
noisy_source = (source + noise) % 2
compare_denoisers(noisy_source, f, k, 0.36, "Fixed-Source", window_length)

# denoise a Bernoulli source
k = 1
p = 0.9
source = rng.binomial(1, p, size=(n, model_length))
noise = rng.binomial(1, f, size=(n, model_length))
noisy_source = (source + noise) % 2
compare_denoisers(noisy_source, f, k, 0.36, "Bernoulli-Source", window_length)

# denoise a Markov source
k = 8
p = 0.01
transition_matrix = np.array([[1 - p, p], [p, 1 - p]])
source = np.zeros((n, 1), dtype=np.int_)
for i in range(1, n):
    source[i] = rng.choice([0, 1], p=transition_matrix[source[i - 1]])
noise = rng.binomial(1, f, size=(n, 1))
noisy_source = (source + noise) % 2
compare_denoisers(noisy_source, f, k, 0.36, "Markov-Source", window_length)

# denoise a 2nd order Markov source
k = 8
transition_matrix = np.array([[[0.99, 0.01], [0.01, 0.99]], [[0.9, 0.1], [0.1, 0.9]]])
initial_state = np.array([0, 0])
source = generate_markov_samples(transition_matrix, initial_state, n).reshape((n, 1))
noise = rng.binomial(1, f, size=(n, 1))
noisy_source = (source + noise) % 2
compare_denoisers(noisy_source, f, k, 0.36, "2nd-Order-Markov-Source", window_length)
