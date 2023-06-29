from inference import OnlineDude, DUDE, EMLE
import numpy as np
from utils.random_sources import generate_markov_samples

bit_flip_p = 0.2
n = 10**5
channel_transition_matrix = np.array([[1-bit_flip_p, bit_flip_p], [bit_flip_p, 1-bit_flip_p]])
loss_matrix = np.array([[0, 1], [1, 0]])
window_length = 50
thr = 0.36

k = int(0.5*np.log2(n))
online_dude = OnlineDude(channel_transition_matrix, loss_matrix, k, hard_decision=True)
dude = DUDE(channel_transition_matrix, loss_matrix, k//2, hard_decision=True)
emle = EMLE(bit_flip_p, thr, window_length, hard_decision=True, soft_input=False)
# generate a random sequence of 0s and 1s (uniformly distributed)
sequence = np.random.randint(0, 2, n)
# set the initial context
online_dude.initial_context(tuple(sequence[:k]))
# denoise a sample
estimate, cost = online_dude.denoise_sample(sequence[k])

online_dude.reset()
# denoise the sequence
estimates, costs = online_dude.denoise_sequence(sequence)
online_dude.reset()
print("----------Estimates-Clean-Sequence----------")
print(f"Online DUDE errors: {np.sum(estimates != sequence)}")
estimates, costs = dude.denoise_sequence(sequence)
print(f"DUDE errors: {np.sum(estimates != sequence)}")
estimates, costs = emle.denoise_sequence(np.reshape(sequence, (n, 1)))
emle.reset()
print(f"EMLE errors: {np.sum(estimates.flatten() != sequence)}")


# Now actually add noise to the sequence
noise = np.random.binomial(1, bit_flip_p, n)
noisy_sequence = (sequence + noise) % 2
print("----------Estimates-Noisy-Sequence----------")
print(f"Noise errors: {np.sum(noisy_sequence != sequence)}")
# denoise the noisy sequence
estimates, costs = online_dude.denoise_sequence(noisy_sequence)
online_dude.reset()
print(f"Online DUDE errors: {np.sum(estimates != sequence)}")
print(f"Online DUDE flips: {np.sum(estimates != noisy_sequence)}")
estimates, costs = dude.denoise_sequence(noisy_sequence)
print(f"DUDE errors: {np.sum(estimates != sequence)}")
print(f"DUDE flips: {np.sum(estimates != noisy_sequence)}")
estimates, costs = emle.denoise_sequence(np.reshape(sequence, (n, 1)))
emle.reset()
print(f"EMLE errors: {np.sum(estimates.flatten() != sequence)}")
print(f"EMLE flips: {np.sum(estimates.flatten() != noisy_sequence)}")

# now try a Fixed source
k = 1
online_dude = OnlineDude(channel_transition_matrix, loss_matrix, 2*k, hard_decision=True)
dude = DUDE(channel_transition_matrix, loss_matrix, k, hard_decision=True)
sequence = np.ones(n, dtype=np.int_)
noise = np.random.binomial(1, bit_flip_p, n)
noisy_sequence = (sequence + noise) % 2
print("----------Estimates-Fixed-Source----------")
print(f"Noise errors: {np.sum(noisy_sequence != sequence)}")
estimates, costs = online_dude.denoise_sequence(noisy_sequence)
online_dude.reset()
print(f"Online DUDE errors: {np.sum(estimates != sequence)}")
print(f"Online DUDE flips: {np.sum(estimates != noisy_sequence)}")
estimates, costs = dude.denoise_sequence(np.array([1,1,0,0,1,1]))
estimates, costs = dude.denoise_sequence(noisy_sequence)
print(f"DUDE errors: {np.sum(estimates != sequence)}")
print(f"DUDE flips: {np.sum(estimates != noisy_sequence)}")
estimates, costs = emle.denoise_sequence(np.reshape(sequence, (n, 1)))
emle.reset()
print(f"EMLE errors: {np.sum(estimates.flatten() != sequence)}")
print(f"EMLE flips: {np.sum(estimates.flatten() != noisy_sequence)}")

# now try a Bernoulli source
k = 1
# factor = 2
# channel_transition_matrix2 = np.array([[1-bit_flip_p*factor, bit_flip_p*factor], [bit_flip_p*factor, 1-bit_flip_p*factor]])
online_dude = OnlineDude(channel_transition_matrix, loss_matrix, 2*k, hard_decision=True)
dude = DUDE(channel_transition_matrix, loss_matrix, k, hard_decision=True)
p = 0.9
sequence = np.random.binomial(1, p, n)
noise = np.random.binomial(1, bit_flip_p, n)
noisy_sequence = (sequence + noise) % 2
print("----------Estimates-Bernoulli-Source----------")
print(f"Noise errors: {np.sum(noisy_sequence != sequence)}")
estimates, costs = online_dude.denoise_sequence(noisy_sequence)
online_dude.reset()
print(f"Online DUDE errors: {np.sum(estimates != sequence)}")
print(f"Online DUDE flips: {np.sum(estimates != noisy_sequence)}")
estimates, costs = dude.denoise_sequence(noisy_sequence)
print(f"DUDE errors: {np.sum(estimates != sequence)}")
print(f"DUDE flips: {np.sum(estimates != noisy_sequence)}")
estimates, costs = emle.denoise_sequence(np.reshape(sequence, (n, 1)))
emle.reset()
print(f"EMLE errors: {np.sum(estimates.flatten() != sequence)}")
print(f"EMLE flips: {np.sum(estimates.flatten() != noisy_sequence)}")


# now try a Markov source
k = 8
online_dude = OnlineDude(channel_transition_matrix, loss_matrix, k, hard_decision=True)
dude = DUDE(channel_transition_matrix, loss_matrix, k//2, hard_decision=True)
p = 0.01
transition_matrix = np.array([[1-p, p], [p, 1-p]])
sequence = np.zeros(n, dtype=np.int_)
for i in range(1, n):
    sequence[i] = np.random.binomial(1, transition_matrix[sequence[i-1], 1])
noise = np.random.binomial(1, bit_flip_p, n)
noisy_sequence = (sequence + noise) % 2
print("----------Estimates-Markov-Source----------")
print(f"Noise errors: {np.sum(noisy_sequence != sequence)}")
estimates, costs = online_dude.denoise_sequence(noisy_sequence)
online_dude.reset()
print(f"Online DUDE errors: {np.sum(estimates != sequence)}")
print(f"Online DUDE flips: {np.sum(estimates != noisy_sequence)}")
estimates, costs = dude.denoise_sequence(noisy_sequence)
print(f"DUDE errors: {np.sum(estimates != sequence)}")
print(f"DUDE flips: {np.sum(estimates != noisy_sequence)}")
estimates, costs = emle.denoise_sequence(np.reshape(sequence, (n, 1)))
emle.reset()
print(f"EMLE errors: {np.sum(estimates.flatten() != sequence)}")
print(f"EMLE flips: {np.sum(estimates.flatten() != noisy_sequence)}")


# now try a second order Markov source
k = 8
online_dude = OnlineDude(channel_transition_matrix, loss_matrix, k, hard_decision=False)
dude = DUDE(channel_transition_matrix, loss_matrix, k//2, hard_decision=True)
transition_matrix = np.array([[[0.99, 0.01], [0.01, 0.99]], [[0.9, 0.1], [0.1, 0.9]]])
initial_states = [0, 0]
sequence = generate_markov_samples(transition_matrix, initial_states, n)
noise = np.random.binomial(1, bit_flip_p, n)
noisy_sequence = (sequence + noise) % 2
print("----------Estimates-Second-Order-Markov-Source----------")
print(f"Noise errors: {np.sum(noisy_sequence != sequence)}")
soft_estimates, costs = online_dude.denoise_sequence(noisy_sequence)
online_dude.reset()
estimates = soft_estimates.argmax(axis=1)
erroneous_flips = (estimates != noisy_sequence) & (estimates != sequence)
good_flips = (estimates != noisy_sequence) & (estimates == sequence)
print(f"Online DUDE errors: {np.sum(estimates != sequence)}")
print(f"Online DUDE flips: {np.sum(estimates != noisy_sequence)}")
print(f"Online DUDE erroneous flips: {np.sum(erroneous_flips)}")
print(f"Online DUDE good flips: {np.sum(good_flips)}")

estimates, costs = dude.denoise_sequence(noisy_sequence)
erroneous_flips = (estimates != noisy_sequence) & (estimates != sequence)
good_flips = (estimates != noisy_sequence) & (estimates == sequence)
print(f"DUDE errors: {np.sum(estimates != sequence)}")
print(f"DUDE flips: {np.sum(estimates != noisy_sequence)}")
print(f"DUDE erroneous flips: {np.sum(erroneous_flips)}")
print(f"DUDE good flips: {np.sum(good_flips)}")

estimates, costs = emle.denoise_sequence(np.reshape(sequence, (n, 1)))
emle.reset()
print(f"EMLE errors: {np.sum(estimates.flatten() != sequence)}")
print(f"EMLE flips: {np.sum(estimates.flatten() != noisy_sequence)}")