import numpy as np


class OnlineDude:
    """Unbalanced DUDE algorithm based on the paper "Online Denoising of Discrete Noisy Data"
    We implement the One-time Online Denoising (OOD) variant of the algorithm, with delta=0.
    """
    def __init__(self, channel_transition_matrix: np.ndarray, loss_matrix: np.ndarray, k: int, hard_decision: bool = True,
                 alphabet_size: int = 2):
        """

        :param channel_transition_matrix: the i,j entry is the probability of receiving j given that the true input is i
        :param loss_matrix: the i,j entry is the loss of estimating j given that the true input is i
        :param k: the left context length is k
        :param hard_decision: whether to use hard decisions or soft decisions. Hard decisions output 0 or 1, soft decisions
        output a probability of the input being 1.
        :param alphabet_size: the size of the alphabet
        """
        self.channel_transition_matrix = channel_transition_matrix
        self.inverse_channel_transition_matrix = np.linalg.inv(channel_transition_matrix)

        self.loss_matrix = loss_matrix
        self.k = k
        self.alphabet_size = alphabet_size
        self.contexts = {}
        self._last_context: tuple[int, ...] = ()
        self.hard_decision = hard_decision

    def denoise_sample(self, observation: int):
        """
        :param observation: a list of observations
        :return: a list of states
        """
        if observation not in range(self.alphabet_size):
            raise ValueError(f"Observation must be in the range 0 to {self.alphabet_size - 1}")
        if len(self._last_context) < self.k:
            # update the context
            self._last_context += (observation,)
            if self.hard_decision:
                return observation, 0
            else:
                return np.reshape(np.eye(self.alphabet_size)[observation],(1,-1)), 0
        if self._last_context not in self.contexts:
            self.contexts[self._last_context] = np.zeros(self.alphabet_size)
        self.contexts[self._last_context][observation] += 1
        empirical_p = self.contexts[self._last_context][np.newaxis]
        # update the context
        self._last_context = self._last_context[1:] + (observation,)
        # estimate the state
        if self.hard_decision:
            posterior_p = np.matmul(empirical_p, self.inverse_channel_transition_matrix) * self.channel_transition_matrix[
                                                                                           :, observation]
            # posterior_p = self.smooth_posterior(empirical_p, observation)
            cost = np.matmul(posterior_p, self.loss_matrix)[0]
            estimate = np.argmin(cost)
            return estimate, cost[estimate]
        else:
            # The estimated posteriors using this approach could have negative values, which is not a valid probability.
            # To fix this, we smoothen the posterior to ensure positive probabilities.
            posterior_p = self.smooth_posterior(empirical_p, observation)
            cost = np.matmul(np.matmul(posterior_p, self.loss_matrix), posterior_p.T)
            return posterior_p, cost[0, 0]

    def denoise_sequence(self, observations: np.ndarray):
        """Denoise a sequence of observations

        :param observations: a list of observations
        :return: a list of states
        """
        if self.hard_decision:
            estimates = np.zeros_like(observations)
        else:
            estimates = np.zeros((len(observations), self.alphabet_size))
        costs = np.zeros_like(observations,dtype=np.float_)
        for i in range(len(observations)):
            estimates[i], costs[i] = self.denoise_sample(observations[i])
        return estimates, costs

    def smooth_posterior(self, empirical_p: np.ndarray, observation: int) -> np.ndarray:
        """Smoothen a binary posterior to ensure positive probabilities

        For details, see the paper "Universal Algorithms for Channel Decoding of Uncompressed Sources"
        :param empirical_p: the posterior to smoothen
        :param observation: the observation
        :return: the smoothened posterior
        """
        tmp = np.matmul(empirical_p, self.inverse_channel_transition_matrix)
        if tmp[0, 0] < 1:
            tmp[0, 0] = 1
            tmp[0, 1] = np.sum(tmp)-1
        elif tmp[0, 1] < 1:
            tmp[0, 1] = 1
            tmp[0, 0] = np.sum(tmp)-1
        tmp *= self.channel_transition_matrix[:, observation]
        return tmp / np.sum(tmp)

    def initial_context(self, context: tuple[int, ...]) -> None:
        """Sets the initial context

        :param context: the initial context
        """
        if len(context) != self.k:
            raise ValueError("Context must be of length k")
        self._last_context = context

    def reset(self) -> None:
        """Resets the context"""
        self._last_context = ()
        self.contexts = {}


class DUDE:
    """DUDE algorithm based on the paper "Universal Discrete Denoising: Known Channel" by Weissman et al.
    """
    def __init__(self, channel_transition_matrix: np.ndarray, loss_matrix: np.ndarray, k: int, hard_decision: bool = True,
                 alphabet_size: int = 2):
        """

        :param channel_transition_matrix: the i,j entry is the probability of receiving j given that the true input is i
        :param loss_matrix: the i,j entry is the loss of estimating j given that the true input is i
        :param k: the context length is 2k+1
        :param hard_decision: whether to use hard decisions or soft decisions. Hard decisions output 0 or 1, soft decisions
        output a probability of the input being 1.
        :param alphabet_size: the size of the alphabet
        """
        self.channel_transition_matrix = channel_transition_matrix
        self.inverse_channel_transition_matrix = np.linalg.inv(channel_transition_matrix)

        self.loss_matrix = loss_matrix
        self.k = k
        self.alphabet_size = alphabet_size
        self.hard_decision = hard_decision

    def denoise_sample(self, observation: int):
        """
        :param observation: a list of observations
        :return: a list of states
        """
        if self._last_context is None:
            raise RuntimeError("Must initially set context before denoising")
        if observation not in range(self.alphabet_size):
            raise ValueError(f"Observation must be in the range 0 to {self.alphabet_size - 1}")
        if self._last_context not in self.contexts:
            self.contexts[self._last_context] = np.zeros(self.alphabet_size)
        self.contexts[self._last_context][observation] += 1

        empirical_p = self.contexts[self._last_context][np.newaxis]
        # update the context
        self._last_context = self._last_context[1:] + (observation,)
        # estimate the state
        if self.hard_decision:
            posterior_p = np.matmul(empirical_p, self.inverse_channel_transition_matrix) * self.channel_transition_matrix[
                                                                                           :, observation]
            # posterior_p = self.smooth_posterior(empirical_p, observation)
            cost = np.matmul(posterior_p, self.loss_matrix)
            estimate = np.argmin(cost)
            return estimate, cost[0, estimate]
        else:
            # The estimated posteriors using this approach could have negative values, which is not a valid probability.
            # To fix this, we smoothen the posterior to ensure positive probabilities.
            posterior_p = self.smooth_posterior(empirical_p, observation)
            cost = np.matmul(np.matmul(posterior_p, self.loss_matrix), posterior_p.T)
            return posterior_p, cost[0, 0]

    def denoise_sequence(self, observations: np.ndarray) -> tuple[np.ndarray, float]:
        """Denoise a sequence of observations

        :param observations: a list of observations
        :return: a list of states
        """
        n = len(observations)
        if self.hard_decision:
            estimates = np.zeros_like(observations)
            estimates[:self.k] = observations[:self.k]
            estimates[-self.k:] = observations[-self.k:]
        else:
            estimates = np.zeros((len(observations), self.alphabet_size))
            estimates[:self.k] = np.eye(self.alphabet_size)[observations[:self.k]]
            estimates[-self.k:] = np.eye(self.alphabet_size)[observations[-self.k:]]

        costs = np.zeros_like(observations,dtype=np.float_)
        # empirical distribution
        contexts = {}
        for i in range(self.k, n-self.k):
            context:tuple[tuple[int, ...], tuple[int, ...]] = tuple(observations[i-self.k:i]), \
                tuple(observations[i+1:i+self.k+1])
            if context not in contexts:
                contexts[context] = np.zeros(self.alphabet_size)
            contexts[context][observations[i]] += 1
        # normalized empirical distribution will result in problem with the smoothing operation, to use need to alter smoothing
        # per article. Currently not used as it isn't expected to have any significant effect.
        # # normalize the empirical distribution
        # for context in contexts:
        #     contexts[context] /= np.sum(contexts[context])

        #denoise the sequence
        for i in range(self.k, n-self.k):
            # compute the posterior
            context = tuple(observations[i-self.k:i]), tuple(observations[i+1:i+self.k+1])
            observation = observations[i]
            empirical_p = contexts[context][np.newaxis]
            if self.hard_decision:
                posterior_p = np.matmul(empirical_p, self.inverse_channel_transition_matrix) * self.channel_transition_matrix[
                                                                                               :, observation]
                # posterior_p = self.smooth_posterior(empirical_p, observation)
                cost= np.matmul(posterior_p, self.loss_matrix)[0]
                estimates[i] = np.argmin(cost)
                costs[i] = cost[estimates[i]]
            else:
                # The estimated posteriors using this approach could have negative values, which is not a valid probability.
                # To fix this, we smoothen the posterior to ensure positive probabilities.
                posterior_p = self.smooth_posterior(empirical_p, observation)
                cost = np.matmul(np.matmul(posterior_p, self.loss_matrix), posterior_p.T)
                estimates[i], costs[i] = posterior_p, cost[0, 0]
        return estimates, costs.mean()
    def smooth_posterior(self, empirical_p: np.ndarray, observation: int) -> np.ndarray:
        """Smoothen a binary posterior to ensure positive probabilities

        For details, see the paper "Universal Algorithms for Channel Decoding of Uncompressed Sources"
        :param empirical_p: the posterior to smoothen
        :param observation: the observation
        :return: the smoothened posterior
        """
        tmp = np.matmul(empirical_p, self.inverse_channel_transition_matrix)
        if tmp[0, 0] < 1:
            tmp[0, 0] = 1
            tmp[0, 1] = np.sum(tmp)-1
        elif tmp[0, 1] < 1:
            tmp[0, 1] = 1
            tmp[0, 0] = np.sum(tmp)-1
        tmp *= self.channel_transition_matrix[:, observation]
        return tmp / np.sum(tmp)


__all__ = ["OnlineDude", "DUDE"]
