# Author: Bradley R. Kinnard
# neuromorphic processing module for spiking neural network computation
# requires brian2 - no mock mode, fails explicitly if unavailable

from dataclasses import dataclass
from typing import Any
import time

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


class NeuromorphicUnavailableError(ImportError):
    """raised when neuromorphic operations are attempted without brian2."""
    pass


def _check_brian2():
    """check if brian2 is available."""
    try:
        import brian2  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class SpikePattern:
    """encoded spike pattern for SNN input."""
    neuron_indices: list[int]
    spike_times: list[float]  # in ms
    duration: float  # total duration in ms
    encoding: str  # "rate", "temporal", "population"


@dataclass
class SNNResult:
    """result from SNN computation."""
    output_spikes: list[tuple[int, float]]  # (neuron_idx, time_ms)
    output_rates: list[float]  # firing rates per output neuron
    computation_time_ms: float
    # energy_estimate removed - not measurable in software


class RateEncoder:
    """encode scalar values as spike rates."""

    def __init__(
        self,
        n_neurons: int = 100,
        max_rate: float = 100.0,  # Hz
        duration: float = 100.0,  # ms
    ):
        self._n_neurons = n_neurons
        self._max_rate = max_rate
        self._duration = duration

    def encode(self, value: float, seed: int | None = None) -> SpikePattern:
        """encode a scalar [0, 1] as a spike pattern."""
        if seed is not None:
            np.random.seed(seed)

        # clamp value to [0, 1]
        value = max(0.0, min(1.0, value))

        # compute firing rate
        rate = value * self._max_rate

        # generate Poisson spikes
        neuron_indices = []
        spike_times = []

        for i in range(self._n_neurons):
            # expected number of spikes
            expected = rate * (self._duration / 1000)

            # sample from Poisson
            n_spikes = np.random.poisson(expected)

            for _ in range(n_spikes):
                t = np.random.uniform(0, self._duration)
                neuron_indices.append(i)
                spike_times.append(t)

        return SpikePattern(
            neuron_indices=neuron_indices,
            spike_times=spike_times,
            duration=self._duration,
            encoding="rate",
        )

    def decode(self, result: SNNResult) -> float:
        """decode SNN output to scalar."""
        if not result.output_rates:
            return 0.0

        # average firing rate normalized by max
        avg_rate = np.mean(result.output_rates)
        return min(1.0, avg_rate / self._max_rate)


class PopulationEncoder:
    """encode values using population coding."""

    def __init__(
        self,
        n_neurons: int = 100,
        min_value: float = 0.0,
        max_value: float = 1.0,
        duration: float = 50.0,
    ):
        self._n_neurons = n_neurons
        self._min_value = min_value
        self._max_value = max_value
        self._duration = duration

        # gaussian tuning curves
        self._centers = np.linspace(min_value, max_value, n_neurons)
        self._sigma = (max_value - min_value) / (n_neurons * 2)

    def encode(self, value: float, seed: int | None = None) -> SpikePattern:
        """encode value using population code."""
        if seed is not None:
            np.random.seed(seed)

        # compute activation for each neuron based on tuning curve
        activations = np.exp(-((value - self._centers) ** 2) / (2 * self._sigma ** 2))

        neuron_indices = []
        spike_times = []

        for i, act in enumerate(activations):
            # spike probability proportional to activation
            if np.random.random() < act:
                t = np.random.uniform(0, self._duration)
                neuron_indices.append(i)
                spike_times.append(t)

        return SpikePattern(
            neuron_indices=neuron_indices,
            spike_times=spike_times,
            duration=self._duration,
            encoding="population",
        )

    def decode(self, result: SNNResult) -> float:
        """decode using center of mass of active neurons."""
        if not result.output_spikes:
            return (self._min_value + self._max_value) / 2

        # find which neurons spiked
        active_neurons = set(idx for idx, _ in result.output_spikes)

        if not active_neurons:
            return (self._min_value + self._max_value) / 2

        # center of mass
        total = sum(self._centers[i] for i in active_neurons if i < len(self._centers))
        return total / len(active_neurons)


class SNNKernel:
    """
    spiking neural network computation kernel.

    provides:
    - contradiction classification
    - belief similarity scoring
    - anomaly detection

    REQUIRES brian2 - no mock mode. Will raise NeuromorphicUnavailableError if brian2 missing.
    """

    def __init__(
        self,
        n_input: int = 100,
        n_hidden: int = 50,
        n_output: int = 10,
        simulation_dt: float = 0.1,  # ms
    ):
        if not _check_brian2():
            raise NeuromorphicUnavailableError(
                "brian2 is required for neuromorphic operations. "
                "Install with: pip install brian2"
            )

        self._n_input = n_input
        self._n_hidden = n_hidden
        self._n_output = n_output
        self._dt = simulation_dt

        self._init_network()

    def _init_network(self) -> None:
        """initialize brian2 network."""
        from brian2 import (
            NeuronGroup,
            Synapses,
            SpikeMonitor,
            Network,
            ms,
            mV,
            start_scope,
        )

        start_scope()

        # LIF neuron model
        eqs = """
        dv/dt = (v_rest - v + I) / tau : volt
        I : volt
        """

        # input layer
        self._input_group = NeuronGroup(
            self._n_input,
            eqs,
            threshold="v > v_thresh",
            reset="v = v_reset",
            method="euler",
            namespace={
                "v_rest": -65 * mV,
                "v_thresh": -50 * mV,
                "v_reset": -65 * mV,
                "tau": 10 * ms,
            },
        )

        # hidden layer
        self._hidden_group = NeuronGroup(
            self._n_hidden,
            eqs,
            threshold="v > v_thresh",
            reset="v = v_reset",
            method="euler",
            namespace={
                "v_rest": -65 * mV,
                "v_thresh": -50 * mV,
                "v_reset": -65 * mV,
                "tau": 10 * ms,
            },
        )

        # output layer
        self._output_group = NeuronGroup(
            self._n_output,
            eqs,
            threshold="v > v_thresh",
            reset="v = v_reset",
            method="euler",
            namespace={
                "v_rest": -65 * mV,
                "v_thresh": -50 * mV,
                "v_reset": -65 * mV,
                "tau": 10 * ms,
            },
        )

        # synapses
        self._syn_ih = Synapses(
            self._input_group,
            self._hidden_group,
            "w : volt",
            on_pre="v_post += w",
        )
        self._syn_ih.connect(p=0.3)
        self._syn_ih.w = "5*mV * rand()"

        self._syn_ho = Synapses(
            self._hidden_group,
            self._output_group,
            "w : volt",
            on_pre="v_post += w",
        )
        self._syn_ho.connect(p=0.3)
        self._syn_ho.w = "5*mV * rand()"

        # monitors
        self._spike_mon = SpikeMonitor(self._output_group)

        # network
        self._network = Network(
            self._input_group,
            self._hidden_group,
            self._output_group,
            self._syn_ih,
            self._syn_ho,
            self._spike_mon,
        )

        logger.info(f"initialized SNN: {self._n_input}-{self._n_hidden}-{self._n_output}")

    def run(
        self,
        input_pattern: SpikePattern,
        duration: float | None = None,
    ) -> SNNResult:
        """run SNN on input pattern."""
        from brian2 import ms, mV

        start_time = time.time()
        sim_duration = duration or input_pattern.duration

        # inject spikes
        # (simplified: set input currents based on spike counts)
        spike_counts = np.zeros(self._n_input)
        for idx in input_pattern.neuron_indices:
            if idx < self._n_input:
                spike_counts[idx] += 1

        self._input_group.I = spike_counts * 5 * mV

        # run simulation
        self._network.run(sim_duration * ms)

        # collect output
        output_spikes = list(zip(
            self._spike_mon.i.astype(int).tolist(),
            (self._spike_mon.t / ms).tolist(),
        ))

        # compute rates
        output_rates = []
        for i in range(self._n_output):
            count = sum(1 for idx, _ in output_spikes if idx == i)
            rate = count / (sim_duration / 1000)  # Hz
            output_rates.append(rate)

        elapsed = (time.time() - start_time) * 1000

        return SNNResult(
            output_spikes=output_spikes,
            output_rates=output_rates,
            computation_time_ms=elapsed,
        )


class ContradictionClassifier:
    """
    SNN-based contradiction classifier.

    classifies belief pairs as:
    - compatible (0)
    - contradictory (1)
    - uncertain (0.5)

    REQUIRES brian2 - raises NeuromorphicUnavailableError if missing.
    """

    def __init__(self):
        if not _check_brian2():
            raise NeuromorphicUnavailableError(
                "brian2 is required for ContradictionClassifier. "
                "Install with: pip install brian2"
            )

        self._encoder = PopulationEncoder(n_neurons=50)
        self._kernel = SNNKernel(n_input=100, n_hidden=30, n_output=3)

    def classify(
        self,
        belief_a_embedding: list[float],
        belief_b_embedding: list[float],
    ) -> tuple[str, float, SNNResult]:
        """
        classify relationship between two belief embeddings.

        returns: (class_name, confidence, raw_result)
        """
        # combine embeddings
        combined = belief_a_embedding[:25] + belief_b_embedding[:25]
        combined_value = np.mean(combined) if combined else 0.5

        # encode as spikes
        pattern = self._encoder.encode(combined_value)

        # run SNN
        result = self._kernel.run(pattern)

        # decode output (3 output neurons: compatible, contradictory, uncertain)
        if not result.output_rates:
            return "uncertain", 0.5, result

        rates = result.output_rates[:3]
        while len(rates) < 3:
            rates.append(0.0)

        # softmax-like normalization
        total = sum(rates) + 1e-10
        probs = [r / total for r in rates]

        classes = ["compatible", "contradictory", "uncertain"]
        best_idx = np.argmax(probs)

        return classes[best_idx], probs[best_idx], result


def is_brian2_available() -> bool:
    """check if brian2 is available."""
    return _check_brian2()
