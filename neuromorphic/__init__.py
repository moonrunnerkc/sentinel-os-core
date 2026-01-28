# Author: Bradley R. Kinnard
# neuromorphic processing module for spiking neural network computation

from dataclasses import dataclass
from typing import Any, Callable
import time

import numpy as np

from utils.helpers import get_logger

logger = get_logger(__name__)


# check if brian2 is available
_BRIAN2_AVAILABLE = False
try:
    from brian2 import (
        NeuronGroup,
        Synapses,
        SpikeMonitor,
        StateMonitor,
        Network,
        ms,
        mV,
        Hz,
        defaultclock,
        start_scope,
    )
    _BRIAN2_AVAILABLE = True
    logger.info("brian2 available - neuromorphic mode enabled")
except ImportError:
    logger.warning("brian2 not available - neuromorphic operations will use mock mode")


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
    energy_estimate_mj: float  # estimated energy in millijoules


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

    uses brian2 when available, mock mode otherwise.
    """

    def __init__(
        self,
        n_input: int = 100,
        n_hidden: int = 50,
        n_output: int = 10,
        simulation_dt: float = 0.1,  # ms
    ):
        self._n_input = n_input
        self._n_hidden = n_hidden
        self._n_output = n_output
        self._dt = simulation_dt
        self._mock_mode = not _BRIAN2_AVAILABLE

        if not self._mock_mode:
            self._init_network()
        else:
            logger.warning("SNN kernel running in mock mode")

    def _init_network(self) -> None:
        """initialize brian2 network."""
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
        start_time = time.time()
        sim_duration = duration or input_pattern.duration

        if self._mock_mode:
            return self._mock_run(input_pattern, sim_duration)

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

        # estimate energy (very rough: ~1pJ per spike)
        n_spikes = len(output_spikes)
        energy_mj = n_spikes * 1e-9  # 1pJ per spike

        return SNNResult(
            output_spikes=output_spikes,
            output_rates=output_rates,
            computation_time_ms=elapsed,
            energy_estimate_mj=energy_mj,
        )

    def _mock_run(
        self,
        input_pattern: SpikePattern,
        duration: float,
    ) -> SNNResult:
        """mock SNN computation."""
        np.random.seed(42)

        # simulate some output based on input
        n_input_spikes = len(input_pattern.spike_times)
        output_rate = n_input_spikes / duration * 10  # arbitrary scaling

        output_spikes = []
        output_rates = []

        for i in range(self._n_output):
            rate = output_rate * (0.5 + 0.5 * np.random.random())
            n_spikes = np.random.poisson(rate * duration / 1000)

            for _ in range(n_spikes):
                t = np.random.uniform(0, duration)
                output_spikes.append((i, t))

            output_rates.append(rate)

        return SNNResult(
            output_spikes=output_spikes,
            output_rates=output_rates,
            computation_time_ms=1.0,  # mock: 1ms
            energy_estimate_mj=len(output_spikes) * 1e-9,
        )


class ContradictionClassifier:
    """
    SNN-based contradiction classifier.

    classifies belief pairs as:
    - compatible (0)
    - contradictory (1)
    - uncertain (0.5)
    """

    def __init__(self):
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
