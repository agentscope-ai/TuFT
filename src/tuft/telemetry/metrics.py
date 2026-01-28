"""Metrics utilities for TuFT.

Provides meter access and predefined metric instruments.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from typing import Any

import psutil
import pynvml
from opentelemetry import metrics
from opentelemetry.metrics import CallbackOptions, Observation


logger = logging.getLogger(__name__)

# Module-level meter cache
_meters: dict[str, Any] = {}


def get_meter(name: str = "tuft"):
    """Get a meter instance by name.

    Args:
        name: Name for the meter (typically module name).

    Returns:
        A Meter instance. When no MeterProvider is configured,
        OpenTelemetry automatically returns a NoOpMeter.
    """
    if name in _meters:
        return _meters[name]

    meter = metrics.get_meter(name)
    _meters[name] = meter
    return meter


def clear_meters() -> None:
    """Clear the meter cache. Used during shutdown."""
    _meters.clear()


class TuftMetrics:
    """Centralized metrics registry for TuFT.

    Provides access to all predefined metrics with proper typing.
    """

    _instance: "TuftMetrics | None" = None
    _lock = threading.Lock()

    def __init__(self):
        meter = get_meter("tuft")

        # Training metrics
        self.training_models_active = meter.create_up_down_counter(
            "tuft.training.models.active",
            description="Number of active training models",
        )

        self.training_tokens_per_second = meter.create_histogram(
            "tuft.training.tokens_per_second",
            description="Training tokens per second",
            unit="tokens/s",
        )

        self.training_checkpoints_saved = meter.create_counter(
            "tuft.training.checkpoints.saved",
            description="Number of checkpoints saved",
        )

        self.training_checkpoint_size = meter.create_histogram(
            "tuft.training.checkpoint.size_bytes",
            description="Checkpoint size in bytes",
            unit="bytes",
        )

        # Sampling metrics
        self.sampling_sessions_active = meter.create_up_down_counter(
            "tuft.sampling.sessions.active",
            description="Number of active sampling sessions",
        )

        self.sampling_requests = meter.create_counter(
            "tuft.sampling.requests",
            description="Number of sampling requests",
        )

        self.sampling_duration = meter.create_histogram(
            "tuft.sampling.duration",
            description="Sampling request duration",
            unit="s",
        )

        self.sampling_tokens_per_second = meter.create_histogram(
            "tuft.sampling.tokens_per_second",
            description="Sampling tokens per second",
            unit="tokens/s",
        )

        self.sampling_output_tokens = meter.create_histogram(
            "tuft.sampling.output_tokens",
            description="Number of output tokens per sample",
            unit="tokens",
        )

        # Future queue metrics
        self.futures_queue_length = meter.create_up_down_counter(
            "tuft.futures.queue_length",
            description="Number of futures in queue",
        )

        self.futures_created = meter.create_counter(
            "tuft.futures.created",
            description="Number of futures created",
        )

        self.futures_completed = meter.create_counter(
            "tuft.futures.completed",
            description="Number of futures completed",
        )

        self.futures_wait_time = meter.create_histogram(
            "tuft.futures.wait_time",
            description="Time waiting for future completion",
            unit="s",
        )

        self.futures_execution_time = meter.create_histogram(
            "tuft.futures.execution_time",
            description="Future execution time",
            unit="s",
        )

        # Redis metrics
        self.redis_operation_duration = meter.create_histogram(
            "tuft.redis.operation.duration",
            description="Redis operation duration",
            unit="s",
        )

    @classmethod
    def get_instance(cls) -> "TuftMetrics":
        """Get the singleton metrics instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance


def get_metrics() -> TuftMetrics:
    """Get the TuFT metrics instance."""
    return TuftMetrics.get_instance()


class ResourceMetricsCollector:
    """Collects system resource metrics periodically.

    Collects CPU, memory, disk, GPU, and network metrics.
    """

    _instance: "ResourceMetricsCollector | None" = None
    _lock = threading.Lock()

    def __init__(self, checkpoint_dir: str | None = None):
        self._checkpoint_dir = checkpoint_dir
        self._running = False
        self._thread: threading.Thread | None = None
        self._nvml_initialized = False
        self._gpu_available = self._check_and_init_gpu()
        self._setup_metrics()

    def _check_and_init_gpu(self) -> bool:
        """Check if GPU monitoring is available and initialize NVML once."""
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
            return True
        except pynvml.NVMLError as e:
            logger.debug("GPU monitoring not available: %s", e)
            return False

    def _shutdown_gpu(self) -> None:
        """Shutdown NVML if it was initialized."""
        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
            except pynvml.NVMLError as e:
                logger.warning("Failed to shutdown NVML: %s", e)

    def _setup_metrics(self) -> None:
        """Set up observable gauges for resource metrics."""
        meter = metrics.get_meter("tuft.resources")

        # CPU metrics
        meter.create_observable_gauge(
            "tuft.resource.cpu.utilization_percent",
            callbacks=[self._cpu_utilization_callback],
            description="CPU utilization percentage",
            unit="%",
        )

        # Memory metrics
        meter.create_observable_gauge(
            "tuft.resource.memory.used_bytes",
            callbacks=[self._memory_used_callback],
            description="Memory used in bytes",
            unit="bytes",
        )
        meter.create_observable_gauge(
            "tuft.resource.memory.total_bytes",
            callbacks=[self._memory_total_callback],
            description="Total memory in bytes",
            unit="bytes",
        )
        meter.create_observable_gauge(
            "tuft.resource.memory.utilization_percent",
            callbacks=[self._memory_utilization_callback],
            description="Memory utilization percentage",
            unit="%",
        )

        # GPU metrics (if available)
        if self._gpu_available:
            meter.create_observable_gauge(
                "tuft.resource.gpu.utilization_percent",
                callbacks=[self._gpu_utilization_callback],
                description="GPU utilization percentage",
                unit="%",
            )
            meter.create_observable_gauge(
                "tuft.resource.gpu.memory_used_bytes",
                callbacks=[self._gpu_memory_used_callback],
                description="GPU memory used in bytes",
                unit="bytes",
            )
            meter.create_observable_gauge(
                "tuft.resource.gpu.memory_total_bytes",
                callbacks=[self._gpu_memory_total_callback],
                description="GPU total memory in bytes",
                unit="bytes",
            )

        # Process metrics
        meter.create_observable_gauge(
            "tuft.resource.process.memory_used_bytes",
            callbacks=[self._process_memory_callback],
            description="Process memory usage in bytes",
            unit="bytes",
        )

    def _cpu_utilization_callback(self, options: CallbackOptions) -> Iterable[Observation]:
        """Callback for CPU utilization metric."""
        yield Observation(psutil.cpu_percent(interval=None))

    def _memory_used_callback(self, options: CallbackOptions) -> Iterable[Observation]:
        """Callback for memory used metric."""
        yield Observation(psutil.virtual_memory().used)

    def _memory_total_callback(self, options: CallbackOptions) -> Iterable[Observation]:
        """Callback for total memory metric."""
        yield Observation(psutil.virtual_memory().total)

    def _memory_utilization_callback(self, options: CallbackOptions) -> Iterable[Observation]:
        """Callback for memory utilization metric."""
        yield Observation(psutil.virtual_memory().percent)

    def _gpu_utilization_callback(self, options: CallbackOptions) -> Iterable[Observation]:
        """Callback for GPU utilization metric."""
        if not self._nvml_initialized:
            return
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                yield Observation(int(util.gpu), {"gpu_id": str(i)})
        except pynvml.NVMLError as e:
            logger.debug("Failed to get GPU utilization: %s", e)

    def _gpu_memory_used_callback(self, options: CallbackOptions) -> Iterable[Observation]:
        """Callback for GPU memory used metric."""
        if not self._nvml_initialized:
            return
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                yield Observation(int(mem_info.used), {"gpu_id": str(i)})
        except pynvml.NVMLError as e:
            logger.debug("Failed to get GPU memory used: %s", e)

    def _gpu_memory_total_callback(self, options: CallbackOptions) -> Iterable[Observation]:
        """Callback for GPU total memory metric."""
        if not self._nvml_initialized:
            return
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                yield Observation(int(mem_info.total), {"gpu_id": str(i)})
        except pynvml.NVMLError as e:
            logger.debug("Failed to get GPU memory total: %s", e)

    def _process_memory_callback(self, options: CallbackOptions) -> Iterable[Observation]:
        """Callback for process memory metric."""
        process = psutil.Process()
        yield Observation(process.memory_info().rss)

    @classmethod
    def start(cls, checkpoint_dir: str | None = None) -> "ResourceMetricsCollector":
        """Start the resource metrics collector."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(checkpoint_dir)
        return cls._instance

    @classmethod
    def shutdown(cls) -> None:
        """Shutdown the resource metrics collector."""
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance._shutdown_gpu()
                    cls._instance = None
