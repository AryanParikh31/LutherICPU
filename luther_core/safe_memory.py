"""lutherICPU Safe Memory Profiler and Execution Guardian.

Ensures zero system crashes and bounds memory usage to safe CPU thresholds.
"""
import gc
import os
import psutil
import logging
from typing import Optional, Callable, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lutherICPU.MemoryGuardian")


class MemoryGuardian:
    """Monitors system and process memory to prevent OOM on 8GB RAM machines."""

    def __init__(self, max_process_ram_gb: float = 3.5, max_system_ram_pct: float = 85.0, max_ram_gb: float = None):
        limit = max_ram_gb if max_ram_gb is not None else max_process_ram_gb
        self.max_process_ram_bytes = int(limit * 1024 * 1024 * 1024)
        self.max_system_ram_pct = max_system_ram_pct
        self.process = psutil.Process(os.getpid())

    @property
    def process_memory_bytes(self) -> int:
        """Current process memory (RSS) in bytes."""
        return self.process.memory_info().rss

    @property
    def process_memory_gb(self) -> float:
        """Current process memory in Gigabytes."""
        return self.process_memory_bytes / (1024 ** 3)

    @property
    def system_memory_percent(self) -> float:
        """Total system RAM usage percentage."""
        return psutil.virtual_memory().percent

    @property
    def available_system_ram_gb(self) -> float:
        """Available system RAM in Gigabytes."""
        return psutil.virtual_memory().available / (1024 ** 3)

    def check_safety(self, context: str = "") -> None:
        """Checks if memory limits are exceeded; performs aggressive cleanup if near limit."""
        proc_mem = self.process_memory_gb
        sys_pct = self.system_memory_percent

        if proc_mem > (self.max_process_ram_bytes / (1024 ** 3)) * 0.85 or sys_pct > self.max_system_ram_pct:
            logger.warning(
                f"[Memory Alert] {context} - Proc RAM: {proc_mem:.2f} GB, Sys RAM: {sys_pct:.1f}%. Triggering GC..."
            )
            gc.collect()

        if proc_mem > (self.max_process_ram_bytes / (1024 ** 3)):
            gc.collect()
            if self.process_memory_gb > (self.max_process_ram_bytes / (1024 ** 3)):
                raise MemoryError(
                    f"lutherICPU Safety Guard: Memory exceeded safe ceiling of "
                    f"{self.max_process_ram_bytes / (1024 ** 3):.2f} GB (Current: {proc_mem:.2f} GB). "
                    f"Execution safely halted to prevent OS crash."
                )

    def checkpoint(self, context: str = "") -> None:
        """Alias for check_safety."""
        self.check_safety(context)

    def get_usage_mb(self) -> dict:
        """Returns current process and system memory usage in MB."""
        proc_mb = self.process_memory_bytes / (1024 * 1024)
        sys_mb = psutil.virtual_memory().used / (1024 * 1024)
        return {
            "process_mb": proc_mb,
            "system_mb": sys_mb,
            "system_pct": self.system_memory_percent
        }


# Global singleton instance for easy import across modules
global_guardian = MemoryGuardian(max_process_ram_gb=3.5, max_system_ram_pct=85.0)
