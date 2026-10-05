"""
Quiet NVIDIA GPU while the agent works: fixed fan speed, lower power limit, and a duty cycle that follows the
temperature. Needs root (NVML refuses fan and power changes otherwise, also inside a container).

Why: the stock fan curve of many cards is on/off (0% when idle, 70%+ under any real load), so "a bit of load"
is impossible without taking the fan over. With the fan held at its lowest stable speed, the agent pauses
between batches as much as needed to stay at GPU_TARGET_TEMP.

    GPU_FAN_SPEED     fan speed in percent while working (lowest stable speed of the card, e.g. 50)
    GPU_POWER_LIMIT   power limit in watts while working (more vectors per watt, smaller heat spikes)
    GPU_TARGET_TEMP   keep the GPU near this temperature by changing the duty cycle (e.g. 70)

The card gets its defaults back (automatic fan, default power limit) whenever the agent does not work: on
stop and during a pause from the indexer. If the temperature still reaches GPU_TARGET_TEMP + 12 (something
else is heating the card), the fan goes back to automatic at once.
"""

import math
import os
import threading
import time
from typing import Callable, Optional

ADJUST_EVERY = 20       # seconds between duty cycle adjustments
MIN_DUTY = 0.03
EMERGENCY_MARGIN = 12   # degrees above target: give the fan back to the driver


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


class GpuCooling:
    def __init__(self, index: int = 0, fan_speed: Optional[int] = None, power_limit: Optional[int] = None,
                 target_temp: Optional[int] = None, duty: float = 1.0):
        self.index = index
        self.fan_speed = fan_speed
        self.power_limit = power_limit
        self.target_temp = target_temp
        self.duty = duty
        # the embedder's DutyCycle; a getter because the embedder is rebuilt after a pause
        self.duty_cycle: Callable[[], object] = lambda: None
        self.nvml = None
        self.handle = None
        self.default_power_limit = None
        self.fan_forced = False
        self.active = threading.Event()
        self.stop_event = threading.Event()

    @classmethod
    def from_env(cls, device: str) -> Optional["GpuCooling"]:
        fan = os.getenv("GPU_FAN_SPEED")
        power = os.getenv("GPU_POWER_LIMIT")
        target = os.getenv("GPU_TARGET_TEMP")
        if not (fan or power or target) or not device.startswith("cuda"):
            return None
        index = int(device.split(":")[1]) if ":" in device else 0
        return cls(index, int(fan) if fan else None, int(power) if power else None,
                   int(target) if target else None, float(os.getenv("DUTY_CYCLE", "1")))

    def start(self) -> None:
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.index)
        self.default_power_limit = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self.handle)
        self.resume()
        if self.target_temp:
            threading.Thread(target=self._loop, name="gpu-cooling", daemon=True).start()
            log(f"GPU target temperature {self.target_temp}C")

    def resume(self) -> None:
        """Agent works: take over the fan and the power limit."""
        if self.power_limit:
            self.nvml.nvmlDeviceSetPowerManagementLimit(self.handle, self.power_limit * 1000)
        if self.fan_speed:
            for fan in range(self.nvml.nvmlDeviceGetNumFans(self.handle)):
                self.nvml.nvmlDeviceSetFanSpeed_v2(self.handle, fan, self.fan_speed)
            self.fan_forced = True
        self._apply_duty()
        self.active.set()
        log(f"GPU fan {self.fan_speed or 'auto'}%, power limit {self.power_limit or 'default'} W")

    def pause(self) -> None:
        """Agent does not work: the card gets its defaults back (the fan stops when it cools down)."""
        self.active.clear()
        self._restore()
        log("GPU fan and power limit back to defaults")

    def _apply_duty(self) -> None:
        duty_cycle = self.duty_cycle()
        if duty_cycle is not None:
            duty_cycle.default = duty_cycle.duty = self.duty

    def _loop(self) -> None:
        last_report = 0.0
        while not self.stop_event.wait(ADJUST_EVERY):
            if not self.active.is_set():
                continue
            try:
                temp = self.nvml.nvmlDeviceGetTemperature(self.handle, self.nvml.NVML_TEMPERATURE_GPU)
            except Exception as e:
                log(f"GPU temperature not available ({e})")
                continue
            if self.fan_forced and temp >= self.target_temp + EMERGENCY_MARGIN:
                self._restore_fan()
                log(f"GPU at {temp}C: fan back to automatic")
            self.duty = min(1.0, max(MIN_DUTY, self.duty * math.exp(0.04 * (self.target_temp - temp))))
            self._apply_duty()
            if time.time() - last_report > 600:
                last_report = time.time()
                log(f"GPU {temp}C, duty cycle {self.duty:.2f}")

    def _restore_fan(self) -> None:
        if not self.fan_forced:
            return
        for fan in range(self.nvml.nvmlDeviceGetNumFans(self.handle)):
            self.nvml.nvmlDeviceSetDefaultFanSpeed_v2(self.handle, fan)
        self.fan_forced = False

    def _restore(self) -> None:
        self._restore_fan()
        if self.power_limit and self.default_power_limit:
            self.nvml.nvmlDeviceSetPowerManagementLimit(self.handle, self.default_power_limit)

    def stop(self) -> None:
        self.stop_event.set()
        self.active.clear()
        if self.nvml is None:
            return
        try:
            self._restore()
            log("GPU fan and power limit back to defaults")
        finally:
            self.nvml.nvmlShutdown()
