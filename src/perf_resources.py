"""Read-only Windows/NVML resource sampling for the local throughput experiment."""

import ctypes as c
import os
import threading
from time import perf_counter

import numpy as np


class MemoryStatus(c.Structure):
    _fields_ = [("length", c.c_uint32), ("load", c.c_uint32)] + [
        (name, c.c_uint64) for name in ("total", "available", "total_page", "available_page",
            "total_virtual", "available_virtual", "available_extended")]


def physical_memory():
    status = MemoryStatus(); status.length = c.sizeof(status)
    if not c.windll.kernel32.GlobalMemoryStatusEx(c.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {"total_RAM_bytes": int(status.total), "available_RAM_bytes": int(status.available),
            "memory_load_percent": int(status.load), "logical_CPUs": os.cpu_count()}


def process_rss(pid):
    class Counters(c.Structure):
        _fields_ = [("cb", c.c_uint32), ("PageFaultCount", c.c_uint32)] + [
            (name, c.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
             "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    kernel, psapi = c.WinDLL("kernel32"), c.WinDLL("psapi")
    kernel.OpenProcess.restype = c.c_void_p
    kernel.CloseHandle.argtypes = (c.c_void_p,)
    handle = kernel.OpenProcess(0x1010, False, int(pid))
    if not handle:
        return 0
    psapi.GetProcessMemoryInfo.argtypes = (c.c_void_p, c.c_void_p, c.c_uint32)
    info = Counters(); info.cb = c.sizeof(info)
    try:
        if not psapi.GetProcessMemoryInfo(handle, c.byref(info), info.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(info.WorkingSetSize)
    finally:
        kernel.CloseHandle(handle)


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(a), "mean": float(np.mean(a)), "median": float(np.median(a)),
            "P90": float(np.quantile(a, .9)), "max": float(np.max(a))} if len(a) else None


class ResourceSampler:
    def __init__(self, interval=.25):
        self.interval = interval
        self.rows, self.children, self.errors = [], [], []
        self.stop_event = threading.Event()
        self.nvml = c.WinDLL("nvml.dll")
        if self.nvml.nvmlInit_v2() != 0:
            raise OSError("NVML initialization failed")
        self.gpu = c.c_void_p()
        if self.nvml.nvmlDeviceGetHandleByIndex_v2(0, c.byref(self.gpu)) != 0:
            raise OSError("NVML GPU handle failed")
        self.pdh = c.WinDLL("pdh.dll")
        self.query = c.c_void_p()
        self.pdh.PdhCloseQuery.argtypes = (c.c_void_p,)
        if self.pdh.PdhOpenQueryW(None, 0, c.byref(self.query)) != 0:
            raise OSError("PDH query failed")
        self.counters = {}
        for name, path in (("page_reads_per_second", r"\Memory\Page Reads/sec"),
                ("pages_input_per_second", r"\Memory\Pages Input/sec"),
                ("pagefile_usage_percent", r"\Paging File(_Total)\% Usage")):
            counter = c.c_void_p()
            if self.pdh.PdhAddEnglishCounterW(self.query, path, 0, c.byref(counter)) != 0:
                raise OSError(f"PDH counter missing: {name}")
            self.counters[name] = counter
        self.pdh.PdhCollectQueryData(self.query)
        self.previous_times = self._cpu_times()
        self.started = perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def _cpu_times():
        idle, kernel, user = c.c_uint64(), c.c_uint64(), c.c_uint64()
        if not c.windll.kernel32.GetSystemTimes(c.byref(idle), c.byref(kernel), c.byref(user)):
            raise OSError("GetSystemTimes failed")
        return np.array([idle.value, kernel.value, user.value], dtype=np.float64)

    def sample(self):
        class Util(c.Structure):
            _fields_ = [("gpu", c.c_uint32), ("memory", c.c_uint32)]
        class Value(c.Structure):
            _fields_ = [("status", c.c_uint32), ("value", c.c_double)]
        util = Util()
        if self.nvml.nvmlDeviceGetUtilizationRates(self.gpu, c.byref(util)) != 0:
            raise OSError("NVML utilization read failed")
        current = self._cpu_times(); delta = current-self.previous_times; self.previous_times = current
        total = delta[1]+delta[2]
        memory = physical_memory()
        row = {"elapsed_seconds": perf_counter()-self.started, "gpu_utilization_percent": int(util.gpu),
            "cpu_utilization_percent": float(100*(1-delta[0]/total)) if total else 0.0,
            "available_RAM_bytes": memory["available_RAM_bytes"], "parent_RSS_bytes": process_rss(os.getpid()),
            "process_tree_RSS_bytes": process_rss(os.getpid())+sum(process_rss(pid) for pid in self.children)}
        self.pdh.PdhCollectQueryData(self.query)
        for name, counter in self.counters.items():
            value = Value()
            code = self.pdh.PdhGetFormattedCounterValue(counter, 0x200, None, c.byref(value))
            row[name] = float(value.value) if code == 0 and value.status in (0, 1) else None
        self.rows.append(row)

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self.errors.append(str(exc)); break

    def finish(self):
        self.stop_event.set(); self.thread.join(timeout=3)
        self.sample()
        self.nvml.nvmlShutdown(); self.pdh.PdhCloseQuery(self.query)
        if self.errors:
            raise RuntimeError(f"resource sampling failed: {self.errors}")
        pagefile = [r["pagefile_usage_percent"] for r in self.rows if r["pagefile_usage_percent"] is not None]
        reads = [r["page_reads_per_second"] for r in self.rows if r["page_reads_per_second"] is not None]
        growth = max(pagefile)-pagefile[0] if pagefile else None
        minimum = min(r["available_RAM_bytes"] for r in self.rows)
        paging = bool(growth is not None and growth > .25 and reads and np.quantile(reads, .9) > 1000)
        return {"gpu_utilization": distribution([r["gpu_utilization_percent"] for r in self.rows]),
            "cpu_utilization": distribution([r["cpu_utilization_percent"] for r in self.rows]),
            "available_RAM_min_bytes": minimum, "peak_parent_RSS_bytes": max(r["parent_RSS_bytes"] for r in self.rows),
            "peak_process_tree_RSS_bytes": max(r["process_tree_RSS_bytes"] for r in self.rows),
            "page_reads": distribution(reads), "pagefile_usage_growth_percentage_points": growth,
            "paging_detected": paging, "memory_safe": minimum >= 1.5*2**30 and not paging,
            "sampling_interval_seconds": self.interval, "sampling_scope": "system CPU/GPU, parent plus DataLoader workers RSS"}
