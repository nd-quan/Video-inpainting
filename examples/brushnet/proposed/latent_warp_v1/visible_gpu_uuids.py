"""Identify CUDA-visible devices, including scheduler remapping."""

import argparse
import ctypes
import ctypes.util
from pathlib import Path
import subprocess
import sys


def pci_key(value):
    domain, bus, device_function = value.strip().split(":")
    device, function = device_function.split(".")
    return tuple(int(part, 16) for part in (domain, bus, device, function))


def load_runtime():
    import torch

    if not torch.version.cuda:
        raise RuntimeError("This PyTorch installation has no CUDA support")
    major = torch.version.cuda.split(".")[0]
    package_root = Path(torch.__file__).resolve().parent
    candidates = []
    for directory in (package_root / "lib", package_root.parent / "nvidia/cuda_runtime/lib",
                      Path(sys.prefix) / "lib"):
        candidates.extend(sorted(directory.glob("libcudart.so." + major + "*")))
    discovered = ctypes.util.find_library("cudart")
    if discovered:
        candidates.append(discovered)
    for candidate in candidates:
        try:
            return ctypes.CDLL(str(candidate))
        except OSError:
            continue
    raise RuntimeError("Cannot locate the CUDA runtime used by PyTorch")


def visible_pci_ids(expected_count):
    if expected_count < 1:
        raise ValueError("At least one CUDA-visible GPU is required")
    runtime = load_runtime()
    runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaGetDeviceCount.restype = ctypes.c_int
    runtime.cudaDeviceGetPCIBusId.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
    runtime.cudaDeviceGetPCIBusId.restype = ctypes.c_int
    count = ctypes.c_int()
    status = runtime.cudaGetDeviceCount(ctypes.byref(count))
    if status or count.value != expected_count:
        raise RuntimeError("Expected {} CUDA-visible GPUs, found {} (CUDA status {})".format(
            expected_count, count.value, status))
    result = []
    for index in range(count.value):
        bus = ctypes.create_string_buffer(32)
        status = runtime.cudaDeviceGetPCIBusId(bus, len(bus), index)
        if status:
            raise RuntimeError("Cannot identify CUDA:{} (CUDA status {})".format(index, status))
        result.append(bus.value.decode("ascii"))
    if len({pci_key(bus) for bus in result}) != len(result):
        raise RuntimeError("Visible CUDA devices do not map to distinct physical GPUs")
    return result


def visible_uuids(expected_count):
    buses = visible_pci_ids(expected_count)
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=pci.bus_id,uuid", "--format=csv,noheader"], text=True)
    by_bus = {}
    for row in inventory.splitlines():
        bus, uuid = row.split(",", 1)
        by_bus[pci_key(bus)] = uuid.strip()
    result = []
    for index, bus in enumerate(buses):
        key = pci_key(bus)
        if key not in by_bus:
            raise RuntimeError("CUDA:{} PCI address missing from nvidia-smi".format(index))
        result.append(by_bus[key])
    if len(set(result)) != len(result):
        raise RuntimeError("Visible CUDA devices do not map to distinct physical GPUs")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected_count", type=int)
    parser.add_argument("--pci-only", action="store_true",
                        help="Validate CUDA-visible devices without calling nvidia-smi")
    args = parser.parse_args()
    try:
        identify = visible_pci_ids if args.pci_only else visible_uuids
        print("\n".join(identify(args.expected_count)))
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit("Cannot verify assigned GPUs: " + str(error))
