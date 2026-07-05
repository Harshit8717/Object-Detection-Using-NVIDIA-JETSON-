import os
import platform
import socket
from datetime import datetime

import cpuinfo
import cv2
import psutil
import torch
import ultralytics


def bytes_to_gb(value):
    return round(value / (1024**3), 2)


def get_system_info():

    info = {}

    cpu = cpuinfo.get_cpu_info()

    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage(os.getcwd())

    process = psutil.Process(os.getpid())

    info["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    info["Hostname"] = socket.gethostname()

    info["Operating System"] = platform.system()

    info["OS Release"] = platform.release()

    info["OS Version"] = platform.version()

    info["Architecture"] = platform.machine()

    info["Python Version"] = platform.python_version()

    info["OpenCV Version"] = cv2.__version__

    info["PyTorch Version"] = torch.__version__

    info["Ultralytics Version"] = ultralytics.__version__

    info["CUDA Available"] = torch.cuda.is_available()

    info["CUDA Version"] = (
        torch.version.cuda if torch.cuda.is_available() else "Not Available"
    )

    info["cuDNN Enabled"] = torch.backends.cudnn.enabled

    try:
        info["CPU Name"] = cpu["brand_raw"]
    except:
        info["CPU Name"] = platform.processor()

    info["CPU Architecture"] = cpu.get("arch", "Unknown")

    info["Bits"] = cpu.get("bits", "Unknown")

    info["Physical Cores"] = psutil.cpu_count(logical=False)

    info["Logical Threads"] = psutil.cpu_count(logical=True)

    freq = psutil.cpu_freq()

    if freq:

        info["CPU Current Frequency (MHz)"] = round(freq.current, 2)

        info["CPU Min Frequency (MHz)"] = round(freq.min, 2)

        info["CPU Max Frequency (MHz)"] = round(freq.max, 2)

    info["CPU Usage (%)"] = psutil.cpu_percent(interval=1)

    info["Per Core Usage (%)"] = psutil.cpu_percent(interval=1, percpu=True)

    info["Total RAM (GB)"] = bytes_to_gb(vm.total)

    info["Available RAM (GB)"] = bytes_to_gb(vm.available)

    info["Used RAM (GB)"] = bytes_to_gb(vm.used)

    info["RAM Usage (%)"] = vm.percent

    info["Swap Total (GB)"] = bytes_to_gb(swap.total)

    info["Swap Used (GB)"] = bytes_to_gb(swap.used)

    info["Swap Usage (%)"] = swap.percent

    info["Disk Total (GB)"] = bytes_to_gb(disk.total)

    info["Disk Used (GB)"] = bytes_to_gb(disk.used)

    info["Disk Free (GB)"] = bytes_to_gb(disk.free)

    info["Disk Usage (%)"] = disk.percent

    mem = process.memory_info()

    info["Process RAM (MB)"] = round(mem.rss / (1024**2), 2)

    info["Virtual Memory (MB)"] = round(mem.vms / (1024**2), 2)

    info["Process Threads"] = process.num_threads()

    info["Boot Time"] = datetime.fromtimestamp(psutil.boot_time()).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    return info


def print_system_info(info):

    print("\n" + "=" * 90)

    print("SYSTEM INFORMATION")

    print("=" * 90)

    for k, v in info.items():

        print(f"{k:<35}: {v}")
