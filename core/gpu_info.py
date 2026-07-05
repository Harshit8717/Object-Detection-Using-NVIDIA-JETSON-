import torch

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


def get_gpu_info():

    info = {}

    info["CUDA Available"] = torch.cuda.is_available()

    if not torch.cuda.is_available():

        info["GPU Detected"] = False
        return info

    info["GPU Detected"] = True

    info["CUDA Version"] = torch.version.cuda

    info["GPU Count"] = torch.cuda.device_count()

    devices = []

    for i in range(torch.cuda.device_count()):

        device = {}

        props = torch.cuda.get_device_properties(i)

        device["Index"] = i

        device["Name"] = props.name

        device["Total Memory (GB)"] = round(props.total_memory / (1024**3), 2)

        device["Compute Capability"] = f"{props.major}.{props.minor}"

        device["Multi Processor Count"] = props.multi_processor_count

        device["Current Allocated Memory (MB)"] = round(
            torch.cuda.memory_allocated(i) / (1024**2), 2
        )

        device["Reserved Memory (MB)"] = round(
            torch.cuda.memory_reserved(i) / (1024**2), 2
        )

        devices.append(device)

    info["Devices"] = devices

    if NVML_AVAILABLE:

        try:

            pynvml.nvmlInit()

            nvml_devices = []

            for i in range(pynvml.nvmlDeviceGetCount()):

                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                util = pynvml.nvmlDeviceGetUtilizationRates(handle)

                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                power = None

                temp = None

                fan = None

                clock = None

                driver = pynvml.nvmlSystemGetDriverVersion().decode()

                try:
                    power = round(
                        pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                        2,
                    )
                except:
                    pass

                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle,
                        pynvml.NVML_TEMPERATURE_GPU,
                    )
                except:
                    pass

                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                except:
                    pass

                try:
                    clock = pynvml.nvmlDeviceGetClockInfo(
                        handle,
                        pynvml.NVML_CLOCK_GRAPHICS,
                    )
                except:
                    pass

                nvml_devices.append(
                    {
                        "Driver Version": driver,
                        "GPU Utilization (%)": util.gpu,
                        "Memory Utilization (%)": util.memory,
                        "Memory Used (MB)": round(mem.used / (1024**2), 2),
                        "Memory Free (MB)": round(mem.free / (1024**2), 2),
                        "Memory Total (MB)": round(mem.total / (1024**2), 2),
                        "Temperature (C)": temp,
                        "Power Draw (W)": power,
                        "Fan Speed (%)": fan,
                        "Graphics Clock (MHz)": clock,
                    }
                )

            info["NVML"] = nvml_devices

            pynvml.nvmlShutdown()

        except Exception as e:

            info["NVML Error"] = str(e)

    else:

        info["NVML"] = "Not Installed"

    return info


def print_gpu_info(info):

    print("\n" + "=" * 80)
    print("GPU INFORMATION")
    print("=" * 80)

    for key, value in info.items():

        if isinstance(value, list):

            print(f"\n{key}")

            for device in value:

                print("-" * 60)

                for k, v in device.items():

                    print(f"{k:<35}: {v}")

        else:

            print(f"{key:<35}: {value}")
