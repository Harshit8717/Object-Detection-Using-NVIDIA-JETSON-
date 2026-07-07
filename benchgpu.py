import time
import psutil
import torch
import threading
from ultralytics import YOLO

from config import *

from core.system_info import *
from core.model_info import *
from core.gpu_info import *
from core.jetson_info import *
from core.statistics_utils import *

from utils.exporter import *
from utils.graphs import *

from core.image_info import *

cpu_samples = []
ram_samples = []
frequency_samples = []

# Determine the target device (defaults to 'cuda' if available, otherwise falls back gracefully)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def monitor_resources(stop_event):
    process = psutil.Process()
    while not stop_event.is_set():
        cpu_samples.append(psutil.cpu_percent(interval=None))
        ram_samples.append(process.memory_info().rss / (1024**2))
        freq = psutil.cpu_freq()
        if freq:
            frequency_samples.append(freq.current)
        time.sleep(0.05)


def load_model():
    start = time.perf_counter()
    model = YOLO(MODEL_PATH)
    # Explicitly move the model to the GPU memory if available
    model.to(DEVICE)
    end = time.perf_counter()
    return model, (end - start) * 1000


def warmup(model):
    print(f"\nRunning Warm-up on {DEVICE.upper()}...\n")
    warmup_times = []
    for _ in range(WARMUP_RUNS):
        start = time.perf_counter()
        # Added device=DEVICE to force GPU usage
        model.predict(IMAGE_PATH, save=False, verbose=False, device=DEVICE)
        end = time.perf_counter()
        warmup_times.append((end - start) * 1000)
    print("Warm-up Complete.")
    return warmup_times


def benchmark(model):
    latency = []
    inference_times = []
    preprocess_times = []
    postprocess_times = []

    cpu_usage = []
    ram_usage = []

    detections = []

    all_confidences = []
    class_counter = {}

    process = psutil.Process()
    stop_event = threading.Event()

    monitor = threading.Thread(target=monitor_resources, args=(stop_event,))
    monitor.start()

    print(f"\nRunning Benchmark ({NUM_RUNS} Runs) on {DEVICE.upper()}...\n")

    for run in range(NUM_RUNS):
        cpu_usage.append(psutil.cpu_percent(interval=None))
        ram_usage.append(process.memory_info().rss / (1024**2))

        start = time.perf_counter()
        # Added device=DEVICE to force GPU usage
        results = model.predict(
            IMAGE_PATH, save=SAVE_IMAGES, verbose=VERBOSE, device=DEVICE
        )
        end = time.perf_counter()

        latency.append((end - start) * 1000)

        speed = results[0].speed
        preprocess_times.append(speed["preprocess"])
        inference_times.append(speed["inference"])
        postprocess_times.append(speed["postprocess"])

        detections.append(len(results[0].boxes))

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                conf = float(box.conf.item())
                cls = int(box.cls.item())

                all_confidences.append(conf)
                class_name = model.names[cls]
                class_counter[class_name] = (
                    class_counter.get(class_name, 0) + 1
                )

                print(f"Run {run+1}/{NUM_RUNS}", end="\r")

    print()
    stop_event.set()
    monitor.join()
    return {
        "Latency": latency,
        "Preprocess": preprocess_times,
        "Inference": inference_times,
        "Postprocess": postprocess_times,
        "CPU": cpu_samples,
        "RAM": ram_samples,
        "CPU Frequency": frequency_samples,
        "Detections": detections,
        "Confidences": all_confidences,
        "Classes": class_counter,
    }


def print_summary(report):
    print("\n")
    print("=" * 90)
    print("EXECUTIVE PERFORMANCE SUMMARY")
    print("=" * 90)

    latency = report["Latency Statistics"]
    inference = report["Inference Statistics"]
    preprocess = report["Preprocess Statistics"]
    postprocess = report["Postprocess Statistics"]
    cpu = report["CPU Statistics"]
    ram = report["RAM Statistics"]
    model = report["Model Information"]
    image = report["Input Image"]
    confidence = report["Confidence Statistics"]

    rows = [
        ("Model", model["Model Name"]),
        ("Parameters", model["Readable Parameters"]),
        ("Model Size", f'{model["Model Size (MB)"]} MB'),
        ("Classes", model["Classes"]),
        ("Input Resolution", image["Resolution"]),
        ("Image Format", image["Format"]),
        ("Average Latency", f'{latency["Mean"]:.2f} ms'),
        ("Median Latency", f'{latency["Median"]:.2f} ms'),
        ("P95 Latency", f'{latency["P95"]:.2f} ms'),
        ("Average FPS", f'{latency["Average FPS"]:.2f}'),
        ("Average Inference", f'{inference["Mean"]:.2f} ms'),
        ("Average Preprocess", f'{preprocess["Mean"]:.2f} ms'),
        ("Average Postprocess", f'{postprocess["Mean"]:.2f} ms'),
        ("Average CPU Usage", f'{cpu["Mean"]:.2f} %'),
        ("Peak CPU Usage", f'{cpu["Maximum"]:.2f} %'),
        ("Average RAM", f'{ram["Mean"]:.2f} MB'),
        ("Peak RAM", f'{ram["Maximum"]:.2f} MB'),
        ("Average Confidence", f'{confidence["Mean"]:.3f}'),
        ("Maximum Confidence", f'{confidence["Maximum"]:.3f}'),
        ("Minimum Confidence", f'{confidence["Minimum"]:.3f}'),
    ]

    print(f'{"Metric":<35} {"Value":>30}')
    print("-" * 70)

    for metric, value in rows:
        print(f"{metric:<35} {str(value):>30}")

    print("=" * 90)


if __name__ == "__main__":
    print("=" * 90)
    print("EDGE AI OBJECT DETECTION PERFORMANCE BENCHMARK FRAMEWORK")
    print("=" * 90)

    model, load_time = load_model()

    system_info = get_system_info()
    gpu_info = get_gpu_info()
    model_info = get_model_info(model, MODEL_PATH)
    image_info = get_image_info(IMAGE_PATH)

    print_system_info(system_info)
    print_gpu_info(gpu_info)
    print_model_info(model_info)
    print_image_info(image_info)

    print(f"\nModel Load Time : {load_time:.2f} ms")

    warmup_times = warmup(model)
    benchmark_results = benchmark(model)

    latency_stats = calculate_statistics(benchmark_results["Latency"])
    preprocess_stats = calculate_statistics(benchmark_results["Preprocess"])
    inference_stats = calculate_statistics(benchmark_results["Inference"])
    postprocess_stats = calculate_statistics(benchmark_results["Postprocess"])

    cpu_stats = calculate_statistics(benchmark_results["CPU"])
    ram_stats = calculate_statistics(benchmark_results["RAM"])
    frequency_stats = calculate_statistics(benchmark_results["CPU Frequency"])

    detection_stats = calculate_statistics(benchmark_results["Detections"])
    confidence_stats = calculate_statistics(benchmark_results["Confidences"])

    warmup_stats = calculate_statistics(warmup_times)

    benchmark_report = {
        "System Information": system_info,
        "GPU Information": gpu_info,
        "Model Information": model_info,
        "Input Image": image_info,
        "Benchmark Configuration": {
            "Model Path": MODEL_PATH,
            "Image Path": IMAGE_PATH,
            "Runs": NUM_RUNS,
            "Warmup Runs": WARMUP_RUNS,
        },
        "Model Load Time (ms)": round(load_time, 4),
        "Warmup Statistics": warmup_stats,
        "Latency Statistics": latency_stats,
        "Preprocess Statistics": preprocess_stats,
        "Inference Statistics": inference_stats,
        "Postprocess Statistics": postprocess_stats,
        "CPU Statistics": cpu_stats,
        "RAM Statistics": ram_stats,
        "CPU Frequency Statistics": frequency_stats,
        "Detection Statistics": detection_stats,
        "Confidence Statistics": confidence_stats,
        "Detected Classes": benchmark_results["Classes"],
    }

    print("\n")
    print("=" * 90)
    print("LATENCY")
    print("=" * 90)
    print_statistics(latency_stats)

    print("\n")
    print("=" * 90)
    print("PREPROCESS")
    print("=" * 90)
    print_statistics(preprocess_stats)

    print("\n")
    print("=" * 90)
    print("INFERENCE")
    print("=" * 90)
    print_statistics(inference_stats)

    print("\n")
    print("=" * 90)
    print("POSTPROCESS")
    print("=" * 90)
    print_statistics(postprocess_stats)

    print("\n")
    print("=" * 90)
    print("CPU")
    print("=" * 90)
    print_statistics(cpu_stats)

    print("\n")
    print("=" * 90)
    print("RAM")
    print("=" * 90)
    print_statistics(ram_stats)

    print("\n")
    print("=" * 90)
    print("CPU FREQUENCY")
    print("=" * 90)
    print_statistics(frequency_stats)

    print("\n")
    print("=" * 90)
    print("DETECTIONS")
    print("=" * 90)
    print_statistics(detection_stats)

    print("\n")
    print("=" * 90)
    print("DETECTION CONFIDENCE")
    print("=" * 90)
    print_statistics(confidence_stats)

    print("\n")
    print("=" * 90)
    print("CLASS DISTRIBUTION")
    print("=" * 90)
    for cls, count in benchmark_results["Classes"].items():
        print(f"{cls:<25}: {count}")

    exported_files = export_all(benchmark_report, RESULTS_DIR)
    print_export_summary(exported_files)

    print("\n")
    print("=" * 90)
    print("BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 90)