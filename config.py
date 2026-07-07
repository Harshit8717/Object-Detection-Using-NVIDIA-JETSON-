"""
=============================================================
YOLO Benchmark Configuration
=============================================================
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

IMAGE_PATH = os.path.join(BASE_DIR, "images.jpg")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

GRAPHS_DIR = os.path.join(BASE_DIR, "graphs")

LOGS_DIR = os.path.join(BASE_DIR, "logs")


os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


NUM_RUNS = 100

WARMUP_RUNS = 5

IMAGE_SIZE = 640

BATCH_SIZE = 1

CONFIDENCE_THRESHOLD = 0.25

IOU_THRESHOLD = 0.45

DEVICE = "cuda"

SAVE_IMAGES = True

VERBOSE = False

EXPORT_TXT = True

EXPORT_JSON = True

EXPORT_CSV = True

EXPORT_GRAPHS = True


GRAPH_DPI = 300

HISTOGRAM_BINS = 20

REPORT_TITLE = "YOLOv8 Performance Benchmark"

AUTHOR = "Abhradeep Kayal"

ORGANIZATION = "IIEST Shibpur"

VERSION = "1.0"
