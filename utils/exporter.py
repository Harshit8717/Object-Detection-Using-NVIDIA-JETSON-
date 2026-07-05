import csv
import json
import os
from datetime import datetime


def _flatten(data, parent_key="", sep="."):

    items = {}

    for key, value in data.items():

        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):

            items.update(_flatten(value, new_key, sep))

        elif isinstance(value, list):

            items[new_key] = str(value)

        else:

            items[new_key] = value

    return items


def export_json(data, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(
        output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(filename, "w", encoding="utf-8") as f:

        json.dump(data, f, indent=4)

    return filename


def export_csv(data, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(
        output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    flat = _flatten(data)

    with open(filename, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(["Parameter", "Value"])

        for k, v in flat.items():

            writer.writerow([k, v])

    return filename


def export_txt(data, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(
        output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

    flat = _flatten(data)

    with open(filename, "w", encoding="utf-8") as f:

        f.write("=" * 80 + "\n")
        f.write("YOLO BENCHMARK REPORT\n")
        f.write("=" * 80 + "\n\n")

        for k, v in flat.items():

            f.write(f"{k:<45}: {v}\n")

    return filename


def export_all(data, output_dir):

    files = {}

    files["JSON"] = export_json(data, output_dir)

    files["CSV"] = export_csv(data, output_dir)

    files["TXT"] = export_txt(data, output_dir)

    return files


def print_export_summary(files):

    print("\n" + "=" * 80)
    print("EXPORT SUMMARY")
    print("=" * 80)

    for name, path in files.items():

        print(f"{name:<15}: {path}")
