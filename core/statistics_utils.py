import statistics
import numpy as np


def calculate_statistics(values):

    if len(values) == 0:
        return {}

    stats = {}

    stats["Count"] = len(values)

    stats["Mean"] = round(statistics.mean(values), 4)

    stats["Median"] = round(statistics.median(values), 4)

    stats["Minimum"] = round(min(values), 4)

    stats["Maximum"] = round(max(values), 4)

    stats["Range"] = round(max(values) - min(values), 4)

    if len(values) > 1:

        stats["Standard Deviation"] = round(statistics.stdev(values), 4)

        stats["Variance"] = round(statistics.variance(values), 4)

    else:

        stats["Standard Deviation"] = 0

        stats["Variance"] = 0

    stats["P10"] = round(np.percentile(values, 10), 4)

    stats["P25"] = round(np.percentile(values, 25), 4)

    stats["P50"] = round(np.percentile(values, 50), 4)

    stats["P75"] = round(np.percentile(values, 75), 4)

    stats["P90"] = round(np.percentile(values, 90), 4)

    stats["P95"] = round(np.percentile(values, 95), 4)

    stats["P99"] = round(np.percentile(values, 99), 4)

    stats["Coefficient of Variation (%)"] = (
        round((statistics.stdev(values) / statistics.mean(values)) * 100, 4)
        if len(values) > 1
        else 0
    )

    stats["Average FPS"] = round(1000 / statistics.mean(values), 4)

    stats["Throughput (Images/sec)"] = round(1000 / statistics.mean(values), 4)

    return stats


def print_statistics(stats):

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    for key, value in stats.items():

        print(f"{key:<40}: {value}")
