import os
import cv2


def bytes_to_mb(size):

    return round(size / (1024 * 1024), 4)


def get_image_info(image_path):

    info = {}

    image = cv2.imread(image_path)

    if image is None:

        raise FileNotFoundError(f"Unable to load image: {image_path}")

    height, width = image.shape[:2]

    channels = image.shape[2] if len(image.shape) == 3 else 1

    info["Image Name"] = os.path.basename(image_path)

    info["Image Width"] = width

    info["Image Height"] = height

    info["Channels"] = channels

    info["Resolution"] = f"{width} x {height}"

    info["Aspect Ratio"] = round(width / height, 3)

    info["File Size (MB)"] = bytes_to_mb(os.path.getsize(image_path))

    extension = os.path.splitext(image_path)[1]

    info["Format"] = extension.replace(".", "").upper()

    info["Total Pixels"] = width * height

    info["Color Type"] = "Color" if channels == 3 else "Grayscale"

    return info


def print_image_info(info):

    print()

    print("=" * 90)

    print("INPUT IMAGE INFORMATION")

    print("=" * 90)

    for key, value in info.items():

        print(f"{key:<30}: {value}")
