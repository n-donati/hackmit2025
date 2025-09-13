import cv2
import numpy as np
import os
import webcolors


def closest_color(requested_rgb):
    min_diff = None
    closest_name = None
    closest_hex = None

    for name in webcolors.names("css3"):
        hex_value = webcolors.name_to_hex(name, spec="css3")
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_value)
        diff = (r_c - requested_rgb[0]) ** 2 + (g_c - requested_rgb[1]) ** 2 + (b_c - requested_rgb[2]) ** 2

        if min_diff is None or diff < min_diff:
            min_diff = diff
            closest_name = name
            closest_hex = hex_value

    return closest_name, closest_hex


def get_band_color(image, y_start, y_end, center_x, band_width=15, white_thresh=240):
    h, w = image.shape[:2]
    x_start = max(0, center_x - band_width // 2)
    x_end = min(w, center_x + band_width // 2)

    band = image[y_start:y_end, x_start:x_end]
    if band.size == 0:
        return (0, 0, 0), (x_start, y_start, x_end, y_end)

    pixels = band.reshape(-1, 3)[:, ::-1]
    mask = np.all(pixels < white_thresh, axis=1)
    valid_pixels = pixels[mask]

    if len(valid_pixels) == 0:
        return (255, 255, 255), (x_start, y_start, x_end, y_end)

    avg_rgb = tuple(map(int, np.mean(valid_pixels, axis=0)))
    return avg_rgb, (x_start, y_start, x_end, y_end)


def analyze_strip_colors(image_path, output_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    height, width = image.shape[:2]
    num_sections = 52
    section_height = height // num_sections
    center_x = width // 2

    result_image = image.copy()
    colors_detected = []

    for i in range(3, num_sections + 1, 3):
        y_start = (i - 1) * section_height
        y_end = min(i * section_height, height)

        avg_rgb, band_box = get_band_color(image, y_start, y_end, center_x)
        color_name, color_hex = closest_color(avg_rgb)

        colors_detected.append({
            "section": i,
            "rgb": avg_rgb,
            "hex": color_hex,
            "color_name": color_name
        })

        (x1, y1, x2, y2) = band_box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(result_image, f"Section {i}: RGB{avg_rgb} (H:{y_end - y_start})", (5, y_start + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"Section {i}: RGB={avg_rgb}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result_image)
    return colors_detected


def main():
    input_path = "strips/output/output.jpg"
    output_path = "strips/output/strip_analysis.jpg"
    txt_path = "strips/output/colors.txt"

    try:
        colors = analyze_strip_colors(input_path, output_path)
        if colors:
            print("\nDetected sequence:")
            print(" -> ".join([f"RGB{c['rgb']}" for c in colors]))

            with open(txt_path, 'w') as f:
                for c in colors:
                    f.write(f"{c['rgb']}\n")
            print(f"Colors saved to {txt_path}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
