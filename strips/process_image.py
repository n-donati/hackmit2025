import cv2
import numpy as np
import os
import base64
import tempfile
import webcolors

# Constants from crop_strip.py
GAUSSIAN_KERNEL = (9, 9)
GAUSSIAN_SIGMA = 0
LAPLACIAN_THRESHOLD = 3
MEDIAN_KERNEL = 3
MAIN_RHO = 1
MAIN_THETA = np.pi / 180
MAIN_THRESHOLD = 200
SECONDARY_RHO = 1
SECONDARY_THETA = np.pi / 180
SECONDARY_THRESHOLD = 50
ORIENTATION_TOLERANCE = 0.01
RHO_TOLERANCE = 40
THRESH_VALUE = 160
KERNEL_SIZE = 1
MIN_AREA = 700
OUTPUT_DIR = "strips/output"
OUTPUT_FILE = "output.jpg"

# Test data from get_values.py
test_data = {
    "Total Alkalinity": {
        "values": [240, 180, 120, 80, 40, 0],
        "colors": [(40,79,58),(55,91,56),(91,116,61),(117,123,63),(127,127,69),(175,151,71)]
    },
    "PH": {
        "values": [8.4, 7.8, 7.6, 7.2, 6.8, 6.2],
        "colors": [(223,53,35),(224,71,33),(219,94,46),(209,110,44),(191,122,57),(169,130,65)]
    },
    "Hardness": {
        "values": [425, 250, 100, 50, 25, 0],
        "colors": [(110,58,87),(92,62,93),(72,72,108),(63,79,121),(52,76,111),(30,67,100)]
    },
    "Hydrogen Sulfide": {
        "values": [10, 5, 3, 2, 1, 0.5, 0],
        "colors": [(116,57,34),(167,104,69),(171,118,79),(216,176,116),(223,198,144),(195,198,181),(180,180,180)]
    },
    "Iron": {
        "values": [5.0, 3.0, 1.0, 0.5, 0.3, 0.0],
        "colors": [(236,71,35),(231,95,37),(224,120,73),(215,145,117),(193,159,155),(180,180,180)]
    },
    "Copper": {
        "values": [5, 2, 1, 0.5, 0.2, 0],
        "colors": [(212,111,119),(233,128,115),(226,140,106),(215,147,110),(197,163,128),(177,175,143)]
    },
    "Lead": {
        "values": [50, 30, 15, 5, 0],
        "colors": [(190,67,54),(203,80,60),(209,104,74),(190,115,78),(173,142,80)]
    },
    "Manganese": {
        "values": [5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.0],
        "colors": [(143,49,76),(163,61,81),(179,73,75),(205,81,71),(210,99,75),(192,130,99),(175,167,95)]
    },
    "Total Chlorine": {
        "values": [20, 10, 5, 3, 1, 0.5, 0],
        "colors": [(66,128,120),(92,149,133),(132,167,126),(150,174,125),(165,183,133),(186,200,165),(163,179,158)]
    },
    "Free Chlorine": {
        "values": [20, 10, 5, 3, 1, 0.5, 0],
        "colors": [(74,128,130),(72,146,147),(114,164,158),(128,169,163),(147,179,179),(159,191,197),(180,180,180)]
    },
    "Nitrate": {
        "values": [500, 250, 100, 50, 25, 10, 0],
        "colors": [(221,65,90),(233,90,108),(240,123,127),(232,138,139),(215,165,166),(206,211,219),(180,180,180)]
    },
    "Nitrite": {
        "values": [80, 40, 20, 10, 5, 1, 0],
        "colors": [(209,44,77),(233,75,105),(237,98,113),(231,121,129),(215,146,157),(204,193,199),(180,180,180)]
    },
    "Sulfate": {
        "values": [1600, 1200, 800, 400, 200, 0],
        "colors": [(120,147,157),(138,150,152),(133,125,137),(128,123,150),(128,117,158),(103,88,137)]
    },
    "Zinc": {
        "values": [100, 50, 30, 10, 5, 0],
        "colors": [(82,101,141),(103,92,127),(139,89,114),(139,82,109),(150,82,98),(134,73,96)]
    },
    "Sodium Chloride": {
        "values": [2000, 1000, 500, 250, 100, 0],
        "colors": [(248,216,141),(251,206,139),(200,119,92),(160,78,77),(145,69,73),(114,63,62)]
    },
    "Fluoride": {
        "values": [100, 50, 25, 10, 4, 0],
        "colors": [(246,154,41),(253,143,48),(240,113,54),(220,93,59),(190,59,65),(135,53,65)]
    }
}

# Functions from crop_strip.py
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return img

def preprocess_image(img):
    img_blur = cv2.GaussianBlur(img, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    gray_img = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    return gray_img

def detect_edges(gray_img):
    laplacian_val = cv2.Laplacian(gray_img, cv2.CV_32F)
    _, laplacian_th = cv2.threshold(laplacian_val, thresh=LAPLACIAN_THRESHOLD, maxval=255, type=cv2.THRESH_BINARY)
    laplacian_med = cv2.medianBlur(laplacian_th, MEDIAN_KERNEL)
    return np.array(laplacian_med, dtype=np.uint8)

def detect_lines(edge_img, rho, theta, threshold):
    lines = cv2.HoughLines(edge_img, rho, theta, threshold)
    if lines is None or len(lines) == 0:
        return []
    return lines.tolist()

def filter_secondary_lines(secondary_lines, main_rho, main_theta):
    filtered = []
    for line in secondary_lines:
        rho, theta = line[0]
        theta_aligned = (abs(theta - main_theta) < ORIENTATION_TOLERANCE or
                         abs(theta - main_theta - np.pi) < ORIENTATION_TOLERANCE)
        rho_not_close = abs(rho - main_rho) > RHO_TOLERANCE
        if theta_aligned and rho_not_close:
            filtered.append(line)
    return filtered

def get_line_points(rho, theta, width, height):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    points = []
    if x1 != x2:
        y_left = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
        if 0 <= y_left <= height:
            points.append((0, int(y_left)))
        y_right = y1 + (y2 - y1) * (width - x1) / (x2 - x1)
        if 0 <= y_right <= height:
            points.append((width, int(y_right)))
    if y1 != y2:
        x_top = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
        if 0 <= x_top <= width:
            points.append((int(x_top), 0))
        x_bottom = x1 + (x2 - x1) * (height - y1) / (y2 - y1)
        if 0 <= x_bottom <= width:
            points.append((int(x_bottom), height))

    return points[:2]

def create_polygon(points1, points2):
    all_points = points1 + points2
    if len(all_points) < 4:
        raise ValueError("Not enough intersection points")
    all_points.sort(key=lambda p: p[1])
    top_points = sorted(all_points[:2], key=lambda p: p[0])
    bottom_points = sorted(all_points[2:], key=lambda p: p[0])
    return np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype=np.int32)

def crop_to_strip(cropped_img):
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, THRESH_VALUE, 255, cv2.THRESH_BINARY)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    if not valid_contours:
        raise ValueError("No valid contours found")
    all_x = []
    all_y = []
    for cnt in valid_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        all_x.extend([x, x + w])
        all_y.extend([y, y + h])
    x_min, y_min = min(all_x), min(all_y)
    x_max, y_max = max(all_x), max(all_y)
    return cropped_img[y_min:y_max, x_min:x_max]

def ensure_vertical(img):
    if img.shape[1] > img.shape[0]:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def crop_strip(image_path):
    img = load_image(image_path)
    height, width = img.shape[:2]

    gray_img = preprocess_image(img)
    edge_img = detect_edges(gray_img)

    main_lines = detect_lines(edge_img, MAIN_RHO, MAIN_THETA, MAIN_THRESHOLD)
    if not main_lines:
        raise ValueError("No main lines detected")
    main_line = main_lines[0][0]

    secondary_lines = detect_lines(edge_img, SECONDARY_RHO, SECONDARY_THETA, SECONDARY_THRESHOLD)
    filtered_lines = filter_secondary_lines(secondary_lines, main_line[0], main_line[1])
    if not filtered_lines:
        raise ValueError("No secondary lines kept")
    secondary_line = filtered_lines[0][0]

    points1 = get_line_points(main_line[0], main_line[1], width, height)
    points2 = get_line_points(secondary_line[0], secondary_line[1], width, height)
    polygon = create_polygon(points1, points2)

    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)
    cropped = img[y_min:y_max, x_min:x_max]

    final_cropped = crop_to_strip(cropped)
    final_cropped = ensure_vertical(final_cropped)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    cv2.imwrite(output_path, final_cropped)
    return output_path

# Functions from get_colors.py
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

def get_colors(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    height, width = image.shape[:2]
    num_sections = 52
    section_height = height // num_sections
    center_x = width // 2

    colors_detected = []

    for i in range(3, num_sections + 1, 3):
        y_start = (i - 1) * section_height
        y_end = min(i * section_height, height)

        avg_rgb, _ = get_band_color(image, y_start, y_end, center_x)
        color_name, color_hex = closest_color(avg_rgb)

        colors_detected.append({
            "section": i,
            "rgb": avg_rgb,
            "hex": color_hex,
            "color_name": color_name
        })

    return colors_detected

# Functions from get_values.py
def is_white_gray(rgb, threshold=170):
    r, g, b = rgb
    return r > threshold and g > threshold and b > threshold

def closest_color_index(rgb, color_list):
    min_diff = float('inf')
    closest_idx = 0
    for i, ref_rgb in enumerate(color_list):
        diff = sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb))
        if diff < min_diff:
            min_diff = diff
            closest_idx = i
    return closest_idx

def map_to_value(index, values_list):
    return values_list[index]

def get_values(colors):
    if not colors:
        return {}
    
    if is_white_gray(colors[0]["rgb"]):
        colors.reverse()
    elif is_white_gray(colors[-1]["rgb"]):
        pass
    else:
        pass
    
    colors.pop()  # Remove last
    
    test_names = list(test_data.keys())
    
    results = {}
    for i, color in enumerate(colors):
        if i < len(test_names):
            test_name = test_names[i]
            data = test_data[test_name]
            ref_colors = data["colors"]
            values_list = data["values"]
            
            idx = closest_color_index(color["rgb"], ref_colors)
            value = map_to_value(idx, values_list)
            results[test_name] = {
                "value": value
            }
        else:
            results[f"Extra {i+1}"] = {
                "value": None
            }
    
    return results

# Main processing function (combines all)
def process_image(base64_image):
    try:
        image_data = base64.b64decode(base64_image)
    except Exception as e:
        raise ValueError("Invalid base64 image")
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_file.write(image_data)
        temp_path = temp_file.name
    
    try:
        cropped_path = crop_strip(temp_path)
        
        colors = get_colors(cropped_path)
        
        values = get_values(colors)
        
        return values
    finally:
        os.unlink(temp_path)
        if os.path.exists(cropped_path):
            os.unlink(cropped_path)