import cv2
import numpy as np
import sys
import os

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

def load_image(image_path):
    """Load image from path and validate."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    return img

def preprocess_image(img):
    """Apply Gaussian blur and convert to grayscale."""
    img_blur = cv2.GaussianBlur(img, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    gray_img = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    return gray_img

def detect_edges(gray_img):
    """Detect edges using Laplacian."""
    laplacian_val = cv2.Laplacian(gray_img, cv2.CV_32F)
    _, laplacian_th = cv2.threshold(laplacian_val, thresh=LAPLACIAN_THRESHOLD, maxval=255, type=cv2.THRESH_BINARY)
    laplacian_med = cv2.medianBlur(laplacian_th, MEDIAN_KERNEL)
    return np.array(laplacian_med, dtype=np.uint8)

def detect_lines(edge_img, rho, theta, threshold):
    """Detect lines using Hough transform."""
    lines = cv2.HoughLines(edge_img, rho, theta, threshold)
    if lines is None or len(lines) == 0:
        return []
    return lines.tolist()

def filter_secondary_lines(secondary_lines, main_rho, main_theta):
    """Filter secondary lines based on alignment and proximity."""
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
    """Get intersection points of line with image boundaries."""
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    points = []
    # Find intersections with boundaries
    if x1 != x2:
        # Left boundary
        y_left = y1 + (y2 - y1) * (0 - x1) / (x2 - x1)
        if 0 <= y_left <= height:
            points.append((0, int(y_left)))
        # Right boundary
        y_right = y1 + (y2 - y1) * (width - x1) / (x2 - x1)
        if 0 <= y_right <= height:
            points.append((width, int(y_right)))
    if y1 != y2:
        # Top boundary
        x_top = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
        if 0 <= x_top <= width:
            points.append((int(x_top), 0))
        # Bottom boundary
        x_bottom = x1 + (x2 - x1) * (height - y1) / (y2 - y1)
        if 0 <= x_bottom <= width:
            points.append((int(x_bottom), height))

    return points[:2]  # Return two points

def create_polygon(points1, points2):
    """Create polygon from line points."""
    all_points = points1 + points2
    if len(all_points) < 4:
        print(f"Only {len(all_points)} intersection points, need at least 4")
        sys.exit(1)
    all_points.sort(key=lambda p: p[1])
    top_points = sorted(all_points[:2], key=lambda p: p[0])
    bottom_points = sorted(all_points[2:], key=lambda p: p[0])
    return np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype=np.int32)

def crop_to_strip(cropped_img):
    """Crop to the white strip region."""
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, THRESH_VALUE, 255, cv2.THRESH_BINARY)
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
    if not valid_contours:
        print("No valid contours found")
        sys.exit(1)
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
    """Ensure image is vertical."""
    if img.shape[1] > img.shape[0]:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def main():
    if len(sys.argv) != 2:
        print("Usage: python crop_strip.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = load_image(image_path)
    height, width = img.shape[:2]

    gray_img = preprocess_image(img)
    edge_img = detect_edges(gray_img)

    main_lines = detect_lines(edge_img, MAIN_RHO, MAIN_THETA, MAIN_THRESHOLD)
    if not main_lines:
        print("No main lines detected")
        sys.exit(1)
    main_line = main_lines[0][0]
    print(f"Main line: rho={main_line[0]:.2f}, theta={main_line[1]:.2f}")

    secondary_lines = detect_lines(edge_img, SECONDARY_RHO, SECONDARY_THETA, SECONDARY_THRESHOLD)
    filtered_lines = filter_secondary_lines(secondary_lines, main_line[0], main_line[1])
    if not filtered_lines:
        print("No secondary lines kept")
        sys.exit(1)
    secondary_line = filtered_lines[0][0]
    print(f"Secondary line: rho={secondary_line[0]:.2f}, theta={secondary_line[1]:.2f}")

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
    print(f"Cropped strip saved to {output_path}")

if __name__ == "__main__":
    main()