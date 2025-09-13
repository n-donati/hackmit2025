import os

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

def is_white_gray(rgb, threshold=170):
    """Check if RGB is close to white or gray."""
    r, g, b = rgb
    return r > threshold and g > threshold and b > threshold

def closest_color_index(rgb, color_list):
    """Find the index of the closest color in the list."""
    min_diff = float('inf')
    closest_idx = 0
    for i, ref_rgb in enumerate(color_list):
        diff = sum((a - b) ** 2 for a, b in zip(rgb, ref_rgb))
        if diff < min_diff:
            min_diff = diff
            closest_idx = i
    return closest_idx

def map_to_value(index, values_list):
    """Get the value corresponding to the index."""
    return values_list[index]

def main():
    txt_path = "cv/output/colors.txt"
    if not os.path.exists(txt_path):
        print(f"Error: {txt_path} not found")
        return
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    colors = []
    for line in lines:
        line = line.strip()
        if line:
            rgb_str = line.strip('()')
            r, g, b = map(int, rgb_str.split(','))
            colors.append((r, g, b))
    
    if not colors:
        print("No colors found")
        return
    
    if is_white_gray(colors[0]):
        print("First is white/gray, reversing list")
        colors.reverse()
    elif is_white_gray(colors[-1]):
        print("Last is white/gray, keeping order")
    else:
        print("No white/gray detected at ends")
    
    colors.pop()  # Remove the last color if it's whiteish
    
    test_names = list(test_data.keys())
    
    print("Color analysis:")
    for i, rgb in enumerate(colors):
        if i < len(test_names):
            test_name = test_names[i]
            data = test_data[test_name]
            ref_colors = data["colors"]
            values_list = data["values"]
            
            idx = closest_color_index(rgb, ref_colors)
            value = map_to_value(idx, values_list)
            print(f"Color {i+1}: RGB{rgb} -> Test '{test_name}' -> Index {idx} -> Value {value}")
        else:
            print(f"Color {i+1}: RGB{rgb} -> No corresponding test")

if __name__ == "__main__":
    main()
