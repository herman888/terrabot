import cv2
import numpy as np

# Load image
img = cv2.imread("sample1.jpg")

if img is None:
    print("Error: Could not load image")
    exit(1)

# Get image dimensions
height, width = img.shape[:2]

# Define region of interest (ROI) - exclude upper portion (sky)
# Ignore top 30% of image to exclude sky
sky_threshold = int(height * 0.3)
roi = img[sky_threshold:, :]

# Convert ROI to HSV color space for efficient processing
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Extract channels
h, s, v = cv2.split(hsv_roi)

# Calculate adaptive thresholds based on image statistics
# Snow characteristics: high brightness (V), low saturation (S)
v_mean = np.mean(v)
v_std = np.std(v)
s_mean = np.mean(s)
s_std = np.std(s)

# Dynamic threshold calculation
# For Value (brightness): look for pixels significantly brighter than average
min_value = max(200, int(v_mean + 0.8 * v_std))  # At least 200, or mean + 0.8 std
max_value = 255

# For Saturation: snow has low saturation
max_saturation = min(50, int(s_mean + 0.5 * s_std))  # Keep it low

print(f"Image stats - V mean: {v_mean:.1f}, V std: {v_std:.1f}")
print(f"Image stats - S mean: {s_mean:.1f}, S std: {s_std:.1f}")
print(f"Dynamic thresholds - V: [{min_value}, {max_value}], S: [0, {max_saturation}]")

# Define dynamic thresholds for snow
lower_snow = np.array([0, 0, min_value])
upper_snow = np.array([180, max_saturation, max_value])

# Create mask for snow in ROI
snow_mask_roi = cv2.inRange(hsv_roi, lower_snow, upper_snow)

# Apply morphological operations for noise reduction (optimized kernel size)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Smaller, more efficient kernel
snow_mask_roi = cv2.morphologyEx(snow_mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)
snow_mask_roi = cv2.morphologyEx(snow_mask_roi, cv2.MORPH_CLOSE, kernel, iterations=1)

# Create full-size mask with sky region excluded
snow_mask = np.zeros((height, width), dtype=np.uint8)
snow_mask[sky_threshold:, :] = snow_mask_roi

# Find contours for better visualization and area calculation
contours, _ = cv2.findContours(snow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by minimum area (remove noise)
min_contour_area = (width * height) * 0.001  # At least 0.1% of image
valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

print(f"Detected {len(valid_contours)} snow regions")

# Create visualization
output = img.copy()

# Draw sky exclusion line
cv2.line(output, (0, sky_threshold), (width, sky_threshold), (255, 0, 0), 2)
cv2.putText(output, "Sky Region (Ignored)", (10, sky_threshold - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Draw contours on output image
cv2.drawContours(output, valid_contours, -1, (0, 0, 255), 2)

# Highlight snow regions with semi-transparent overlay
highlight = img.copy()
highlight[snow_mask > 0] = [0, 0, 255]
output = cv2.addWeighted(output, 0.75, highlight, 0.25, 0)

# Add information text
total_snow_pixels = np.sum(snow_mask > 0)
snow_percentage = (total_snow_pixels / (width * height)) * 100
info_text = f"Snow coverage: {snow_percentage:.2f}%"
cv2.putText(output, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Snow Mask", snow_mask)
cv2.imshow("Snow Detection (Sky Excluded)", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
