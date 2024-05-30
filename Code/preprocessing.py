import os
import cv2


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)

def enhance_contrast(image, alpha=0.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_equalized = cv2.equalizeHist(l)
    # Blend the original L channel with the equalized L channel
    l_blended = cv2.addWeighted(l, 1 - alpha, l_equalized, alpha, 0)
    lab = cv2.merge((l_blended, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced_image

def normalize_image(image):
    return image / 255.0

def preprocess_image(image_path, alpha=0.5):
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read the image from {image_path}")

    blurred_image = apply_gaussian_blur(original_image)
    enhanced_image = enhance_contrast(blurred_image, alpha)
    normalized_image = normalize_image(enhanced_image)

    return normalized_image

# Specify the folder containing your images
input_folder = "./keyframes_farneback_vdo3"
output_folder = "./preprocessed_images"

# Create the output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        input_image_path = os.path.join(input_folder, filename)
        preprocessed_image = preprocess_image(input_image_path)
        
        output_image_path = os.path.join(output_folder, filename)
        
        # Save the final preprocessed image
        cv2.imwrite(output_image_path, (preprocessed_image * 255).astype('uint8'))

print("Preprocessing complete! Preprocessed images saved in:", output_folder)
