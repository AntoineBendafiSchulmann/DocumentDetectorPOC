import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os
import random
import glob

input_image_path = "dataset/original/document.jpg"
output_dir = "dataset/generated/"
background_colors = [(255, 255, 255), (128, 128, 128), (0, 0, 0)] 

if os.path.exists(output_dir):
    files = glob.glob(os.path.join(output_dir, '*'))
    for f in files:
        os.remove(f)
else:
    os.makedirs(output_dir, exist_ok=True)

base_image = cv2.imread(input_image_path)

def add_background(image, color):
    h, w, _ = image.shape
    background = np.full((h + 100, w + 100, 3), color, dtype=np.uint8)
    background[50:h + 50, 50:w + 50] = image
    return background

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated

def perspective_transform(image):
    h, w = image.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts2 = np.float32([[random.randint(0, 20), random.randint(0, 20)], 
                       [w - random.randint(0, 20), random.randint(0, 20)], 
                       [random.randint(0, 20), h - random.randint(0, 20)], 
                       [w - random.randint(0, 20), h - random.randint(0, 20)]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, matrix, (w, h))

def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def blur_image(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def zoom_image(image, zoom_factor=1.1):
    h, w = image.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    resized = cv2.resize(image, (new_w, new_h))
    crop_h, crop_w = (new_h - h) // 2, (new_w - w) // 2
    return resized[crop_h:crop_h + h, crop_w:crop_w + w]

def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer_brightness = ImageEnhance.Brightness(pil_img)
    enhancer_contrast = ImageEnhance.Contrast(pil_img)
    img = enhancer_brightness.enhance(brightness)
    img = enhancer_contrast.enhance(contrast)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

count = 0
max_images = 100 
for color in background_colors:
    for angle in random.sample([10, 20, 30, -10, -20, -30], 3):
        for zoom in random.sample([1.0, 1.1, 1.2], 2):
            for brightness in random.sample([0.8, 1.0, 1.2], 2):
                for contrast in random.sample([0.8, 1.0, 1.2], 2):
                    transformed_image = add_background(base_image, color)
                    rotated_image = rotate_image(transformed_image, angle)
                    distorted_image = perspective_transform(rotated_image)
                    zoomed_image = zoom_image(distorted_image, zoom)
                    noisy_image = add_noise(zoomed_image)
                    blurred_image = blur_image(noisy_image)
                    final_image = adjust_brightness_contrast(blurred_image, brightness, contrast)

                    output_path = os.path.join(output_dir, f"doc_{count}.jpg")
                    cv2.imwrite(output_path, final_image)
                    count += 1

                    print(f"Image générée : {output_path}")

                    if count >= max_images:
                        print("Limite atteinte. Génération du dataset terminée.")
                        exit()

print("Génération du dataset terminée.")
