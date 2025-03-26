import cv2
import os
import numpy as np

# Пути к папкам
input_folder = "/home/aram/Desktop/FaceDetection/output/faces"
output_folder = "/home/aram/Desktop/FaceDetection/output/faces2"
os.makedirs(output_folder, exist_ok=True)

# Функция улучшения изображения
def enhance_face(image):
    # Увеличение резкости (Unsharp Mask)
    gaussian = cv2.GaussianBlur(image, (0, 0), 3)
    sharp = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)

    # Автоматическое улучшение контраста (CLAHE)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced

# Функция коррекции насыщенности и контраста после суперразрешения
def fix_colors(image, saturation_scale=1.2, contrast_alpha=1.1, brightness_beta=10):
    # Преобразуем в HSV, увеличиваем насыщенность
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, saturation_scale)  # Увеличиваем насыщенность
    s = np.clip(s, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Коррекция контраста и яркости
    final = cv2.convertScaleAbs(enhanced, alpha=contrast_alpha, beta=brightness_beta)
    
    return final

# Загрузка модели суперразрешения (EDSR x2)
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("/home/aram/Desktop/FaceDetection/models/EDSR_x2.pb")  # Убедись, что модель скачана
sr.setModel("edsr", 2)  # Увеличение в 2 раза

# Обработка всех изображений
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Ошибка загрузки {filename}, пропускаю...")
            continue

        # Улучшение изображения
        enhanced = enhance_face(image)

        # Применение суперразрешения
        upscaled = sr.upsample(enhanced)

        # Коррекция цветов после суперразрешения
        final_image = fix_colors(upscaled)

        # Сохранение результата
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, final_image)
        print(f"Обработано: {filename}")

print("✅ Все изображения обработаны и сохранены в папке output/")
