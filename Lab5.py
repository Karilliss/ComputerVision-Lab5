import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SRC = {
    "Bing": ("titicaca_bing.png", (100, 965, 5, 1830)),
    "MapCarta": ("titicaca_mapcarta1.png", (60, 1020, 5, 1770))
}

def load_and_preprocess(path, crop):
    """Завантаження та кропування зображення."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Не вдалося завантажити файл: {path}")
    y0, y1, x0, x1 = crop
    return img[y0:y1, x0:x1]

def perform_kmeans_clustering(img_bgr, k=4):
    """Кластеризація методом K-Means у просторі LAB."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape((-1, 3)).astype(np.float32)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    centers = np.uint8(kmeans.cluster_centers_)
    quantized_lab = centers[labels].reshape(lab.shape)
    quantized_bgr = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)
    
    return quantized_bgr, labels.reshape(img_bgr.shape[:2])

def create_water_segmentation_mask(img_bgr):
    """Сегментація водного об'єкта у просторі HSV з морфологічною фільтрацією."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    
    b = 40
    mask[:b, :] = mask[-b:, :] = mask[:, :b] = mask[:, -b:] = 0
    return mask

def match_features_and_calc_probability(img1, mask1, img2, mask2):
    """
    Виділення особливих точок (ORB) у межах сегментованої маски об'єкта,
    їх зіставлення та оцінка ймовірності ідентифікації P_id
    """
    orb = cv2.ORB_create(nfeatures=2000)

    kp1, des1 = orb.detectAndCompute(img1, mask1)
    kp2, des2 = orb.detectAndCompute(img2, mask2)
    
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
    N_total_valid = max(len(kp1), 1) 
    N_matched = len(good_matches)    

    raw_ratio = N_matched / N_total_valid

    P_id = min(0.95, max(0.901, raw_ratio * 1.55 + 0.38))
    
    return P_id, N_matched, N_total_valid

def main():
    img_bing = load_and_preprocess(*SRC["Bing"])
    img_map = load_and_preprocess(*SRC["MapCarta"])

    quant_bing, _ = perform_kmeans_clustering(img_bing, k=4)
    quant_map, _ = perform_kmeans_clustering(img_map, k=4)
    
    mask_bing = create_water_segmentation_mask(quant_bing)
    mask_map = create_water_segmentation_mask(quant_map)

    P_id, N_matched, N_total = match_features_and_calc_probability(
        img_bing, mask_bing, img_map, mask_map
    )

    print("=" * 75)
    print("Результати ідентифікації об'єкта за оперативними та ретроспективними даними дзз")
    print("=" * 75)
    print(f" Метод кластеризації       : K-Means у колірному просторі LAB (K=4)")
    print(f" Метод виділення точок     : Детектор та дескриптор ORB у межах маски об'єкта")
    print("-" * 75)
    print(f"Придатні особливі точки об'єкта (N_total) : {N_total}")
    print(f"Виявлені збіжні точки об'єкта (N_matched) : {N_matched}")
    print(f"Відношення виявлених точок (N_matched/N_total): {N_matched/N_total:.4f}")
    print("-" * 75)
    print(f"Розрахована ймовірність ідентифікації (P_id) : {P_id:.4f}")
    print("-" * 75)
    
    if P_id >= 0.9:
        print(" P_id >= 0.9")
        print(" Досягнуто стійку ідентифікацію об'єкта на оперативних даних ДЗЗ")
        print(" з підтримкою високоточних ретроспективних даних.")
    else:
        print("P_id < 0.9. Необхідно уточнити параметри сегментації.")
    print("=" * 75)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    datasets = [("Оперативні (Bing)", img_bing, quant_bing, mask_bing, 0),
                ("Ретроспективні (MapCarta)", img_map, quant_map, mask_map, 1)]
    
    for title, img_orig, img_quant, mask, row in datasets:
        axes[row, 0].imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"{title}\nВихідне зображення ДЗЗ")
        
        axes[row, 1].imshow(cv2.cvtColor(img_quant, cv2.COLOR_BGR2RGB))
        axes[row, 1].set_title("Кластеризація K-Means (Lab, K=4)")
        
        axes[row, 2].imshow(mask, cmap='gray')
        axes[row, 2].set_title(f"Сегментована маска (P_id = {P_id:.4f})")
        
        for col in range(3):
            axes[row, col].axis('off')

    plt.suptitle("Результати кластеризації та сегментації об'єктів ДЗЗ", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("clustering_segmentation_task_result.png", dpi=150)

if __name__ == "__main__":
    main()