import cv2
import mediapipe as mp
import numpy as np

# Inițializare MediaPipe pentru mâini
mp_maini = mp.solutions.hands
maini = mp_maini.Hands(max_num_hands=2, min_detection_confidence=0.5)
desen = mp.solutions.drawing_utils

# Culori diferite pentru fiecare deget (BGR)
culori_degete = {
    'thumb': (0, 0, 255),       # roșu
    'index': (0, 255, 0),       # verde
    'middle': (255, 0, 0),      # albastru
    'ring': (0, 255, 255),      # galben
    'pinky': (255, 0, 255)      # mov
}

# Structura degetelor în funcție de landmark-uri MediaPipe
degete = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}

# Pornire cameră
camera = cv2.VideoCapture(1)  # dacă nu merge, schimbă în 1

while True:
    ok, cadru = camera.read()
    if not ok:
        print("❌ Nu se poate accesa camera.")
        break

    cadru = cv2.flip(cadru, 1)
    h, w, _ = cadru.shape
    fundal = 255 * np.ones((h, w, 3), dtype=np.uint8)

    # 🔹 Desenăm chipul static (față mică, proporționată)
    centru_x, centru_y = w // 2, h // 5
    cv2.circle(fundal, (centru_x - 40, centru_y - 10), 15, (0, 0, 0), -1)
    cv2.circle(fundal, (centru_x + 40, centru_y - 10), 15, (0, 0, 0), -1)
    puncte_nas = np.array([[centru_x, centru_y],
                           [centru_x - 12, centru_y + 25],
                           [centru_x + 12, centru_y + 25]], np.int32)
    cv2.fillPoly(fundal, [puncte_nas], (0, 0, 0))
    cv2.ellipse(fundal, (centru_x, centru_y + 55), (35, 15), 0, 0, 180, (0, 0, 0), 4)

    # Procesăm imaginea pentru MediaPipe
    rgb = cv2.cvtColor(cadru, cv2.COLOR_BGR2RGB)
    rezultat = maini.process(rgb)

    # 🔹 Detectăm și desenăm mâinile
    if rezultat.multi_hand_landmarks:
        for mana in rezultat.multi_hand_landmarks:
            # Mai întâi desenăm structura completă cu linii negre
            desen.draw_landmarks(
                fundal,
                mana,
                mp_maini.HAND_CONNECTIONS,
                desen.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=2),
                desen.DrawingSpec(color=(0, 0, 0), thickness=2)
            )

            # Apoi colorăm fiecare deget separat, peste structura originală
            h_img, w_img, _ = fundal.shape
            landmarke = mana.landmark
            puncte = [(int(lm.x * w_img), int(lm.y * h_img)) for lm in landmarke]

            for nume, idx in degete.items():
                culoare = culori_degete[nume]
                for i in range(len(idx) - 1):
                    cv2.line(fundal, puncte[idx[i]], puncte[idx[i + 1]], culoare, 4)
                for i in idx:
                    cv2.circle(fundal, puncte[i], 6, culoare, -1)

    # Afișăm imaginea finală
    cv2.imshow("Detectie maini colorate + chip static", fundal)

    # Iesire cu ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.release()
cv2.destroyAllWindows()
