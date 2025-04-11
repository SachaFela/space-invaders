import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets

# Initialisation MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Fonctions utilitaires
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def are_fingers_folded(landmarks):
    return all([
        landmarks[8][1] > landmarks[6][1],
        landmarks[12][1] > landmarks[10][1],
        landmarks[16][1] > landmarks[14][1],
        landmarks[20][1] > landmarks[18][1]
    ])

def thumb_direction(landmarks):
    dx = landmarks[4][0] - landmarks[2][0]
    if abs(dx) > 30:
        return "droite" if dx > 0 else "gauche"
    return None

# Programme principal asynchrone
async def detect_and_send():
    uri = "ws://localhost:8765"
    print("Connexion au jeu Space Invaders...")

    async with websockets.connect(uri) as websocket:
        print("Connecté ✅ (caméra activée pour les gestes)")
        cap = cv2.VideoCapture(0)
        frame_count = 0
        last_command = None

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            h, w, _ = frame.shape

            command = None
            posture = "aucune posture"

            if results.multi_hand_landmarks:
                hands_data = []
                for hand_landmarks in results.multi_hand_landmarks:
                    lm = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]
                    hands_data.append(lm)
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ❤️ Cœur avec 2 mains
                if len(hands_data) == 2:
                    lm1, lm2 = hands_data[0], hands_data[1]
                    center1 = np.mean(lm1, axis=0)
                    center2 = np.mean(lm2, axis=0)
                    if (
                        distance(lm1[4], lm1[8]) < 40 and
                        distance(lm2[4], lm2[8]) < 40 and
                        distance(center1, center2) < 100
                    ):
                        command = "ENTER"
                        posture = "coeur ❤️"

                # 👌 Rond / 👈👉 Pouce gauche/droite
                elif len(hands_data) == 1:
                    lm = hands_data[0]
                    d_thumb_index = distance(lm[4], lm[8])

                    if d_thumb_index < 40:
                        command = "FIRE"
                        posture = "rond 👌"
                    elif are_fingers_folded(lm):
                        direction = thumb_direction(lm)
                        if direction == "gauche":
                            command = "LEFT"
                            posture = "pouce 👈"
                        elif direction == "droite":
                            command = "RIGHT"
                            posture = "pouce 👉"

            # Envoi de commande si changement ou toutes les 10 frames
            frame_count += 1
            if command and (frame_count % 10 == 0 or command != last_command):
                await websocket.send(command)
                print(f"🖐️ Geste : {posture} → Commande envoyée : {command}")
                last_command = command

            # Affichage
            cv2.putText(frame, f"Posture : {posture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Contrôle gestuel - Space Invaders", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        cap.release()
        cv2.destroyAllWindows()

# Lancement du script
if __name__ == "__main__":
    asyncio.run(detect_and_send())
