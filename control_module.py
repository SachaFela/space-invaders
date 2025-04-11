import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets

# === Initialisation MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Fonctions utilitaires ===
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
    if abs(dx) > 50:
        return "LEFT" if dx < 0 else "RIGHT"
    return None

async def gesture_control():
    uri = "ws://localhost:8765"
    print("Connexion au jeu Space Invaders...")

    try:
        async with websockets.connect(uri) as websocket:
            print("Connecté !")
            print("Controles possibles :")
            print("- Pouce à gauche = GAUCHE")
            print("- Pouce à droite = DROITE")
            print("- Pouce et index proches = TIR")
            print("- Tapez 'enter' pour commencer")
            print("- Tapez 'a' pour quitter")

            loop = asyncio.get_running_loop()
            # Attente de la commande initiale "ENTER"
            while True:
                user_input = await loop.run_in_executor(None, input, "Commande : ")
                user_input = user_input.lower().strip()
                if user_input in ["enter", "s"]:
                    print("Envoi de ENTER...")
                    await websocket.send("ENTER")
                    print("Partie lancée !")
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        print(f"Réponse du serveur après ENTER : {response}")
                    except asyncio.TimeoutError:
                        print("Aucune réponse du serveur après ENTER")
                    break
                elif user_input == "a":
                    print("Quitter...")
                    return

            print("Contrôle gestuel activé...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Erreur : Impossible d'ouvrir la caméra")
                return

            frame_count = 0
            try:
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        print("Erreur : Impossible de lire l'image")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)
                    h, w, _ = frame.shape

                    command = None
                    posture = "aucun geste"

                    print("Analyse de l'image...")  # Log pour chaque frame
                    if results.multi_hand_landmarks:
                        print("Main détectée !")
                        for hand_landmarks in results.multi_hand_landmarks:
                            lm = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                            d_thumb_index = distance(lm[4], lm[8])
                            print(f"Distance pouce-index : {d_thumb_index}")
                            if d_thumb_index < 60:  # Seuil assoupli
                                command = "FIRE"
                                posture = "TIR (pouce-index)"
                            elif are_fingers_folded(lm):
                                print("Doigts repliés détectés")
                                direction = thumb_direction(lm)
                                if direction:
                                    command = direction
                                    posture = f"Pouce {direction}"

                    # Envoi de la commande (simplifié pour tester)
                    frame_count += 1
                    if command:
                        print(f"Commande détectée : {command}")
                        await websocket.send(command)
                        print(f"Envoi : {command}")
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            print(f"Réponse du serveur : {response}")
                        except asyncio.TimeoutError:
                            print(f"Aucune réponse du serveur pour {command}")

                    # Affichage
                    cv2.putText(frame, f"Geste : {posture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Space Invaders - Contrôle gestuel", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord('a'):
                        print("Arrêt demandé")
                        break

            finally:
                cap.release()
                cv2.destroyAllWindows()
                print("Caméra arrêtée")

    except websockets.exceptions.ConnectionClosed:
        print("Erreur : Connexion WebSocket fermée")
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    asyncio.run(gesture_control())
