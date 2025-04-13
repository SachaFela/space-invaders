import cv2
import asyncio
import websockets
from ultralytics import YOLO
import time

# === CONFIGURATION ===
MODEL_PATH = "best.pt"  # Ton modèle est à la racine du repo
WS_URI = "ws://localhost:8765"
CLASSES_TO_COMMANDS = {
    'LEFT': 'LEFT',
    'RIGHT': 'RIGHT',
    'FIRE': 'FIRE'
}
COMMAND_COOLDOWN = 0.2  # Délai entre deux commandes (en secondes)

# === DÉTECTION ET ENVOI DES COMMANDES ===
async def detect_and_send():
    print("📦 Chargement du modèle YOLOv8...")
    try:
        model = YOLO(MODEL_PATH)
        print("✅ Modèle chargé")
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return

    print("🔌 Connexion au serveur Space Invaders...")
    try:
        async with websockets.connect(WS_URI) as websocket:
            print("✅ Connecté au serveur WebSocket")
            input("Appuie sur ENTRÉE pour démarrer la partie...")

            print("📤 Envoi de la commande ENTER...")
            await websocket.send("ENTER")
            print("✅ Commande ENTER envoyée")

            # 🔧 Attente de la réponse du serveur
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                print(f"🛰️ Réponse du serveur après ENTER : {response}")
            except asyncio.TimeoutError:
                print("⚠️ Aucune réponse du serveur après ENTER")

            print("🚀 Partie lancée !")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("❌ Erreur : impossible d’ouvrir la caméra")
                return

            last_command = None
            last_command_time = 0

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("❌ Erreur lecture caméra")
                        break

                    # Prédiction avec YOLO
                    start_time = time.time()
                    try:
                        results = model.predict(source=frame, imgsz=320, conf=0.4, verbose=False)
                        annotated = results[0].plot()
                    except Exception as e:
                        print(f"❌ Erreur lors de la prédiction YOLO : {e}")
                        continue

                    print(f"🕒 Temps de détection : {time.time() - start_time:.3f} s")

                    # Vérification des détections
                    names = results[0].names
                    command = None
                    detected_classes = [names[int(cls_id)] for cls_id in results[0].boxes.cls]
                    print(f"🎯 Classes détectées : {detected_classes}")

                    for cls_id in results[0].boxes.cls:
                        class_name = names[int(cls_id)]
                        if class_name in CLASSES_TO_COMMANDS:
                            command = CLASSES_TO_COMMANDS[class_name]
                            break

                    # Envoi de la commande
                    current_time = time.time()
                    if command and current_time - last_command_time >= COMMAND_COOLDOWN:
                        print(f"🖐️ Geste détecté : {class_name} → Commande : {command}")
                        try:
                            await websocket.send(command)
                            print(f"📤 Commande envoyée : {command}")

                            # 🔧 Lire la réponse du serveur
                            try:
                                response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                                print(f"📨 Réponse du serveur : {response}")
                            except asyncio.TimeoutError:
                                print(f"⌛ Aucune réponse du serveur pour {command}")

                            last_command = command
                            last_command_time = current_time
                        except Exception as e:
                            print(f"❌ Erreur lors de l’envoi de la commande : {e}")

                    # Affichage de l’image
                    cv2.imshow("Détection YOLOv8 - Appuie sur ESC pour quitter", annotated)

                    # Attente minimale pour OpenCV
                    if cv2.waitKey(1) & 0xFF == 27:
                        print("🛑 Arrêt demandé")
                        break

            finally:
                cap.release()
                cv2.destroyAllWindows()
                print("👋 Fin du contrôle gestuel.")

    except websockets.exceptions.ConnectionClosed:
        print("❌ Erreur : Connexion WebSocket fermée")
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")

# === LANCEMENT ===
if __name__ == "__main__":
    asyncio.run(detect_and_send())
