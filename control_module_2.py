import cv2
import asyncio
import websockets
from ultralytics import YOLO

# === CONFIGURATION ===
MODEL_PATH = "best.pt"  # Ton modèle est à la racine du repo
WS_URI = "ws://localhost:8765"
CLASSES_TO_COMMANDS = {
    'LEFT': 'LEFT',
    'RIGHT': 'RIGHT',
    'FIRE': 'FIRE'
}

# === DÉTECTION ET ENVOI DES COMMANDES ===
async def detect_and_send():
    print("📦 Chargement du modèle YOLOv8...")
    model = YOLO(MODEL_PATH)

    print("🔌 Connexion au serveur Space Invaders...")
    async with websockets.connect(WS_URI) as websocket:
        print("✅ Connecté !")
        input("Appuie sur ENTRÉE pour démarrer la partie...")
        await websocket.send("ENTER")
        print("🚀 Partie lancée !")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Erreur : impossible d’ouvrir la caméra")
            return

        last_command = None
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Erreur lecture caméra")
                break

            results = model.predict(source=frame, imgsz=416, conf=0.5, verbose=False)
            annotated = results[0].plot()

            names = results[0].names
            for cls_id in results[0].boxes.cls:
                class_name = names[int(cls_id)]
                if class_name in CLASSES_TO_COMMANDS:
                    command = CLASSES_TO_COMMANDS[class_name]
                    if command != last_command:
                        await websocket.send(command)
                        print(f"🖐️ Geste détecté : {class_name} → Commande envoyée : {command}")
                        last_command = command
                    break

            cv2.imshow("Détection YOLOv8 - Appuie sur ESC pour quitter", annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        print("👋 Fin du contrôle gestuel.")

# === LANCEMENT ===
if __name__ == "__main__":
    asyncio.run(detect_and_send())
