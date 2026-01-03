import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("vegetable_model.keras")

# IMPORTANT: must match training folder order
# (we will verify this next)
class_names = ['Cabbage', 'Capsicum', 'Carrot', 'Potato', 'Tomato']

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

print("✅ Camera started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Center crop (improves accuracy)
    h, w, _ = frame.shape
    crop = frame[h//4:3*h//4, w//4:3*w//4]

    # Preprocess
    img = cv2.resize(crop, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    preds = model.predict(img, verbose=0)
    idx = np.argmax(preds)
    confidence = preds[0][idx] * 100
    label = class_names[idx]

    # Display only confident predictions
    if confidence > 80:
        text = f"{label} ({confidence:.1f}%)"
    else:
        text = "Detecting..."

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("Vegetable Prediction System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
