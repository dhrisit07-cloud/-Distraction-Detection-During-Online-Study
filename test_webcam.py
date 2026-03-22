import cv2

# CAP_MSMF is required for Windows built-in webcams
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Try changing index 0 to 1 or 2.")
    exit()

print("Webcam opened successfully! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame.")
        break

    cv2.putText(frame, "Webcam OK - Press Q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Focus Guardian - Webcam Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")
