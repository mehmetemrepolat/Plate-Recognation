import cv2

# video okuma
cap = cv2.VideoCapture(0)

# tracker tanımlama
tracker = cv2.TrackerKCF_create()

# ilk kare okuma
ret, frame = cap.read()

# bölge seçimi
bbox = cv2.selectROI("tracking", frame, False)

# tracker'ı başlat
tracker.init(frame, bbox)

while True:
    # kareleri okuma
    ret, frame = cap.read()
    if not ret:
        break

    # tracker'ı güncelleme
    success, bbox = tracker.update(frame)

    if success:
        # bölgeyi çerçeveleme
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        # tracker kaybı
        cv2.putText(frame, "Object lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # görüntüleri gösterme
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) == 27:
        break

# bellek boşaltma
cap.release()
cv2.destroyAllWindows()
