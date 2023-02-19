import cv2
import numpy as np
# Görüntü akışını başlat
cap = cv2.VideoCapture(0)

# Seçilen alanın koordinatları
x, y, w, h = 0, 0, 0, 0

# Alan seçim işlevi
def select_roi(event, x_in, y_in, flags, param):
    global x, y, w, h
    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = x_in, y_in
    elif event == cv2.EVENT_LBUTTONUP:
        w, h = x_in - x, y_in - y
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Pencere oluştur ve alan seçimi işlevini bağla
cv2.namedWindow('Goruntu')
cv2.setMouseCallback('Goruntu', select_roi)

# Nesne takip işlevi
def track_object(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Renk histogramını hesaplama
    roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Takip için ayarlar
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Takip için CAMSHIFT algoritması
        ret, track_window = cv2.CamShift(dst, (x, y, w, h), term_crit)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)


        cv2.imshow('Goruntu', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Goruntu', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if w > 0 and h > 0:
        track_object(frame, x, y, w, h)

cap.release()
cv2.destroyAllWindows()