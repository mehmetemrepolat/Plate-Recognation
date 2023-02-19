import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
old_gray = gray.copy()
old_frame = frame.copy()
lk_params = dict(winSize=(100, 100),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
p0 = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, qualityLevel=0.3, minDistance=100, blockSize=7)

is_object_moving = True
time_object_stopped_moving = None

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # İleriki konumda objenin anahtar noktalarını belirleyin
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]
    mask = np.zeros_like(old_frame)
    color = np.random.randint(0, 255, (100, 3))

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 0, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)

    img = cv2.add(frame, mask)

    # Obje hareket etmediyse yeşil noktaların rengini kırmızıya çevir
    if np.all(good_new == None) and is_object_moving:
        time_object_stopped_moving = time.time()
        is_object_moving = False
    elif np.all(good_new != None) and not is_object_moving:
        is_object_moving = True

    if not is_object_moving:
        if time.time() - time_object_stopped_moving > 1:
            img[np.where((img == [0, 255, 0]).all(axis=2))] = [0, 0, 255]

    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    old_gray = gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
