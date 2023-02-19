import cv2
import numpy as np

# Webcamden video akışı için VideoCapture sınıfını kullanıyoruz
cap = cv2.VideoCapture(0)

# ShiTomasi köşe algılama için parametreleri belirliyoruz
# Orjinal kodda lk_params değişkeni ismiyle kullanılıyor
# Burada shi_params ismini kullanacağız
shi_params = dict( maxCorners = 10,
                   qualityLevel = 0.3,
                   minDistance = 7,
                   blockSize = 7 )

# Lucas Kanade optik akış algoritması için parametreleri belirliyoruz
# Orjinal kodda shi_params değişkeni ismiyle kullanılıyor
# Burada lk_params ismini kullanacağız
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Rastgele renkler seçiyoruz
color = np.random.randint(0,255,(10,3))

# İlk kareyi yakalıyoruz
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Köşeleri belirliyoruz
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **shi_params)

# Maske oluşturuyoruz
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Optik akışı hesaplıyoruz
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Sadece iyi eşleşen köşeleri seçiyoruz
    good_new = p1[st==1]
    good_old = p0[st==1]

    # Her köşe için hareket yönünü çiziyoruz
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)

        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    # İkili görüntü olarak iki kare arasındaki farkı alıyoruz
    img = cv2.add(frame,mask)

    # Görüntüleri gösteriyoruz
    cv2.imshow('frame',img)

    # Çıkmak için ESC tuşuna basın
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Bir sonraki kare için hazırlık yap
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
