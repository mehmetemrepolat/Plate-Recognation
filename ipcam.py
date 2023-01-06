
def IP_kamera():
    import numpy as np
    import cv2
    import pytesseract
    from PIL import Image
    import urllib
    rtsp_url = 'rtsp://admin:admin@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1'
    cap = cv2.VideoCapture(rtsp_url)




    while (True):
        # Çerçeveler halinde görüntü yakalar
        ret, frame = cap.read()
        img = frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plate_data = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
        number = plate_data.detectMultiScale(img, 1.2)

        if len(number) != 0:
            print(str(len(number)), " Plaka tespit edildi.")
        if len(number) != 0:
            for numbers in number:
                (x,y,w,h) = numbers
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+h]
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

            cv2.imshow('Plaka Tespit', img)

            gray = roi_gray
            ####Gürültü Temizleme islemi
            #C:\Program Files\Tesseract-OCR\tesseract.exe
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
            noise_removal = cv2.bilateralFilter(gray, 9, 75, 75)
            ## Daha iyi sonuç elde etmek için histogram eşitleme yapıldı
            equal_histogram = cv2.equalizeHist(noise_removal)
            ## Dikdörtgen yapı elemanı ile morfolojik açılım
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=15)
            ## Görüntü çıkarma (Morph görüntüsünü histogram eşitlenmiş görüntüsünden çıkarmak)
            sub_morp_image = cv2.subtract(equal_histogram, morph_image)
            ## Görüntüyü eşikleme
            ret, thresh_image = cv2.threshold(sub_morp_image, 0, 255, cv2.THRESH_OTSU)
            ## Canny Edge algılama uygulanması
            canny_image = cv2.Canny(thresh_image, 250, 255)
            ## Display Image
            canny_image = cv2.convertScaleAbs(canny_image)
            ## Kenarları güçlendirmek için genleşme
            kernel = np.ones((3, 3), np.uint8)
            ## Genişletme için çekirdek oluşturma
            dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
            ## Sonuç Çerçeveyi Görüntüleme:
            cv2.imshow('Son_Hal', dilated_image)
            print(f"Plaka:{pytesseract.image_to_string(dilated_image)}")
        else:
            cv2.imshow('Görüntü', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    # Destroy all windows
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    text = ""

