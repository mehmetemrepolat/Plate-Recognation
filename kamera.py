def hazir_goruntu():
    import numpy as np
    import cv2
    import pytesseract


    goruntu = cv2.imread(f"Resim/{input('Resim adını Giriniz: ')}")
    img = goruntu
    cv2.imshow('Goruntu', img)

    # Çerçeveler halinde görüntü yakalar
    # Üzerinde işlem yapacağımız çerçeve buraya gelsin
    gray = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    plate_data = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
    number = plate_data.detectMultiScale(img, 1.2)
    print(str(len(number)), " Plaka tespit edildi.")
    if len(number) != 0:
        for numbers in number:
            (x, y, w, h) = numbers
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + h]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow('Plaka Tespit', img)

        cv2.imshow('Gri', roi_gray)

        ####Gürültü Temizleme islemi
        # C:\Program Files\Tesseract-OCR\tesseract.exe
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        noise_removal = cv2.bilateralFilter(roi_gray, 9, 75, 75)
        cv2.imshow('Gurultu Temizleme', noise_removal)

        ## Histogram eşitleme yapıldı
        equal_histogram = cv2.equalizeHist(noise_removal)
        cv2.imshow('Histogram Esitleme', equal_histogram)

        ## Morfolojik açılım
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations=15)
        sub_morp_image = cv2.subtract(equal_histogram, morph_image)

        ## Görüntüyü eşikleme
        ret, thresh_image = cv2.threshold(sub_morp_image, 0, 255, cv2.THRESH_OTSU)

        ## Canny Edge
        canny_image = cv2.Canny(thresh_image, 250, 255)
        cv2.imshow('Kenar Algilama', canny_image)

        canny_image = cv2.convertScaleAbs(canny_image)

        ## Genleştirme
        kernel = np.ones((3, 3), np.uint8)

        dilated_image = cv2.dilate(canny_image, kernel, iterations=1)

        ## Sonuç:
        cv2.imshow('Son Hali:', dilated_image)
        print(f"Plaka:{pytesseract.image_to_string(dilated_image)}")
    else:
        cv2.imshow('Görüntü', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


   

