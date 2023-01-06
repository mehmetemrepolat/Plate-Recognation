#import islem
import resim
import kamera
import ipcam


print("Kamera'dan plaka tanıma işlemi yapmak için 1 giriniz:")
print("Var Olan Resim ile işlem için 2 giriniz:")
print("IP Kameradan plaka tanıma işlemi yapmak için 3 giriniz:")

try:
    secim = int(input("Seçiminiz:"))

    if secim == 1:
        resim.kamera()
    elif secim == 2:
        kamera.hazir_goruntu()
    elif secim == 3:
        ipcam.IP_kamera()
        # ipcam
    else:
        exit()

except ValueError:
    print("Hatalı seçim Lütfen menüden seçim yapınız.")

