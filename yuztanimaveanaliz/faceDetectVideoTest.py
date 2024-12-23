import cv2
from faceDetect import FaceDetect


def main():
    try:
        # OpenCV ile görüntü yakalamak için WebCam'i kullanıyoruz
        capture = cv2.VideoCapture(0)

        if not capture.isOpened():
            raise ValueError("Kamera açılamadı. Lütfen cihazınızı kontrol edin.")

        # Yüz tanıma sınıfımızı kullanabilmek için bir nesne oluşturuyoruz
        face_detect = FaceDetect()

        # Sonsuz bir döngü ile kameradan sürekli görüntü alacağız
        while True:
            # Her döngü yenilenmesinde bir kare alıyoruz
            ret, frame = capture.read()

            if not ret:
                raise ValueError("Kare alınamadı. Kamerayı kontrol edin.")

            detect = face_detect.run(frame)['frame']

            # Görüntü çıktısını alıyoruz
            cv2.imshow('frame', detect)

            # Q tuşu ile döngünün kırılmasını sağlıyoruz
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Hata oluştu: {e}")

    finally:
        # Kamerayı kapatıyoruz
        capture.release()

        # Tüm pencereleri sonlandırıyoruz
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
