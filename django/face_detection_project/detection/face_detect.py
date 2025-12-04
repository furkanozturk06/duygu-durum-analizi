import cv2

class FaceDetect:
    def analyze_image(self, image_path):
        # OpenCV ile yüz tespiti
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Yüzleri tespit et
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Yüzleri işaretle
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # İşlenmiş görüntüyü kaydet
        output_path = image_path.replace('.', '_processed.')
        cv2.imwrite(output_path, image)

        return output_path
