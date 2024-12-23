# Bulunan yüz verisini ölçeklendirip numpy ve keras ile işliyoruz
                    roi_gray = cv2.resize(roi_gray, (64, 64))
                    roi_gray = roi_gray.astype('float') / 255.0
                    roi_gray = img_to_array(roi_gray)
                    roi_gray = np.expand_dims(roi_gray, axis=0)