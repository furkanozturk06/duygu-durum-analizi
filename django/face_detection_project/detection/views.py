import os
from django.conf import settings
from django.shortcuts import render
from .forms import ImageUploadForm
from .face_detect import FaceDetect

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Fotoğrafı kaydet
            image = form.cleaned_data['image']
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)

            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Yüz tespiti yap
            detector = FaceDetect()
            processed_image_path = detector.analyze_image(image_path)

            # İşlenmiş görüntüyü frontend'e gönder
            return render(request, 'detection/result.html', {
                'processed_image_url': processed_image_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL),
                'original_image_url': image_path.replace(settings.MEDIA_ROOT, settings.MEDIA_URL),
            })

    else:
        form = ImageUploadForm()

    return render(request, 'detection/upload.html', {'form': form})
