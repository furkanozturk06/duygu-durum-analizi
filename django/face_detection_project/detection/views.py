import os
from django.conf import settings
from django.core.exceptions import SuspiciousOperation
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from django.utils.text import get_valid_filename
from .forms import ImageUploadForm
from .face_detect import FaceDetect

def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Fotoğrafı kaydet
            image = form.cleaned_data['image']

            # Dosya adini sanitize et (path traversal'i engellemek icin)
            safe_name = get_valid_filename(os.path.basename(image.name))

            # FileSystemStorage cakisan adlar icin benzersiz ad uretir ve
            # location disina yazmayi engeller.
            fs = FileSystemStorage(location=settings.MEDIA_ROOT)
            saved_name = fs.save(safe_name, image)

            # Olusan yolun MEDIA_ROOT icinde kaldigini dogrula.
            image_path = os.path.realpath(fs.path(saved_name))
            media_root = os.path.realpath(settings.MEDIA_ROOT)
            if os.path.commonpath([image_path, media_root]) != media_root:
                fs.delete(saved_name)
                raise SuspiciousOperation('Gecersiz dosya yolu.')

            # Yüz tespiti yap
            detector = FaceDetect()
            processed_image_path = detector.analyze_image(image_path)

            # İşlenmiş görüntü adini MEDIA_ROOT'a gore URL'e cevir
            processed_name = os.path.relpath(
                os.path.realpath(processed_image_path), media_root
            ).replace(os.sep, '/')

            # İşlenmiş görüntüyü frontend'e gönder
            return render(request, 'detection/result.html', {
                'processed_image_url': settings.MEDIA_URL + processed_name,
                'original_image_url': fs.url(saved_name),
            })

    else:
        form = ImageUploadForm()

    return render(request, 'detection/upload.html', {'form': form})
