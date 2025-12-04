from django.contrib import admin
from .models import UploadedImage

# Modeli admin paneline kaydedelim
class UploadedImageAdmin(admin.ModelAdmin):
    list_display = ('image', 'uploaded_at', 'processed', 'error_message')  # Görüntülenecek alanlar
    list_filter = ('processed',)  # Filtreleme için alan
    search_fields = ('error_message',)  # Arama yapılacak alanlar

# Admin paneline modelimizi kaydediyoruz
admin.site.register(UploadedImage, UploadedImageAdmin)
