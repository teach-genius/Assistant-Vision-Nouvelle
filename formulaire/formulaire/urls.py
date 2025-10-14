
from django.contrib import admin
from django.urls import path
from .views import bot_views,chat_api
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", bot_views, name="bot_vn"),
    path('api/chat/',chat_api, name='chat_api'),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

