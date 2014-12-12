from django.conf import settings
from django.conf.urls import patterns, include, url
from django.conf.urls.static import static
from django.contrib import admin
from django.views.generic import RedirectView

from . import views

urlpatterns = patterns('',
    url(r'^$', views.index, name='index'),
    url(r'^start$', views.start, name='start'),
    url(r'^outcome/(?P<eid>\w+)/(?P<bid>\w+)$',
        views.outcome, name='outcome'),
    url(r'^category/(?P<cat>\w+)$', views.category, name='cat')
)
