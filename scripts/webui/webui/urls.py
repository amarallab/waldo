from django.conf.urls import patterns, include, url
from django.contrib import admin
from django.views.generic import RedirectView

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'webui.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),

    url(r'^collisions/', include('screencollisions.urls', namespace='sc')),
    url(r'^gaps/', include('screengaps.urls', namespace='sg')),
    url(r'^$', RedirectView.as_view(pattern_name='sc:index', permanent=False), name='home'),

    url(r'^accounts/login/$', 'django.contrib.auth.views.login', name="login"),
    url(r'^accounts/logout/$', 'django.contrib.auth.views.logout', name="logout"),
)
