from django.conf.urls import patterns, include, url
from django.contrib import admin
from django.views.generic import RedirectView

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'webui.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),

    url(r'^collisions/', include('screencollisions.urls', namespace='collisions')),
    url(r'^gaps/', include('screengaps.urls', namespace='gaps')),
    url(r'^outcomes/', include('screenoutcomes.urls', namespace='outcomes')),
    url(r'^$', RedirectView.as_view(pattern_name='collisions:index', permanent=False), name='home'),

    url(r'^accounts/login/$', 'django.contrib.auth.views.login', name="login"),
    url(r'^accounts/logout/$', 'django.contrib.auth.views.logout', name="logout"),
)
