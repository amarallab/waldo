"""
Django settings for webui project.

For more information on this file, see
https://docs.djangoproject.com/en/dev/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/dev/ref/settings/
"""

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
import os
import pathlib
BASE_DIR = os.path.dirname(os.path.dirname(__file__))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/dev/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '1v=b%f_(mp$!i4oze5b=pm3gk0@e9-fvdox*vzdqtdo*c6i(^%'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

TEMPLATE_DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = (
    # django core
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # third party
    'bootstrap3',
    #'geordi',
    #'debug_toolbar.apps.DebugToolbarConfig',

    # ours
    'webui',
    'screencollisions',
    'screengaps',
    'screenoutcomes',
)

MIDDLEWARE_CLASSES = (
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.auth.middleware.SessionAuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    #'geordi.VisorMiddleware',
)

ROOT_URLCONF = 'webui.urls'

WSGI_APPLICATION = 'webui.wsgi.application'

# Database
# https://docs.djangoproject.com/en/dev/ref/settings/#databases
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# Internationalization
# https://docs.djangoproject.com/en/dev/topics/i18n/
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'America/Chicago'
USE_I18N = True
USE_L10N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/dev/howto/static-files/
STATIC_URL = '/static/'

SCREEN_BASE = (pathlib.Path(BASE_DIR) /
        '..' / # waldo/scripts/
        '..' / # waldo/
        'data' / 'screen')

COLLISION_IMAGES = SCREEN_BASE / 'collisions'
GAP_IMAGES = SCREEN_BASE / 'gaps'
OUTCOME_IMAGES = SCREEN_BASE / 'outcomes'

STATICFILES_DIRS = (
    str(COLLISION_IMAGES),
    str(GAP_IMAGES),
    str(OUTCOME_IMAGES),
)


