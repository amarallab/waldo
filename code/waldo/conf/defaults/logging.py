from __future__ import absolute_import
import logging

#LOG_THRESHOLD = logging.WARN

LOG_CONFIGURATION = {
    'version': 1,
    'disable_existing_loggers': False,

    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'WARN',
            'class':'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'WARN',
            'propagate': True
        }
    }
}
