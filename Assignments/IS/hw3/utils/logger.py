import logging
import logging.config

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} | {asctime} | {module} | {funcName} | {lineno} |{process:d} | {thread:d} | {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} | {asctime} | {module} | {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'app.log',
            'formatter': 'verbose',
            'maxBytes': 1024 * 1024,  # 1 MB
            'backupCount': 2,
        },
    },
    'loggers': {
        'basic': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}

logging.config.dictConfig(LOGGING)

logger = logging.getLogger('basic')
