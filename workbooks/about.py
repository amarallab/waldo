from __future__ import print_function
import sys
import os
import platform
import socket

def about():
    print('Python {} ({}) [{}] on {}, Host: {}'.format(
            platform.python_version(),
            ', '.join(platform.python_build()),
            platform.python_compiler(),
            sys.platform,
            socket.gethostname(),
        ))
