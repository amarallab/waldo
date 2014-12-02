from __future__ import absolute_import, division, print_function
import six
from six.moves import (zip, filter, map, reduce, input, range)

import time
import uuid

try:
    clock = time.monotonic
except AttributeError:
    clock = time.time

from IPython.display import HTML, Javascript, display

MAX_UPDATE_INTERVAL = 0.2

class ProgressBar(object):
    def __init__(self, name=None):
        self.barid = str(uuid.uuid4())
        self.txtid = str(uuid.uuid4())
        self.lastupdate = -1e30

        if not name:
            name = 'Progress:'
        pb = HTML(
        """
        <div style='width:700px;clear:both'>
            <div style='width:150px;float:left;text-align:right;margin-right:10px'>{name}</div>
            <div style='width:440px;float:left'>
                <div class='progress' style='margin-bottom:5px'>
                    <div id='{barid}' class='progress-striped progress-info bar' role='progressbar' aria-valuenow='0' aria-valuemin='0' aria-valuemax='100'>&nbsp;</div>
                </div>
            </div>
            <div id='{txtid}' style='width:70px;float:left;text-align:right;margin-left:10px'>
                Starting...
            </div>
        </div>
        """.format(name=name, barid=self.barid, txtid=self.txtid))
        display(pb)

    def _bar(self, pct):
        display(Javascript("$('#{0}').css('width', {1:0.0f}+'%').attr('aria-valuenow', {1:0.0f});".format(self.barid, pct)))

    def _text(self, text):
        display(Javascript("$('div#{}').text('{}')".format(self.txtid, text)))

    def callback(self, progress):
        if progress == 1:
            self.done()
            return
        elif clock() < self.lastupdate + MAX_UPDATE_INTERVAL:
            return

        pct = 100 * progress
        self._bar(pct)
        self._text('{:0.1f}%'.format(pct))
        self.lastupdate = clock()

    def done(self):
        self._bar(100)
        self._text('Done!')
