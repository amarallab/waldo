import datetime
import inspect
import sys

time_format = '%H:%M:%S.%f'

begin_time = None
last_tag_time = None

def begin(s=None):
    global begin_time, last_tag_time
    begin_time = datetime.datetime.now()
    last_tag_time = begin_time
    print >>sys.stderr, "I: %s" % "--" if s is None else s

def tag(s=None):
    global last_tag_time
    now = datetime.datetime.now()
    elapsed = now - last_tag_time
    last_tag_time = now
    back = inspect.currentframe().f_back
    print >>sys.stderr, "D(%s:%d, %s, %s): %s" % (back.f_code.co_filename, back.f_lineno, now.strftime(time_format), str(elapsed), "--" if s is None else s)


def end(s=None):
    now = datetime.datetime.now()
    elapsed = now - last_tag_time
    total_elapsed = now - begin_time
    print >>sys.stderr, "I(elapsed time: %s, total: %s): %s" % (str(elapsed), str(total_elapsed), "--" if s is None else s)
