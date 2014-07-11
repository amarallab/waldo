import os
from subprocess import Popen, PIPE, STDOUT
HERE = os.path.dirname(__file__)
C_PATH = os.path.join(HERE, 'cpp')


def compile_cpp(cpath=C_PATH):
    """
    runs the Makefile used to compile c++ scripts used for
    timeseries segementation.
    """
    # TODO make this robust.
    cur_dir = os.getcwd()
    os.chdir(cpath)
    os.system('make')
    os.chdir(cur_dir)


def breakpoints(x, y):
    """
    returns a list of breakpoints found using
    the fukuda algorithm for the segmentation of a timeseries
    into regions with stationary means.

    params
    -----
    x: (list of floats)
       the independent variable. (ie. index)
    y: (list of floats)
       the dependent variable which will be used for segmentation.

    returns
    -----
    breakpoints: (list of floats)
       the x coordinates denoting a shift in behavior.
    """
    c_path=os.path.abspath(C_PATH)
    cfile = '{path}/fukuda_breakpoints'.format(path=c_path)
    lines = ['{x} {y}'.format(x=xi, y=yi) for (xi, yi) in zip(x, y)]
    whole_file = '\n'.join(lines)
    process = Popen([cfile], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    stdout, stderr = process.communicate(input=whole_file)
    breakpoints = [s for s in stdout.split('\n') if s]
    if stderr:
        print stderr
    return breakpoints

def segments(x, y):
    """
    uses the fukuda algorithm for the segmentation of a timeseries
    into regions with stationary means.

    params
    -----
    x: (list of floats)
       the independent variable. (ie. index)
    y: (list of floats)
       the dependent variable which will be used for segmentation.

    returns
    -----
    segments: (list of lists)
        a list of segments, where each segement is a list of (x, y)
        tuples.
    """
    c_path=os.path.abspath(C_PATH)
    cfile = '{path}/fukuda_segmentation'.format(path=c_path)
    lines = ['{x} {y}'.format(x=xi, y=yi) for (xi, yi) in zip(x, y)]
    whole_file = '\n'.join(lines)
    process = Popen([cfile], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    stdout, stderr = process.communicate(input=whole_file)
    if stderr:
        print stderr
    segment, segments = [], []
    for line in stdout.split('\n'):
        if line:
            x, y = line.split()
            segment.append((float(x), float(y)))
        else:
            if segment:
                segments.append(segment)
            segment = []
    return segments


# old code which does not seem useful.
'''
def pipe_dat_into_C(dat_file, cfile, args=[]):

    assert os.path.isfile(dat_file), 'data file does not exist:{f}'.format(f=dat_file)
    assert os.path.isfile(cfile), 'compiled c code does not exist:{f}'.format(f=cfile)
    with open(dat_file, 'r') as f:
        whole_file = f.read()
    args = [cfile] + args
    print args
    process = Popen(args, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    stdout, stderr = process.communicate(input=whole_file)
    return stdout, stderr

def file_breakpoints(dat_file, c_path=C_PATH):
    cfile = '{path}/fukuda_breakpoints'.format(path=c_path.rstrip('/'))
    stdout, stderr = pipe_dat_into_C(dat_file=dat_file, cfile=cfile)
    if stderr:
        print stderr
    breakpoints = [s for s in stdout.split('\n') if s]
    return breakpoints


def file_fukuda_segmentation(dat_file, c_path=C_PATH):
    cfile = '{path}/fukuda_segmentation'.format(path=c_path.rstrip('/'))
    stdout, stderr = pipe_dat_into_C(dat_file=dat_file, cfile=cfile)
    if stderr:
        print stderr
    segment, segments = [], []
    for line in stdout.split('\n'):
        if line:
            x, y = line.split()
            segment.append((float(x), float(y)))
        else:
            if segment:
                segments.append(segment)
            segment = []
    return segments

def read_dat_file(filename):
    with open(filename, 'r') as f:
        lines = [l.strip('\n').split() for l in f.readlines()]
    t, x = zip(*lines)
    return map(float, t), map(float, x)



if __name__ == '__main__':
    dat_file = 'steps.dat'
    breaks = file_breakpoints(dat_file)
    print breaks
    #print fukuda_segmentation(dat_file)
'''
