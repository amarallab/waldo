from math import floor
import time
import sys

from Shared.Code.Plotting.HeatMap.timeparse import timestr


# step in in percentages

# timestep is in seconds

def progress_bar(num_iterations, step=10, timestep=120):
    """Generator to keep track of progress.
    Use this with your rate determining loop.
    Shouts the progress (percentage and time) at every
    percentage step or every timestep seconds, whichever
    comes first."""
    counter = 1
    completed = 0
    starttime = time.time()
    time_shout = starttime
    shout = False

    # while percent <= 100:
    while True:
        Did_I_just_shout = False
        percent =  100.*counter/num_iterations
        time_passed = time.time() - starttime
        time_since_last_shout = time.time() - time_shout
        time_remaining_est =  time_passed * (100. - percent) / percent
        counter += 1
        percent = floor(percent)

        if percent > completed:
            completed = percent
            if completed % step == 0 or completed == 100:
                shout = True

        if time_since_last_shout >= timestep:
            shout = True

        if shout:
            print >> sys.stderr, "  %.0f%% completed." % completed
            print >> sys.stderr, '  %s elapsed.' % timestr(time_passed)
            print >> sys.stderr, '  %s remaining.' % timestr(time_remaining_est)
            time_shout = time.time()
            Did_I_just_shout = True
            shout = False
            
        yield Did_I_just_shout


#########################################
# Iterator Class version (early - scrapped)
class Progress_Bar:
    def __init__(self, num_iterations, step=1):
        self.num_iterations = num_iterations
        self.step = step
        self.counter = 1
        self.percent = 0
        self.completed = 0
    def __iter__(self):
        return self
    def next(self):
        self.percent =  floor(100.*self.counter/self.num_iterations)
        if  (self.percent % self.step == 0 or self.percent == 100) \
                and self.percent > self.completed:
            self.completed = self.percent
            print >> sys.stderr, "  %.0f%% completed" % self.completed
        # if self.percent >= 100: raise StopIteration
        self.counter += 1
############################################




# Example of use

if __name__ == '__main__':

    p = progress_bar(250, step=10, timestep=10)
    for i in xrange(250):
        ##### doing your stuff ##########
        calculations = 6 * 7            #
        weird = 'stuff' + 'in the loop' #
        time.sleep(2)                   #
        #################################
        p.next()


