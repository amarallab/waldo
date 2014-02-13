def timeparse(time_in_secs, milisecs=False):
    """take time as seconds, return a tuple of
    years, months, days, hours, mins, secs
    (and miliseconds if chosen)"""
    t = time_in_secs
    years = t // (365 * 24 * 3600)
    t = t %  (365 * 24 * 3600)
    months = t // (30 * 24 * 3600)
    t = t % (30 * 24 * 3600)
    days = t // (24 * 3600)
    t = t % (24 * 3600)
    hours = t // 3600
    t = t % 3600
    mins = t // 60
    t = t % 60
    secs = t // 1
    t = t % 1
    milisecs = t //.01
    if milisecs:
        return years, months, days, hours, mins, secs, milisecs
    else:
        return years, months, days, hours, mins, secs



def timestr(time_in_secs):
    t = time_in_secs
    parsed = zip(('years','months','days',
                  'h','min','s'),
                 timeparse(t))
    info = ""
    for unit, val in parsed:
        if val == 1 and unit[-1] == 's' and len(unit) > 2:
            unit = unit[:-1]
        if val != 0:
            info += "%i %s " % (val, unit)

    if info == "": info = "0 s "

    return info[:-1]



if __name__ == '__main__':
    
    print timestr(134790293477.12341)

