import datetime

def ex_id_to_datetime(ex_id):
    ''' converts an experiment id to a datetime object '''     
    parts = ex_id.split('_')
    if len(parts) != 2:
        print 'Error: something is off with this ex_id', ex_id
        return None
    yearmonthday, hourminsec = parts
    year, month, day = map(int, [yearmonthday[:4], yearmonthday[4:6], yearmonthday[6:]])
    h, m, s = map(int, [hourminsec[:2], hourminsec[2:-2], hourminsec[-2:]])
    return datetime.datetime(year, month, day, h, m, s)


def ex_id_to_age(ex_id):
    ''' returns the age of the worms (hours) at the start of
    a plate in the N2_aging dataset.
    '''
    platetime1 = datetime.datetime(2013, 03, 16, 11, 00)
    platetime2 = datetime.datetime(2013, 04, 06, 11, 00)
    if int(ex_id.split('_')[0]) < 20130407:
        pt = platetime1
    else:
        pt = platetime2    
    return (ex_id_to_datetime(ex_id) - pt).total_seconds()/3600.
