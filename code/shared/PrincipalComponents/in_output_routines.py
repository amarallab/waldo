


def get_data_ids_from_file(listfile):
    
    """
        first entry is time, which is skipped
        input file such as 
        id var var var ....
        id var var var ....
    """
    x=[]
    ids=[]
    f=open(listfile, "r")
    for l in f.readlines():
        s=l.split()
        dat=[]
        for k in s[1:]:
            dat.append(eval(k))
        if len(dat)>0:
            x.append(dat)
        ids.append(s[0])
    
    f.close()
    return x, ids




def get_data_from_file(listfile):
    
    """
        simply sets x as list of lists
    """
    x=[]
    f=open(listfile, "r")
    for l in f.readlines():
        s=l.split()
        dat=[]
        for k in s:
            dat.append(eval(k))
        if len(dat)>0:
            x.append(dat)
    
    f.close()
    return x




def read_data_and_put_in_a_list(filename):
    
    
    x=[]
    f=open(filename, "r")
    for l in f.readlines():
        s=l.split()
        for k in s:
            x.append(eval(k))
    f.close()
    return x




def write_list_of_lists(filename, list_of_lists):
    
    f=open(filename, "w")
    for l in list_of_lists:
        for w in l:
            f.write(str(w)+" ")
        f.write("\n")
    f.close()




def write_list_of_lists_with_headers(filename, headers, list_of_lists):


    """
        headers is a list of strings, one for each list in list_of_lists
    """
    
    
    f=open(filename, "w")
    for k,l in enumerate(list_of_lists):

        f.write(headers[k]+" ")
        for w in l:
            f.write(str(w)+" ")
        f.write("\n")
    f.close()





