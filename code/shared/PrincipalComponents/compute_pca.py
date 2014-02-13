
# standard imports
import os
import numpy as np

# nonstandard imports
from in_output_routines import read_data_and_put_in_a_list, \
    write_list_of_lists, write_list_of_lists_with_headers
from utilities import standardize_matrix, euclidean_cosine_distances


def write_file_for_R(data_in_rows, filename):
    """
        data_in_rows is a list of lists
        data_in_rows[0] is a list of data_in_rows for snapshot 0
        filename is the file where to write the data_in_rows
    """
    try:
        len_data_in_rows = len(data_in_rows[0])
        f = open(filename, "w")
        for k in xrange(len_data_in_rows):
            f.write("v" + str(k) + " ")
        f.write("\n")
        for datavar_list in data_in_rows:
            for a in datavar_list:
                f.write(str(a) + " ")
            f.write("\n")
        f.close()
    except Exception as e:
        print e, "\ndata_in_rows is empty"


def read_components(number_of_prcs, outfile):
    x = []
    for k in xrange(1, number_of_prcs + 1):
        x.append(read_data_and_put_in_a_list(outfile + "_" + str(k) + ".txt"))
    return x


def compute_loadings(r, components, just_this=-1):
    """
        r is the real datavar vector
        components are the principal components
        the function returns the loadings
        just_this is to restrict the projection along just_this components
        if just_this==-1 all the components are considered
    """

    if just_this == -1:
        cs2 = components
    else:
        cs2 = []
        cs2.append(components[just_this])

    weights = []

    for c in cs2:
        weights.append(np.dot(r, c))

    return weights


def compute_all_weights(std_data, components):
    weights = []
    for r in std_data:
        weights.append(compute_loadings(r, components))
    return weights


def compute_approx_data(weights, av_stds, components, just_this=-1):
    """
        weights are the loadings respect with the components
        components are the principal components
        components[0] is a list of datavars of the principal component 0
        the function returns the approximated ones
        av_stds are the averages and stds of the variables
        just_this is to restrict the projection along just_this components
        if just_this==-1 all the components are considered
    """

    if just_this == -1:
        cs2 = components
    else:
        cs2 = [components[just_this]]

    approx_datavars = []
    for datavar_j, _ in enumerate(cs2[0]):
        approx_datavar = 0.
        for j2 in xrange(min(len(weights), len(cs2))):
            approx_datavar += weights[j2] * cs2[j2][datavar_j]
        approx_datavars.append(av_stds[datavar_j][0] + av_stds[datavar_j][1] * approx_datavar)
    return approx_datavars


def compute_approx_data_all(weights, av_stds, components):
    """
        simply loops compute_approx_data over all the weights
    """

    approx_data = []
    for w in weights:
        approx_data.append(compute_approx_data(w, av_stds, components))
    return approx_data


def write_R_script(outfile, number_of_prcs):
    f = open("temp_R_script.R", "w")
    f.write("rm(list=ls())\n")
    f.write("Ta<-read.table(\"temp_datafile_for_R.txt\", header=TRUE)\n")
    f.write("Ta.pca <- prcomp(Ta, retx=TRUE, center=TRUE, scale.=TRUE)\n")
    f.write("A=t(Ta.pca$rotation)\n")
    f.write("for(i in 1:" + str(number_of_prcs) + ") {\n")
    f.write("\twrite(A[i,], paste(\"" + outfile + "_\",i,\".txt\", sep=\"\"))\n")
    f.write("}\n")
    f.write("write(Ta.pca$sdev, \"temp_R_dev.txt\")\n")
    f.close()


def execute_R(data_in_rows, outfile="temp_principal_components", del_temp=True, number_of_prcs=-1):
    """
        data_in_rows[0] is a list with the variables in the first snapshot 
        number_of_prcs is the number of principal components that we what R to write
        principal components will be written in [outfile]_1, [outfile]_2, etc
        if del_temp:
            temp files will be written and then deleted
        
        temp files created:
            temp_datafile_for_R.txt
            temp_R_script.R
            [outfile]_*
            temp_R_output.txt
            temp_R_dev.txt
    """
    try:
        if number_of_prcs == -1:
            number_of_prcs = len(data_in_rows[0])
        number_of_prcs = min(number_of_prcs, len(data_in_rows))
    except Exception as e:
        print e, "data_in_rows is empty"


    # standardizing the data 
    std_data = []
    for a in data_in_rows:
        std_data.append(list(a))

    #av_stds reports all the averages and std deviations of all the len(data_in_rows[0]) variables
    av_stds = standardize_matrix(std_data)

    #write data in format ready for R
    write_file_for_R(std_data, "temp_datafile_for_R.txt")

    # execute R
    write_R_script(outfile, number_of_prcs)
    os.system("R -f temp_R_script.R > temp_R_output.txt")

    # read useful information: components, loadings and std deviations
    """
        components[0] is a list with the variables of principal component 1
        sdev is a list with all the standard devs associated with the principal components
        weights[0] is a list with the loadings of the first row of std_data
    """
    components = read_components(number_of_prcs, outfile)
    sdev = read_data_and_put_in_a_list("temp_R_dev.txt")
    loadings = compute_all_weights(std_data, components)

    # deleting the temp files
    if del_temp:
        os.system("rm " + outfile + "_*")
        os.system("rm temp_datafile_for_R.txt")
        os.system("rm temp_R_script.R")
        os.system("rm temp_R_output.txt")
        os.system("rm temp_R_dev.txt")

    return components, sdev, loadings, av_stds


def write_component_report(ids, data_in_rows, outfolder, overwrite_folder=False):
    """
        INPUT:
        ids is a list of strings to identify the data_rows
        data_in_rows[0] is a list of variables for the first snapshot
        outfolder is a folder which will be delted and created
        
        OUTPUT:
        it's writing a bunch of things in "outfolder/*"
        1. ids-loadings - loadings.txt
        2. prcs.txt
        3. variances along the principal components - components_variances.txt -
        4. ids- accuracy of the reconstruction (from 1 to 6 prcs) - accuracy_1.txt - euclidean and cosine distances (this was commented out)
        5. averages and standard deviations of the variables - av_std.txt -     each variable along a different line    
        RETURNS: 
        principal components and loadings
        
    """

    if overwrite_folder == False and os.path.exists(outfolder):
        print outfolder + " exists. exiting..."
        exit()

    if overwrite_folder and os.path.exists(outfolder):
        print "removing ", outfolder
        os.system("rm -r " + outfolder)

    print "creating ", outfolder
    os.system("mkdir " + outfolder)

    components, sdev, loadings, av_stds = execute_R(data_in_rows)

    # get variances instead of stds
    variances = [v ** 2 for v in sdev]

    # writing loadings
    write_list_of_lists(outfolder + "/loadings.txt", loadings)

    # writing prcs
    write_list_of_lists(outfolder + "/prcs.txt", components)

    # writing standard deviations along the components
    write_list_of_lists(outfolder + "/components_variances.txt", [variances])

    # writing average and standard deviations of the variables
    f = open(outfolder + "/av_stds.txt", "w")
    for l in av_stds:
        f.write(str(l[0]) + " " + str(l[1]) + "\n")

    if False:
        #reconstructing the data
        for approx_num in xrange(1, 7):
            approx_data = compute_approx_data_all(loadings, av_stds, components[:approx_num])
            write_list_of_lists_with_headers(outfolder + "/accuracy_" + str(approx_num) + ".txt", ids, \
                                             euclidean_cosine_distances(approx_data, data_in_rows))

    return components, variances, loadings, av_stds


if __name__ == '__main__':
    '''
        runs a short examples
    '''
    data_in_rows = [[1, 2, 3, 3], [2, 3, 3, 1], [4, 5, 6, 3]]
    ids = ["michael", "donatello", "raphael"]
    print os.getcwd() + "/turtles"
    write_component_report(ids, data_in_rows, os.getcwd() + "/turtles", overwrite_folder=True)


