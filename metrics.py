import numpy as np

########## VARIANCE ACCOUNTED FOR (VAF) ##########

def get_vaf(y_test,y_test_pred):

    """
    Function to get VAF (variance accounted for)

    Parameters
    ----------
    y_test - the true outputs (a matrix of size number of examples x number of outputs)
    y_test_pred - the predicted outputs (a matrix of size number of examples x number of outputs)

    Returns
    -------
    vaf_list: A list of VAFs for each output
    """

    vaf_list=[] #Initialize a list that will contain the vafs for all the outputs
    for i in range(y_test.shape[1]): #Loop through outputs
        #Compute vaf for each output
        y_mean=np.mean(y_test[:,i])
        vaf=1-np.sum((y_test_pred[:,i]-y_test[:,i])**2)/np.sum((y_test[:,i]-y_mean)**2)
        vaf_list.append(vaf) #Append VAF of this output to the list
    return vaf_list #Return the list of VAFs
