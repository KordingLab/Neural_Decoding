
# coding: utf-8

# In[ ]:

import numpy as np


# In[ ]:

###Function that puts spikes into bins###
#The output is a matrix of size "number of time bins" x "number of neurons"
def bin_spikes(spike_times,dt,wdw_start,wdw_end):
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1
    num_neurons=spike_times.shape[0] #Number of neurons
    neural_data=np.empty([num_bins,num_neurons]) #Initialize array for binned neural data
    #Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        neural_data[:,i]=np.histogram(spike_times[i],edges)[0]
    return neural_data


# In[ ]:

###Function that puts outputs into bins###
#The output is a matrix of size "number of time bins" x "number of features in the output"
def bin_output(outputs,output_times,dt,wdw_start,wdw_end,downsample_factor=1):

    ###Downsample output###
    if downsample_factor!=1:
        downsample_idxs=np.arange(0,output_times.shape[0],downsample_factor)
        outputs=outputs[downsample_idxs,:]
        output_times=output_times[downsample_idxs]

    ###Put outputs into bins###
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1

    # t1=time.time()
    output_dim=outputs.shape[1]
    outputs_binned=np.empty([num_bins,output_dim])
    for i in range(num_bins):
        idxs=np.where((np.squeeze(output_times)>edges[i]) & (np.squeeze(output_times)<edges[i+1]))[0] #Indices to consider the output signal (when it's in the correct time range)
        for j in range(output_dim):
            outputs_binned[i,j]=np.mean(outputs[idxs,j])
    # t2=time.time()-t1
    # t2
    return outputs_binned


# In[ ]:

###Function that creates the covariate matrix of neural activity###
#For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
#The output is a matrix of size "number of total time bins" x "number of time bins used as history" x "number of neurons"
def get_spikes_with_history(neural_data,bins_before,bins_after):
    num_examples=neural_data.shape[0]
    num_neurons=neural_data.shape[1]
    X=np.empty([num_examples,bins_before+bins_after,num_neurons])
    X[:] = np.NaN
    start_idx=0
    for i in range(num_examples-bins_before-bins_after):
        end_idx=start_idx+bins_before+bins_after;
        X[i+bins_before,:,:]=neural_data[start_idx:end_idx,:]
        start_idx=start_idx+1;
    return X
