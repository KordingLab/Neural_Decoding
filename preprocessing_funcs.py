import numpy as np


######## BIN_SPIKES ########
def bin_spikes(spike_times,dt,wdw_start,wdw_end):
    """
    Function that puts spikes into bins

    Parameters
    ----------
    spike_times: an array of arrays
        an array of neurons. within each neuron's array is an array containing all the spike times of that neuron
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for putting spikes in bins
    wdw_end: number (any format)
        the end time for putting spikes in bins

    Returns
    -------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    """
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1
    num_neurons=spike_times.shape[0] #Number of neurons
    neural_data=np.empty([num_bins,num_neurons]) #Initialize array for binned neural data
    #Count number of spikes in each bin for each neuron, and put in array
    for i in range(num_neurons):
        neural_data[:,i]=np.histogram(spike_times[i],edges)[0]
    return neural_data



######## BIN_OUTPUT #######
def bin_output(outputs,output_times,dt,wdw_start,wdw_end,downsample_factor=1):
    """
    Function that puts outputs into bins

    Parameters
    ----------
    outputs: matrix of size "number of times the output was recorded" x "number of features in the output"
        each entry in the matrix is the value of the output feature
    output_times: a vector of size "number of times the output was recorded"
        each entry has the time the output was recorded
    dt: number (any format)
        size of time bins
    wdw_start: number (any format)
        the start time for binning the outputs
    wdw_end: number (any format)
        the end time for binning the outputs
    downsample_factor: integer, optional, default=1
        how much to downsample the outputs prior to binning
        larger values will increase speed, but decrease precision

    Returns
    -------
    outputs_binned: matrix of size "number of time bins" x "number of features in the output"
        the average value of each output feature in every time bin
    """

    ###Downsample output###
    if downsample_factor!=1:
        downsample_idxs=np.arange(0,output_times.shape[0],downsample_factor)
        outputs=outputs[downsample_idxs,:]
        output_times=output_times[downsample_idxs]

    ###Put outputs into bins###
    edges=np.arange(wdw_start,wdw_end,dt) #Get edges of time bins
    num_bins=edges.shape[0]-1


    output_dim=outputs.shape[1]
    outputs_binned=np.empty([num_bins,output_dim])
    for i in range(num_bins):
        idxs=np.where((np.squeeze(output_times)>edges[i]) & (np.squeeze(output_times)<edges[i+1]))[0] #Indices to consider the output signal (when it's in the correct time range)
        for j in range(output_dim):
            outputs_binned[i,j]=np.mean(outputs[idxs,j])

    return outputs_binned


###$$ GET_SPIKES_WITH_HISTORY #####
def get_spikes_with_history(neural_data,bins_before,bins_after,bins_current=1):
    """
    Function that creates the covariate matrix of neural activity

    Parameters
    ----------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    bins_before: integer
        How many bins of neural data prior to the output are used for decoding
    bins_after: integer
        How many bins of neural data after the output are used for decoding
    bins_current: 0 or 1, optional, default=1
        Whether to use the concurrent time bin of neural data for decoding
        
    Returns
    -------
    X: a matrix of size "number of total time bins" x "number of time bins used as history" x "number of neurons"
        For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
    """

    num_examples=neural_data.shape[0]
    num_neurons=neural_data.shape[1]
    total_bins=bins_before+bins_after+bins_current
    X=np.empty([num_examples,total_bins,num_neurons])
    X[:] = np.NaN
    start_idx=0
    for i in range(num_examples-bins_before-bins_after):
        end_idx=start_idx+total_bins;
        X[i+bins_before,:,:]=neural_data[start_idx:end_idx,:]
        start_idx=start_idx+1;
    return X
