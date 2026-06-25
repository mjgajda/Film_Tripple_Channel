# analysis/roi_analysis.py
import numpy as np
import math




def BatchROIanalysis(OD_pre_vals, OD_post_vals):

   

    numrois = len(OD_post_vals)

    values, uncertainties, pre_val, post_val, IDs = [], [], [], [], []
    # Apply mask to both scans
    def do_roi(OD_pre_cal,  ID):

        # Median + uncertainty for both
        OD_pre_med = np.mean(OD_pre_cal)
        

        OD_pre_std = np.std(OD_pre_cal)
       
        # Compute net OD


        # Record data
        IDs.append(ID)
        pre_val.append(OD_pre_med)
        uncertainties.append(OD_pre_std)            # uncertainty %

    for i in range(len(OD_post_vals)):

        do_roi(OD_pre_vals[i],i+1)
    
    for j in range(len(OD_post_vals)):

        do_roi(OD_post_vals[j], j+ 2 + i)
        


    # Combine results
    return np.column_stack([
         np.array(IDs), 
        np.array(uncertainties), np.array(pre_val)
    ])






