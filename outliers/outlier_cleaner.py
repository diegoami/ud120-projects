#!/usr/bin/python

import numpy
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """


    #print(perc)
    agarr = ages.transpose().tolist()[0]
    narr = net_worths.transpose().tolist()[0]
    parr = (predictions.transpose() - net_worths.transpose()).tolist()[0]


    cleaned_uns_data = zip(agarr, narr, parr)
    cleaned_s_data = sorted(cleaned_uns_data, key=lambda x: abs(x[2]))
    cleaned_p_data = cleaned_s_data[0:len(cleaned_s_data)*9/10]
    cleaned_data = cleaned_p_data
    print cleaned_data
    return cleaned_data

