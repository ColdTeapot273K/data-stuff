#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    zipper = zip(ages, net_worths, abs(predictions-net_worths))
    cleaned_data = list(zipper)
    cleaned_data.sort(key=lambda x: x[2])

    for i in range(int(len(cleaned_data)*0.10)):
        cleaned_data.pop()
    
    return cleaned_data

