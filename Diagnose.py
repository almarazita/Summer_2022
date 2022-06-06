# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 19:00:24 2022

@author: almar

Quiz 3 code
"""

def diagnose(result = True, p_sick = 1/100000000, positive = 0.99, falsePositive = 0.02):
    """
    Parameters
    ----------
    result : Boolean result of test for disease.
        Defaults to true.
    p_sick : The chance of having the disease.
        Defaults to one in 100 million.
    positive : The chance of testing positive when you have the disease.
        Defaults to 99%.
    falsePositive : The chance of testing positive when you do not have the disease.
        Defaults to 2%.

    Returns
    -------
    None. Displays the diagnosis that maximizes likelihood
    and the one that maximizes a posteriori (takes prior into account).

    """
    
    p_notSick = 1 - p_sick
    falseNegative = 1 - positive
    negative = 1 - falsePositive
    
    if(result):
        print("Tested positive.\n")
    else:
        print("Tested negative.\n")
    
    print("Using the maximum likelihood (ML) criterion: ")
    if(result):
        if(positive > falsePositive):
            print("You have the disease.")
        else:
            print("You do not have the disease.")
    else:
        if(negative > falseNegative):
            print("You do not have the disease.")
        else:
            print("You have the disease.")
    
    print("Using the maximum a posteriori (MAP) criterion: ")
    if(result):
        chanceSick = positive*p_sick
        chanceNotsick = falsePositive*p_notSick
        if(chanceSick > chanceNotsick):
            print("You have the disease.")
        else:
            print("You do not have the disease.")
    else:
        chanceNotsick = negative*p_notSick
        chanceSick = falseNegative*p_sick
        if(chanceNotsick > chanceSick):
            print("You do not have the disease.")
        else:
            print("You have the disease.")

def main():
    # Example with quiz question info
    diagnose()
    print("\n")
    
    # Diagnosing based on test result and disease stats
    # Values are not checked for legitimacy
    r = int(input("What is the result of your test (0 or 1)? "))
    p = float(input("What is the probability of having this disease? "))
    a = float(input("How accurate is the test (positive when sick)? "))
    fp = float(input("What is the rate of false positives? "))
    print("\n")
    diagnose(r, p, a, fp)

if __name__ == "__main__":
    main()