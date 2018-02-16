from __future__ import division, print_function
import numpy as np
import re
from mechanize import Browser
from bs4 import BeautifulSoup

def grab_stilism_data(ell, bee):

    # open the stilism website
    browser = Browser()
    browser.set_handle_robots(False)
    browser.open("http://stilism.obspm.fr")

    # form inputs 
    browser.select_form(nr=0)
    browser['frame'] = ['galactic']
    browser['vlong'] = str(ell)
    browser['vlat'] = str(bee)

    response = browser.submit()
    content = response.read()

    # Parse HTML with beautifulsoup
    soup = BeautifulSoup(content)

    # The table of data has HTML tag 'sql'
    reddening_table = soup.find(id='sql')

    # make it a string so it's hashable
    rtstr = str(reddening_table)

    # remove beginning and ending characters 
    rtstr = rtstr[218:-10]

    # remove line break characters
    rtstr = rtstr.splitlines()

    # put it all into a numpy array
    rtarray = np.zeros((len(rtstr), 4), np.float_)
    lastrowindex = len(rtstr) - 1
    for i, rowstr in enumerate(rtstr):
        if i < lastrowindex:
            reduced_rowstr = rowstr[1:-2]
        
        # last row has no trailing comma
        elif i == lastrowindex:
            reduced_rowstr = rowstr[1:-1]
        
        reduced_rowstr_split = reduced_rowstr.split(',')
    
        for _col in np.arange(4):
            rtarray[i, _col] = np.float(reduced_rowstr_split[_col][1:-1])
        
    distancebins = rtarray[:, 0]
    ebmv = rtarray[:, 1]
    dist_uncertainty = rtarray[:, 2]
    ebmv_uncertainty = rtarray[:, 3]

    return distancebins, ebmv, dist_uncertainty, ebmv_uncertainty

# Load fiber properties from Larry's table
all_prop = np.loadtxt("/Users/susanclark/Downloads/all_prop.txt", skiprows=1, delimiter=',')
all_l = all_prop[:, 2]
all_b = all_prop[:, 3]

# output array will be (l, b, distance, ebmv, dist uncertainty, ebmv uncertainty, lower limit flag)
lbdistances = np.zeros((len(all_l), 7), np.float_)

# find the E(B-V) value closest to 0.015
LB_value = 0.015


for i, (ell, bee) in enumerate(zip(all_l, all_b)):
    
    # record (l, b)
    lbdistances[i, 0] = ell
    lbdistances[i, 1] = bee

    # grab data from stilism website
    distancebins, ebmv, dist_uncertainty, ebmv_uncertainty = grab_stilism_data(ell, bee)

    if np.max(ebmv) < LB_value:
        print("LB wall not reached for l = {}, b = {}".format(ell, bee))
        lbdistances[i, 6] = 1 # Lower limit flag  
        
        lbdistances[i, 2] = distancebins[-1]
        lbdistances[i, 3] = ebmv[-1]
        lbdistances[i, 4] = dist_uncertainty[-1]
        lbdistances[i, 5] = ebmv_uncertainty[-1]
        
    else:
        LBval_indx = np.abs(ebmv - LB_value).argmin()
        lbdistances[i, 2] = distancebins[LBval_indx]
        lbdistances[i, 3] = ebmv[LBval_indx]
        lbdistances[i, 4] = dist_uncertainty[LBval_indx]
        lbdistances[i, 5] = ebmv_uncertainty[LBval_indx]


    