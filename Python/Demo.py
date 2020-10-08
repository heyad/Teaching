#!/usr/bin/env python
# coding: utf-8

# In[6]:

### To run this file, simply save it locally, and then from the directory 
### type python Demo.py
### Run the above several times with different number of arguments and check the output
### For example Demo.py Arg1 Arg2

import sys
 
# Get the total number of argumentas passed to the Demo.py
total = len(sys.argv)
 
# Get the arguments list 
cmdargs = str(sys.argv)
 
# Print it
print ("The total numbers of args passed to the script: %d " % total)
print ("Args list: %s " % cmdargs)

