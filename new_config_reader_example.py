#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:58:26 2019

@author: mirco
"""

import re
import sys

# This script reads the config file and converts it into a hierarchical python dictionary.

config_file='cfg/TIMIT_baselines/TIMIT_MLP_mfcc_fbank_new.cfg'


# readinng the config file
config_lst = open(config_file, "r")

# initialization of the config dictionary
config={}

# active tags tracking list
active_tags=[]

# search pattern
open_tag='\[.*\]' # open tags (e.g., [exp]) 
close_tag='\[/.*\]' # close tags (e.g, [/exp])
value_pattern='(.*)=(.*)' # field=value lines
begin_block='\{' # this pattern indicates the beginning of a code block
end_block='\}' # this pattern indicatws the end of a code block


# the following varible is 
reading_string_block=False

for line in config_lst:
    
    # removing empty characters
    line=line.strip()
  
    # Detecting open tags [..]
    if bool(re.search(open_tag, line)) and not(bool(re.search(close_tag, line))):
        
        # remove spaces
        active_tags.append(line.replace(' ',''))
        
        # initialize the curr_dict
        curr_dict=config
        
        for tag in active_tags:
            tag=tag.replace('[','')
            tag=tag.replace(']','')
            
            # finding the current position within the dictionary
            if  tag in curr_dict.keys(): 
                curr_dict=curr_dict[tag]
            else:
                # if tag does not exist, create entry
                curr_dict[tag]={}
                curr_dict=curr_dict[tag]
                
   
    # Detecting close tags [/..]
    if bool(re.search(close_tag, line)):
        
        # remove spaces
        closed_tag=line.replace(' ','')
          
        # check if tags are correctly closed
        if closed_tag.replace('[/','[')!=active_tags[-1]:
            sys.stderr.write('ERROR: the tag %s is not closed properly! It should be closed before %s \n'%(active_tags[-1],closed_tag))
            sys.exit(0)
        else:
            # removing from the active tag list the closed element
            del(active_tags[-1])
      
    # check if a block of code has started
    if begin_block in line:
        reading_string_block=True
        block_str=[]
        
        
    # Detecting value lines and adding them into the dictionary
    if bool(re.search(value_pattern, line)) and not(reading_string_block):
         entries=line.split('=')
         field=entries[0].strip()
         value='='.join(entries[1:]).strip()
         curr_dict[field]=value
         
    # check if a block of code is ended    
    if end_block in line:
        reading_string_block=False
        block_str.append(line.replace(end_block,''))
        curr_dict[tag]=list(filter(None, block_str))
        
    # read the lines of the code
    if  reading_string_block:
        block_str.append(line.replace(begin_block,''))
        
# check if all the tags are closed        
if len(active_tags)>0:
   sys.stderr.write('ERROR: the following tags are opened but not closed! %s \n' %(active_tags))
   sys.exit(0)

# closing the config file    
config_lst.close()



def print_dict_rec(dictionary,n_tabs,f):
    # This function reads the config dictionary and convert into into a text form
    
    # managing tabs for better identation
    tabs=''
    for i in range(n_tabs):
        tabs=tabs+'\t'

    # reading all the keys of the dictionary    
    for key in dictionary.keys():
        
        if type(dictionary[key]) is dict:
            f.write('%s[%s]\n' %(tabs,key))
            # if the value is a dictionary I fully explore this branch using a recursive function
            print_dict_rec(dictionary[key],n_tabs+1,f)
            f.write('%s[/%s]\n' %(tabs,key))
        else:
            # if the value is not dictionary, I reached a leaf value
            if type(dictionary[key]) is str:
                f.write('%s%s = %s\n' %(tabs,key,dictionary[key]))
            if type(dictionary[key]) is list:
                for line in dictionary[key]:
                    f.write('%s%s\n'%(tabs,line))
          
            

def dict_to_conf(config,config_file):
    # This function converts a config dictionary into a config file in txt format.
    config_file = open(config_file, 'w')
    print_dict_rec(config,0,config_file)
    config_file.close()



        
        
        
    
    



