##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import numpy as np
import sys
from utils import compute_cw_max,dict_fea_lab_arch,is_sequential_dict
import os
import configparser
import re, gzip, struct
import threading
import glob
import math
import random 
import torch

def LoadData(cfg_file,lab_all,shared_list):

        
        # Reading chunk-specific cfg file (first argument-mandatory file) 
        if not(os.path.exists(cfg_file)):
             sys.stderr.write('ERROR: The config file %s does not exist!\n'%(cfg_file))
             sys.exit(0)
        else:
            config = configparser.ConfigParser()
            config.read(cfg_file)
            
        
        # Reading some cfg parameters
        output_folder=config['exp']['out_folder']
        dataset=config['exp']['dataset']
        seed=int(config['exp']['seed'])
        batch_size=int(config['batches']['batch_size'])
        
        max_seq_length=int(config['batches']['max_seq_length']) # Read the maximum length allowed for the speech sequence
                
        # Putting information about features, labels, and architectures on dictionaries
        # this information will be later useful to initialize the model and forward the data through the model
        
        [fea_info,lab_info,arch_info]=dict_fea_lab_arch(config)
        
        
        # Check if the model is recurrent (sequential) or non-recurrent (feed-forward)
        rnn_model=is_sequential_dict(config,arch_info)


        # Detecting the maximum length of the specified context windows (useful to male different features of the same length)
        [cw_left_max,cw_right_max]=compute_cw_max(fea_info)

        
        # Loading all the features in parallel (stored in the fea_data_dict[fea_type][snt_id] dictionary)
        fea_data_dict={}
        
        #create a list of threads
        threads = []
        
        for fea in fea_info.keys():
            fea_scp=fea_info[fea][1]
            fea_opts=fea_info[fea][2]
            fea_type=detect_fea_type(fea_scp)            
            
            p=threading.Thread(target=load_features, args=(fea_type,fea_scp,fea_opts,output_folder,fea,fea_data_dict))
            threads.append(p)
            p.start()

            
        # Wait until all processes are finished 
        for p in threads:
            p.join()
            
            
        # Check features (check if all the feature types have the sentence-ids and the same length)
        fea_data_dict=check_features(fea_data_dict,output_folder)

        # Cross-check between features and labels (check if labels and features contain the same sentence-ids and are of the same length)
        common_set=cross_check_lab_fea(fea_data_dict,lab_all,dataset,output_folder)
        
        # select only the subset of features that have a corresponding label
        fea_data_dict=select_fea(fea_data_dict,common_set)
        
        # select only the subset of labels of the current chunk
        lab_chunk=select_lab(lab_all,dataset,common_set)
        
        # compute mean and standard deviation statistics
        [mean_dict,std_dict]=compute_mean_std(fea_data_dict)
                       
        # Split the sequences that are longer than the specified max_seq_length 
        fea_data_dict=split_fea(fea_data_dict,max_seq_length)
        lab_chunk=split_lab(lab_chunk,max_seq_length)
        
        # Expand the sentences with a context window (if needed)
        fea_data_dict=apply_context_window(fea_data_dict,fea_info)
        
        # Check again that all the features have same temporal dimension
        double_check_fea_lab(fea_data_dict,lab_chunk)
        
        # Convert the features to pytorch
        fea_data_dict=fea_to_pytorch(fea_data_dict)
        
        
        # Add final dimensionality for each feature type into fea_info dict
        for fea_type in fea_data_dict.keys():
            snt_id=list(fea_data_dict[fea_type].keys())[0]
            fea_info[fea_type].append(fea_data_dict[fea_type][snt_id].shape[1])
        
        # Order the list of sentence (ascending, discending, shuffle)
        sentence_order='shuffle'
        snt_lst=sentence_ordering(fea_data_dict,sentence_order,seed)
                          
        # Create batches (the creation of batches is different between recurrent and non-recurrent networks)
        if not(rnn_model):   
            # Create batches for non-sequential neural network (i.e., feed-forward)
            [fea_batches_dict,lab_batches_dict]=create_batches_feed_forward(fea_info,batch_size,fea_data_dict,lab_chunk,snt_lst,sentence_order,seed)
        else:
            # Create batches for recurrent neural networks
            [fea_batches_dict,lab_batches_dict]=create_batches_recurrent(fea_info,batch_size,fea_data_dict,lab_chunk,snt_lst,seed)
        
        
        # The generated dictionaries are structured as follows:             
        # fea_batches_dict[batch_id][fea_id][sample_id]
        # lab_batches_dict[batch_id][label_id][sample_id]

        
        shared_list.append(fea_batches_dict)
        shared_list.append(fea_info)
        shared_list.append(lab_batches_dict)
        shared_list.append(lab_info)
        shared_list.append(arch_info)

                 
        # To do: try DNN experiment and check performance, try RNN experiment and check performance, CMVN options


                    
        
        
def preload_labels(config):
    
    # This function loads all the labels in RAM once. 
    # This way the labels don’t have to be read from disk every time. 
    # We do it with labels only because features can be much more memory demanding and might not fit into the memory.
    
    data_dict=create_data_dict(config)
    tr_data_lst=config['data_use']['train_with'].split(',')
    valid_data_lst=config['data_use']['valid_with'].split(',')
    output_folder=config['exp']['out_folder']
    

    # Reading labels in parallel (one thread for each ali.*.tar.gz file)
    lab_all={}
    for data in tr_data_lst+valid_data_lst:
        lab_all[data]={}
        
        for key in data_dict[data]['lab'].keys():
            lab_folder=data_dict[data]['lab'][key][0].split('lab_folder=')[1]
            
            # Check if folder exists
            if not(os.path.isdir(lab_folder)):
                sys.stderr.write('ERROR the lab_folder %s does not exist\n' %(lab_folder)) 
                sys.exit(0)
                
            lab_type=detect_lab_type(lab_folder)
            lab_opts=data_dict[data]['lab'][key][1].split('lab_opts=')[1]
            ali_list=glob.glob(lab_folder+'/ali.*.gz')
            
            # Check if alignments exists
            if len(ali_list)==0:
                sys.stderr.write('ERROR the lab_folder %s does not contain alignments (*ali.*.gz)\n' %(lab_folder)) 
                sys.exit(0)
                
            lab_parallel ={}
            lab_all[data][key]={}
            
            threads = []
            
            for ali_file in ali_list:
                ali_index=ali_file.split('ali.')[1].split('.gz')[0]
                p=threading.Thread(target=load_labels, args=(lab_type,ali_file, lab_opts, lab_folder, output_folder, lab_parallel ,ali_index))            
                threads.append(p)
                p.start()
                
            # Wait until all processes are finished 
            for p in threads:
                p.join()
                
            # gather all the results into a single dictionary     
            for ali_id in lab_parallel.keys():
               lab_all[data][key].update(lab_parallel[ali_id]) 
               
    # check labels
    lab_all=check_labels(lab_all,output_folder)
    
    # convert labels to pytorch
    lab_all=lab_to_pytorch(lab_all)

         
    

    
    return lab_all


def load_labels(lab_type,ali_file, lab_opts, lab_folder, output_folder,lab_shared,ali_index):
    if lab_type=='kaldi':
        lab_shared[ali_index] = { k:v for k,v in read_vec_int_ark('gunzip -c '+ali_file+' | '+lab_opts+' '+lab_folder+'/final.mdl ark:- ark:-|',output_folder)}


def load_features(fea_type,fea_scp,fea_opts,output_folder,fea_name,dict_fea):
  if fea_type=='kaldi':
      dict_fea[fea_name] = { k:m for k,m in read_mat_ark('ark:copy-feats scp:'+fea_scp+' ark:- |'+fea_opts,output_folder) }

def apply_context_window(fea_data_dict,fea_dict):
    
    # This function loops over all the sentences and features and apply the context windows specified in the cfg file. 
    # At the boundaries of the sentences, zero padding has been performed.   
  
    for fea_type in fea_data_dict.keys():
        cw_left=int(fea_dict[fea_type][3])
        cw_right=int(fea_dict[fea_type][4])

        if  cw_left>0 or cw_right>0:
            for snt_id in  fea_data_dict[fea_type].keys():
                N_fea=fea_data_dict[fea_type][snt_id].shape[0]
                fea_dim=fea_data_dict[fea_type][snt_id].shape[1]
                
                fea_expanded=np.zeros([N_fea,fea_dim,cw_left+cw_right+1])
                index_cnt=0
                for lag in range(-cw_left,cw_right+1):
                    fea_expanded[:,:,index_cnt]=np.roll(fea_data_dict[fea_type][snt_id],-lag,axis=0)
                    if lag<0:
                       fea_expanded[0:-lag,:,index_cnt]=np.zeros(fea_expanded[0:-lag,:,index_cnt].shape) 
                    
                    if lag>0:
                       fea_expanded[-lag:,:,index_cnt]=np.zeros(fea_expanded[-lag:,:,index_cnt].shape)
                    
                    index_cnt=index_cnt+1
                # reshape tensor (put all the context features in the same feature vector)    
                fea_data_dict[fea_type][snt_id]=fea_expanded.reshape(fea_expanded.shape[0],-1)

                
    return fea_data_dict

def create_batches_feed_forward(fea_dict,batch_size,fea_data_dict,lab_chunk,snt_lst,sentence_order,seed):
    
    # This function creates the batches when a feed-forward neural network is used. 
    # Batch-specific dictionaries are formed for both features and labels.  
    # For instance,  fea_batches_dict[‘batch_11’][‘mfcc’] contains a numpy matrix of dimension (N_batch, fea_dim). 
    # Similarly,  lab_batches_dict[‘batch_11’][‘lab_cd’] contains the labels for batch 11. 
    # Create list of all the feature frames (useful for shuffling)
    
    # Initializing the batch dictionaries
    fea_batches_dict={}
    lab_batches_dict={}
    
    # Create a list of all the features in the chunk
    fea_all_lst=create_all_fea_lst(fea_data_dict,snt_lst)

    # Shuffle the list (if needed)
    if sentence_order=='shuffle':
        random.seed(seed)
        random.shuffle(fea_all_lst)
        
    # Split the list features in list of batches
    fea_all_lst=list(split_list(fea_all_lst,batch_size))
    
    # Removing the last batch because can contain only few elements
    if len(fea_all_lst[-1])<batch_size:
        del (fea_all_lst[-1])
            
    # Compute feature/label dimensionalities
    fea_dim=compute_dim(fea_data_dict)
    lab_dim=compute_dim(lab_chunk)
        
    N_batches=len(fea_all_lst)
    
    
    for batch_id in range(N_batches):
         key_batch='batch_'+str(batch_id)
         
         # Initializing batch-specific dictionaries
         fea_batches_dict[key_batch]={}
         lab_batches_dict[key_batch]={}
         
         for fea in fea_dict.keys():
             fea_batches_dict[key_batch][fea]=torch.zeros([batch_size,fea_dim[fea]])
         
         for lab in lab_chunk.keys():
             lab_batches_dict[key_batch][lab]=torch.zeros([batch_size,lab_dim[lab]])
         
         # Fill all the batch-specific dictionaries with the selected features and labels
         for sample in range(batch_size):
             
            # Detecting sample_id and frame_id from fea_all_lst
            [sample_id,frame_id]=fea_all_lst[batch_id][sample].split(':Frame_')
            frame_id=int(frame_id)
            
            # filling the batch-specific dictionary
            for fea in fea_dict.keys():
                fea_batches_dict[key_batch][fea][sample]=fea_data_dict[fea][sample_id][frame_id]
                
            for lab in lab_chunk.keys():
                lab_batches_dict[key_batch][lab][sample]=lab_chunk[lab][sample_id][frame_id]
                
    return [fea_batches_dict,lab_batches_dict]


def create_batches_recurrent(fea_dict,batch_size,fea_data_dict,lab_chunk,snt_lst,seed):
    
    # This function creates the batches when a recurrent neural network is used. 
    # Batch-specific dictionaries are formed for both features and labels.  
    # For instance,  fea_batches_dict[‘batch_11’][‘mfcc’][0] contains a numpy matrix of dimension (N_time_steps, fea_dim) that refers to the first sample of mfcc features of the batch 11
    # Similarly,  lab_batches_dict[‘batch_11’][‘lab_cd’][0] contains the labels for batch 11-sample 0. 
    
    # Initializing the batch dictionaries
    fea_batches_dict={}
    lab_batches_dict={}
    
    # Split the sentence list in N_batch sub-lists
    snt_lst=list(split_list(snt_lst,batch_size))
    
    # Removing the last batch because can contain only few elements
    if len(snt_lst[-1])<batch_size:
        del (snt_lst[-1])
        
    N_batches=len(snt_lst)
    
    
    for batch_id in range(N_batches):
         key_batch='batch_'+str(batch_id)
         
         # Initializing batch-specific dictionaries
         fea_batches_dict[key_batch]={}
         lab_batches_dict[key_batch]={}
         
         for fea in fea_dict.keys():
             fea_batches_dict[key_batch][fea]=[]
         
         for lab in lab_chunk.keys():
             lab_batches_dict[key_batch][lab]=[]
         
         # Fill all the batch-specific dictionaries with the selected features and labels
         for snt_id in snt_lst[batch_id]:
            
            # filling the batch-specific dictionary
            for fea in fea_dict.keys():
                fea_batches_dict[key_batch][fea].append(fea_data_dict[fea][snt_id])
                
            for lab in lab_chunk.keys():
                lab_batches_dict[key_batch][lab].append(lab_chunk[lab][snt_id])
                
    return [fea_batches_dict,lab_batches_dict]

def sentence_ordering(fea_data_dict,sentence_order,seed):
    
    # This function sorts the sentence of the chunks as specified in the config file. 
    # If the specified order is ascending, the sentences are sorted from the shortest to the longest one. 
    # If the order is descending, the sentences are sorted from the longest to the shortest. 
    # Finally, if the  order is set as “shuffle”, the sentences are randomized. 
    
    fea_lst=list(fea_data_dict.keys())
    
    if sentence_order=='ascending':
        snt_lst=sorted(sorted(fea_data_dict[fea_lst[0]].keys()), key=lambda k: len(fea_data_dict[fea_lst[0]][k]))
    
    if sentence_order=='descending':
        snt_lst=sorted(sorted(fea_data_dict[fea_lst[0]].keys()), key=lambda k: -len(fea_data_dict[fea_lst[0]][k]))
    
    if sentence_order=='shuffle':
        snt_lst=list(fea_data_dict[fea_lst[0]].keys())
        random.seed(seed)
        random.shuffle(snt_lst)
        
    return snt_lst 
                
                
def check_features(fea_data_dict,output_folder):
        # check if all the feature types have the sentence-ids and the same length
        
        diff_th_error=0.9 # Rise an error if the features types differs too much
        
        fea_keys=list(fea_data_dict.keys())
        
        for i in range(len(fea_keys)):
            fea_key=fea_keys[i]
            current_set=set(fea_data_dict[fea_key])
            if i==0:
                global_set=current_set
                diff_set=set(global_set-current_set)
            else:
                diff_set=set(global_set-current_set)
                global_set=global_set.intersection(current_set)
              
            
            if len(diff_set)>0:
                with open(output_folder+'/log.log', 'a+') as logfile:
                    for line in diff_set:
                        logfile.write('WARNING: No Features for '+ line+'\n')          
                
            # Rise an error if the featues set are too different    
            if len(global_set)<int(len(current_set)*diff_th_error):
                sys.stderr.write('ERROR: The given features differ too much. Please, take a look into the provided labels and make sure they have the same sentence_id. See the file %s\n' %(output_folder+'/log.log')) 
                sys.exit(0)

        # Select the subset of common labels
        fea_data_dict_sel={}
        for key in fea_data_dict.keys():
            fea_data_dict_sel[key] = { k:v for k, v in  fea_data_dict[key].items() if k in list(global_set)}
        
        fea_data_dict=fea_data_dict_sel
              
        
        # check if all the features types have the same length
        for snt_id in list(global_set):
            fea_type_key=list(fea_data_dict.keys())
            for i in range(len(fea_type_key)):
                if i==0:
                    ref= fea_data_dict[fea_type_key[i]][snt_id].shape[0]
                else:
                    val= fea_data_dict[fea_type_key[i]][snt_id].shape[0]
                    if val!=ref:
                       sys.stderr.write('ERROR: the sentence %s has different feature lengths between %s and %s. All the labels must have the same length.\n' %(snt_id,fea_type_key[i],fea_type_key[0])) 
                       sys.exit(0)
                       
        return fea_data_dict


def cross_check_lab_fea(fea_data_dict,lab_all,dataset,output_folder):
    
    fea_check=fea_data_dict[list(fea_data_dict.keys())[0]]
    lab_check=lab_all[dataset][list(lab_all[dataset].keys())[0]]
    
    # derive the common set of features and labels
    common_set=set(fea_check).intersection(set(lab_check))
    
    # derive the differences between features and labels        
    diff_set_fea=set(set(fea_check)-common_set)
    
    # Writing the differences on file 
    if len(diff_set_fea)>0:
        with open(output_folder+'/log.log', 'a+') as logfile:
            for line in diff_set_fea:
                logfile.write('WARNING: No labels for '+ line+'\n')
     
    diff_th_error=0.05
    # Rise an error if the labels set are too different    
    if len(common_set)<len(fea_check.keys())*(1-diff_th_error):
            sys.stderr.write('ERROR: The given labels differ too much. Please, take a look into the provided labels and make sure they have the same sentence_ids. See the file %s\n' %(output_folder+'/log.log')) 
            sys.exit(0) 
            
    return common_set
      
def check_labels(lab_data_dict,output_folder):
    
    # This function does some checks on the labels. 
    # A warning is raised in the log file if some labels are missing. 
    # If too many labels are missing (e.g., 5%) an error is raised.
    # Only the common set of labels is selected. 
    # The function also check if all the labels have the same length.
        
    diff_th_error=0.05 
    lab_data_dict_sel={}
    
    for dataset in lab_data_dict.keys():
        lab_keys=list(lab_data_dict[dataset].keys())
        
        current_set_len_sum=0
        for i in range(len(lab_keys)):
            lab_key=lab_keys[i]
            current_set=set(lab_data_dict[dataset][lab_key])
            current_set_len_sum=current_set_len_sum=current_set_len_sum+len(current_set)
            
            if i==0:
                global_set=current_set
                diff_set=set(global_set-current_set)
            else:
                diff_set=set(global_set-current_set)
                global_set=global_set.intersection(current_set)
              
            
            if len(diff_set)>0:
                with open(output_folder+'/log.log', 'a+') as logfile:
                    for line in diff_set:
                        logfile.write('WARNING: No labels for '+ line+'\n')
            

        mean_set_len=int(current_set_len_sum/len(lab_keys))

        # Rise an error if the labels set are too different    
        if len(global_set)< mean_set_len*(1-diff_th_error):
            sys.stderr.write('ERROR: The given labels differ too much. Please, take a look into the provided labels and make sure they have the same sentence_ids. See the file %s\n' %(output_folder+'/log.log')) 
            sys.exit(0)
   
        
        # Select the subset of common labels
        lab_data_dict_sel[dataset]={}
        for key in lab_data_dict[dataset].keys():
            lab_data_dict_sel[dataset][key] = { k:v for k, v in  lab_data_dict[dataset][key].items() if k in list(global_set)}
        
    
        # check if all the label types have the same length
        for snt_id in list(global_set):
            lab_type_key=list(lab_data_dict_sel[dataset].keys())
            for i in range(len(lab_type_key)):
                if i==0:
                    ref= lab_data_dict_sel[dataset][lab_type_key[i]][snt_id].shape[0]
                else:
                    val= lab_data_dict_sel[dataset][lab_type_key[i]][snt_id].shape[0]
                    if val!=ref:
                       sys.stderr.write('ERROR: the sentence %s has different label lengths between %s and %s. All the labels must have the same length.\n' %(snt_id,lab_type_key[i],lab_type_key[0])) 
                       sys.exit(0)
                   
    return lab_data_dict_sel

def select_fea(fea_data_dict,common_set):  
    # This function selects only the subset of features that have the corresponding label
    
    for fea_key in fea_data_dict.keys():
        snt_lst=list(fea_data_dict[fea_key].keys())
        for snt_id in snt_lst.copy():
            if snt_id not in common_set:
                print(snt_id)
                del(fea_data_dict[fea_key][snt_id ])   
    return fea_data_dict

def select_lab(lab_all,dataset,common_set): 
    # This function selects only the subset of labels that are actually used in this chunk 
    
    lab_chunk={}            
    for lab_key in lab_all[dataset].keys():
        lab_chunk[lab_key]={}
        for snt_id in lab_all[dataset][lab_key].keys():
            if snt_id in common_set:
                lab_chunk[lab_key][snt_id]=lab_all[dataset][lab_key][snt_id] 
    return lab_chunk

def split_fea(fea_data_dict,max_seq_length):
    
    # Split the features that are longer than Max_length 
    for fea in fea_data_dict.keys():   
    
      snt_ids=list(fea_data_dict[fea].keys())
      
      for k in snt_ids.copy():
          if fea_data_dict[fea][k].shape[0]> max_seq_length:
              N_split=math.ceil(fea_data_dict[fea][k].shape[0]/max_seq_length)
              data_split=np.array_split(fea_data_dict[fea][k],N_split)
              data_split_name=[k+'_split'+str(i) for i in range(N_split)]
              del fea_data_dict[fea][k]
              fea_data_dict[fea].update(zip(data_split_name,data_split))
              
    return fea_data_dict

def split_lab(lab_chunk,max_seq_length):
    
    # Split the labels that are longer than Max_length 
    for lab_type in lab_chunk.keys():
        snt_ids=list(lab_chunk[lab_type].keys())
        for k in snt_ids.copy():
            if lab_chunk[lab_type][k].shape[0]> max_seq_length:
                N_split=math.ceil(lab_chunk[lab_type][k].shape[0]/max_seq_length)
                lab_split=np.array_split(lab_chunk[lab_type][k],N_split)
                lab_split_name=[k+'_split'+str(i) for i in range(N_split)]
                del lab_chunk[lab_type][k]
                lab_chunk[lab_type].update(zip(lab_split_name,lab_split))
                
    return lab_chunk

def split_list(l,n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def compute_dim(data_dict):
    dim_dict={}
    for key in data_dict.keys():
        snt_id=list(data_dict[key].keys())
        shape_dim=data_dict[key][snt_id[0]].shape
        if len(shape_dim) > 1:
            dim=shape_dim[-1]
        else:
            dim=1
        
        dim_dict[key]=dim
    return dim_dict
            
            
def create_all_fea_lst(fea_data_dict,snt_lst):
    
    # This function creates a list containing all the features in the chuck. 
    # The output is a list containing elements formatted in the following way: “fea_id_time_step_id”. 
    # We can shuffle the dataset by simply shuffling this list.
    
    list_all_fea=[]
    fea_id=list(fea_data_dict.keys())[0]

    for snt_id in snt_lst:
        N_time_steps=fea_data_dict[fea_id][snt_id].shape[0]
        for time_step in range(N_time_steps):
            list_all_fea.append(snt_id+':Frame_'+str(time_step))  
    return list_all_fea

def compute_mean_std(fea_data_dict):
    #This function computes for each sentence the mean and the standard deviation. 
    #The global mean and standard deviations are stored in the key “global”.
    
    mean_dict={}
    std_dict={}
    
    for fea_id in fea_data_dict.keys():
        mean_dict[fea_id]={}
        std_dict[fea_id]={}
        count=0
        for snt_id in fea_data_dict[fea_id].keys():
            mean_dict[fea_id][snt_id]=np.mean(fea_data_dict[fea_id][snt_id],axis=0)
            std_dict[fea_id][snt_id]=np.std(fea_data_dict[fea_id][snt_id],axis=0)
            if count==0:
                std_dict[fea_id]['global']=std_dict[fea_id][snt_id]
                mean_dict[fea_id]['global']=mean_dict[fea_id][snt_id]
            else:
                std_dict[fea_id]['global']=std_dict[fea_id]['global']+std_dict[fea_id][snt_id]
                mean_dict[fea_id]['global']=mean_dict[fea_id]['global']+mean_dict[fea_id][snt_id]
                
            count=count+1
        std_dict[fea_id]['global']=std_dict[fea_id]['global']/len(list(fea_data_dict[fea_id].keys()))
        mean_dict[fea_id]['global']=mean_dict[fea_id]['global']/len(list(fea_data_dict[fea_id].keys()))
        
    return [mean_dict,std_dict]



def apply_cmnv(fea_data_dict,mean_dict,std_dict):
    
    #This function performs mean and variance normalization using the statistics previously computed.   
    for fea_id in fea_data_dict.keys():
        for snt_id in fea_data_dict[fea_id].keys():
            fea_data_dict[fea_id][snt_id]=(fea_data_dict[fea_id][snt_id]-mean_dict[fea_id]['global'])/std_dict[fea_id]['global']                  
    
    return fea_data_dict


def double_check_fea_lab(fea_data_dict,lab_chunk):
                         
    # This function double checks that all the features have the same number of frames.
    
    fea_ids=list(fea_data_dict.keys())
    lab_ids=list(lab_chunk.keys())
    
    for snt_id in fea_data_dict[fea_ids[0]].keys():
        count=0

        for fea_id in fea_ids:
             len_snt=fea_data_dict[fea_id][snt_id].shape[0]
             if count==0:
                 len_snt_prev=len_snt
                 count=1
             else:
                 if len_snt_prev!=len_snt:
                      sys.stderr.write('ERROR the sentence %s has different length between %s and %s\n' %(snt_id,fea_ids[0],fea_id)) 
                      sys.exit(0)                         
             
        for lab_id in lab_ids:
            len_snt=lab_chunk[lab_id][snt_id].shape[0]
            if len_snt_prev!=len_snt:
                      sys.stderr.write('ERROR the sentence %s has different length between %s and %s\n' %(snt_id,fea_ids[0],lab_id)) 
                      sys.exit(0)  
                      
                      
def detect_lab_type(lab_folder):
      
      lab_type='not_founded'
      
      if len(lab_folder.split('.'))==2:
          
          if (lab_folder.split('.')[1])=='npy':
              # check if the file exsits:
              if os.path.isfile(lab_folder):
                  lab_type='dict'
              else:
                 sys.stderr.write('ERROR: The file %s does not exists.\n'%(lab_folder))
                 sys.exit(0)   
          
          if (lab_folder.split('.')[1])=='lst':
              # check if the file exsits:
              if not(os.path.isfile(lab_folder)):
                 sys.stderr.write('ERROR: The file %s does not exists.\n'%(lab_folder))
                 sys.exit(0)   
             
              else: 
                with open(lab_folder) as f:
                    content = f.readlines()
                if len(content)==sum(1 for i in content if (len(i.split(' '))==2 and i.strip().split(' ')[1].isdigit())):
                  lab_type='lab_lst'
                else:
                    sys.stderr.write('ERROR: The file %s must only contains lines formatted in the following way “sentence_id label”. The label must be an integer number. Please make sure you only have integer numbers in the file. Make also sure no additional spaces are placed after the label or between the sentence-id and the label. Moreover, delete any empty line in the file  if present (including the last one).\n'%(lab_folder))
                    sys.exit(0)
                    
      else:
          
          if lab_folder!='none':
              if not(os.path.isdir(lab_folder)):
                sys.stderr.write('ERROR: the Kaldi label folder %s does not exist! \n'%(lab_folder))
                sys.exit(0)
              else:
                  # check if model final.mld is present
                  if not(os.path.isfile(lab_folder+'/final.mdl')):
                      sys.stderr.write('ERROR: the file final.mdl is not present in %s \n'%(lab_folder))
                      sys.exit(0)  
                      
                  if not(os.path.isfile(lab_folder+'/ali.1.gz')):
                      sys.stderr.write('ERROR: alignments are not present in %s. You should have ali.*.gz files \n'%(lab_folder))
                      sys.exit(0) 
                      
                  lab_type='kaldi'
          else:
               lab_type='none' 
    
    
              
      if lab_type=='not_founded':
           sys.stderr.write('ERROR: Not able to detect the right type of labels. Possibilities are “kaldi”, “lab_lst”, and “dict”, and "none". If you would like to use lab_lst the file must have ".lst" extension and must contain "sentence_id label" entries. The dict, instead, should have a npy extension. Use none is you do not have the labels (e.g, for production/testing purposes). Please check %s\n'%(lab_folder))
           sys.exit(0)
           
      return lab_type          
    

def detect_fea_type(fea_scp):
    return 'kaldi'

def create_data_dict(config):
    #This function summarizes the features and labels of each used dataset into a dictionary.
    
    data_dict={}
    for sec in  config.sections():
        if "dataset" in sec:
            data_name=config[sec]['data_name']
            data_dict[data_name]={}
            data_dict[data_name]['fea']={}
            data_dict[data_name]['lab']={}
            
    
            fea_lst=list(filter(None, config[sec]['fea'].split('fea_name=')))
            
            for fea in fea_lst:
                list_fea=list(filter(None,fea.split('\n')))
                fea_name=list_fea[0]
                if fea_name  in config['model']['model']: # Select only features actually used in [Model]
                    data_dict[data_name]['fea'][fea_name]=list_fea[1:]
             
            if data_name in config['data_use']['train_with'] or data_name in config['data_use']['valid_with']:
                
                lab_lst=list(filter(None, config[sec]['lab'].split('lab_name=')))
                
                for lab in lab_lst:
                    list_lab=list(filter(None,lab.split('\n')))
                    lab_name=list_lab[0]
                    if lab_name  in config['model']['model']: # Select only labels actually used in [Model]
                        data_dict[data_name]['lab'][lab_name]=list_lab[1:]
                    
    return data_dict

def check_if_fea_exist(config):
    
    #This function checks if all the specified features exist. 
    #It raises an error if at least one feature is not found.

    data_dict=create_data_dict(config)
    for dataset in data_dict.keys():
        for fea_name in data_dict[dataset]['fea'].keys():
            fea_scp=data_dict[dataset]['fea'][fea_name][0].split('=')[1]
            fea_paths=[]
            for fea_line in open(fea_scp):
                fea_paths.append(fea_line.split(' ')[1].split(':')[0])

            for fea_path in set(fea_paths):
                if not(os.path.isfile(fea_path)):
                    sys.stderr.write('ERROR file %s in %s does not exist\n' %(fea_path,fea_scp)) 
                    sys.exit(0)
                     
                    
                
def load_counts(class_counts_file):
    with open(class_counts_file) as f:
        row = next(f).strip().strip('[]').strip()
        counts = np.array([ np.float32(v) for v in row.split() ])
    return counts 


def lab_to_pytorch(lab_all):
    # This function converts all the entries of the label dictionary from numpy to pytorch tensors.
    for data_id in lab_all.keys():
        for lab_id in lab_all[data_id].keys():
            for snt_id in lab_all[data_id][lab_id].keys():
                lab_all[data_id][lab_id][snt_id]=torch.from_numpy(lab_all[data_id][lab_id][snt_id].copy())
    return lab_all

def fea_to_pytorch(fea_data_dict):
    # This function converts all features from numpy to pytorch tensors.
    for fea_id in fea_data_dict.keys():
        for snt_id in fea_data_dict[fea_id].keys():
            fea_data_dict[fea_id][snt_id]=torch.from_numpy(fea_data_dict[fea_id][snt_id])
    return fea_data_dict 
        

def fea_to_cuda(features):
    # This function converts all features to cuda
    for fea_id in features.keys():
        features[fea_id]=features[fea_id].cuda()
        
    return features

# The following libraries are copied from kaldi-io-for-python project (https://github.com/vesis84/kaldi-io-for-python)
    
# Copyright 2014-2016  Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")
    
#################################################
# Define all custom exceptions,
class UnsupportedDataType(Exception): pass
class UnknownVectorHeader(Exception): pass
class UnknownMatrixHeader(Exception): pass

class BadSampleSize(Exception): pass
class BadInputFormat(Exception): pass

class SubprocessFailed(Exception): pass

#################################################
# Data-type independent helper functions,

def open_or_fd(file, output_folder,mode='rb'):
  """ fd = open_or_fd(file)
   Open file, gzipped file, pipe, or forward the file-descriptor.
   Eventually seeks in the 'file' argument contains ':offset' suffix.
  """
  offset = None

  try:
    # strip 'ark:' prefix from r{x,w}filename (optional),
    if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
      (prefix,file) = file.split(':',1)
    # separate offset from filename (optional),
    if re.search(':[0-9]+$', file):
      (file,offset) = file.rsplit(':',1)
    # input pipe?
    if file[-1] == '|':
      fd = popen(file[:-1], output_folder,'rb') # custom,
    # output pipe?
    elif file[0] == '|':
      fd = popen(file[1:], output_folder,'wb') # custom,
    # is it gzipped?
    elif file.split('.')[-1] == 'gz':
      fd = gzip.open(file, mode)
    # a normal file...
    else:
      fd = open(file, mode)
  except TypeError:
    # 'file' is opened file descriptor,
    fd = file
  # Eventually seek to offset,
  if offset != None: fd.seek(int(offset))
  
  return fd

# based on '/usr/local/lib/python3.4/os.py'
def popen(cmd, output_folder,mode="rb"):
  if not isinstance(cmd, str):
    raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))

  import subprocess, io, threading

  # cleanup function for subprocesses,
  def cleanup(proc, cmd):
    ret = proc.wait()
    if ret > 0:
      raise SubprocessFailed('cmd %s returned %d !' % (cmd,ret))
    return

  # text-mode,
  if mode == "r":
    err=open(output_folder+'/log.log',"a")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=err)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return io.TextIOWrapper(proc.stdout)
  elif mode == "w":
    err=open(output_folder+'/log.log',"a")
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,stderr=err)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return io.TextIOWrapper(proc.stdin)
  # binary,
  elif mode == "rb":
    err=open(output_folder+'/log.log',"a")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=err)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return proc.stdout
  elif mode == "wb":
    err=open(output_folder+'/log.log',"a")
    proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE,stderr=err)
    threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
    return proc.stdin
  # sanity,
  else:
    raise ValueError("invalid mode %s" % mode)


def read_key(fd):
  """ [key] = read_key(fd)
   Read the utterance-key from the opened ark/stream descriptor 'fd'.
  """
  key = ''
  while 1:
    char = fd.read(1).decode("latin1")
    if char == '' : break
    if char == ' ' : break
    key += char
  key = key.strip()
  if key == '': return None # end of file,
  assert(re.match('^\S+$',key) != None) # check format (no whitespace!)
  return key


#################################################
# Integer vectors (alignments, ...),

def read_ali_ark(file_or_fd,output_folder):
  """ Alias to 'read_vec_int_ark()' """
  return read_vec_int_ark(file_or_fd,output_folder)

def read_vec_int_ark(file_or_fd,output_folder):
  """ generator(key,vec) = read_vec_int_ark(file_or_fd)
   Create generator of (key,vector<int>) tuples, which reads from the ark file/stream.
   file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

   Read ark to a 'dictionary':
   d = { u:d for u,d in kaldi_io.read_vec_int_ark(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    key = read_key(fd)
    while key:
      ali = read_vec_int(fd,output_folder)
      yield key, ali
      key = read_key(fd)
  finally:
    if fd is not file_or_fd: fd.close()

def read_vec_int(file_or_fd,output_folder):
  """ [int-vec] = read_vec_int(file_or_fd)
   Read kaldi integer vector, ascii or binary input,
  """
  fd = open_or_fd(file_or_fd,output_folder)
  binary = fd.read(2).decode()
  if binary == '\0B': # binary flag
    assert(fd.read(1).decode() == '\4'); # int-size
    vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # vector dim
    if vec_size == 0:
      return np.array([], dtype='int32')
    # Elements from int32 vector are sored in tuples: (sizeof(int32), value),
    vec = np.frombuffer(fd.read(vec_size*5), dtype=[('size','int8'),('value','int32')], count=vec_size)
    assert(vec[0]['size'] == 4) # int32 size,
    ans = vec[:]['value'] # values are in 2nd column,
  else: # ascii,
    arr = (binary + fd.readline().decode()).strip().split()
    try:
      arr.remove('['); arr.remove(']') # optionally
    except ValueError:
      pass
    ans = np.array(arr, dtype=int)
  if fd is not file_or_fd : fd.close() # cleanup
  return ans

# Writing,
def write_vec_int(file_or_fd, output_folder, v, key=''):
  """ write_vec_int(f, v, key='')
   Write a binary kaldi integer vector to filename or stream.
   Arguments:
   file_or_fd : filename or opened file descriptor for writing,
   v : the vector to be stored,
   key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

   Example of writing single vector:
   kaldi_io.write_vec_int(filename, vec)

   Example of writing arkfile:
   with open(ark_file,'w') as f:
     for key,vec in dict.iteritems():
       kaldi_io.write_vec_flt(f, vec, key=key)
  """
  fd = open_or_fd(file_or_fd, output_folder, mode='wb')
  if sys.version_info[0] == 3: assert(fd.mode == 'wb')
  try:
    if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
    fd.write('\0B'.encode()) # we write binary!
    # dim,
    fd.write('\4'.encode()) # int32 type,
    fd.write(struct.pack(np.dtype('int32').char, v.shape[0]))
    # data,
    for i in range(len(v)):
      fd.write('\4'.encode()) # int32 type,
      fd.write(struct.pack(np.dtype('int32').char, v[i])) # binary,
  finally:
    if fd is not file_or_fd : fd.close()


#################################################
# Float vectors (confidences, ivectors, ...),

# Reading,
def read_vec_flt_scp(file_or_fd,output_folder):
  """ generator(key,mat) = read_vec_flt_scp(file_or_fd)
   Returns generator of (key,vector) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the scp:
   for key,vec in kaldi_io.read_vec_flt_scp(file):
     ...

   Read scp to a 'dictionary':
   d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    for line in fd:
      (key,rxfile) = line.decode().split(' ')
      vec = read_vec_flt(rxfile)
      yield key, vec
  finally:
    if fd is not file_or_fd : fd.close()

def read_vec_flt_ark(file_or_fd,output_folder):
  """ generator(key,vec) = read_vec_flt_ark(file_or_fd)
   Create generator of (key,vector<float>) tuples, reading from an ark file/stream.
   file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

   Read ark to a 'dictionary':
   d = { u:d for u,d in kaldi_io.read_vec_flt_ark(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    key = read_key(fd)
    while key:
      ali = read_vec_flt(fd)
      yield key, ali
      key = read_key(fd)
  finally:
    if fd is not file_or_fd: fd.close()

def read_vec_flt(file_or_fd,output_folder):
  """ [flt-vec] = read_vec_flt(file_or_fd)
   Read kaldi float vector, ascii or binary input,
  """
  fd = open_or_fd(file_or_fd,output_folder)
  binary = fd.read(2).decode()
  if binary == '\0B': # binary flag
    return _read_vec_flt_binary(fd)
  else:  # ascii,
    arr = (binary + fd.readline().decode()).strip().split()
    try:
      arr.remove('['); arr.remove(']') # optionally
    except ValueError:
      pass
    ans = np.array(arr, dtype=float)
  if fd is not file_or_fd : fd.close() # cleanup
  return ans

def _read_vec_flt_binary(fd):
  header = fd.read(3).decode()
  if header == 'FV ' : sample_size = 4 # floats
  elif header == 'DV ' : sample_size = 8 # doubles
  else : raise UnknownVectorHeader("The header contained '%s'" % header)
  assert (sample_size > 0)
  # Dimension,
  assert (fd.read(1).decode() == '\4'); # int-size
  vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # vector dim
  if vec_size == 0:
    return np.array([], dtype='float32')
  # Read whole vector,
  buf = fd.read(vec_size * sample_size)
  if sample_size == 4 : ans = np.frombuffer(buf, dtype='float32')
  elif sample_size == 8 : ans = np.frombuffer(buf, dtype='float64')
  else : raise BadSampleSize
  return ans


# Writing,
def write_vec_flt(file_or_fd, output_folder, v, key=''):
  """ write_vec_flt(f, v, key='')
   Write a binary kaldi vector to filename or stream. Supports 32bit and 64bit floats.
   Arguments:
   file_or_fd : filename or opened file descriptor for writing,
   v : the vector to be stored,
   key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

   Example of writing single vector:
   kaldi_io.write_vec_flt(filename, vec)

   Example of writing arkfile:
   with open(ark_file,'w') as f:
     for key,vec in dict.iteritems():
       kaldi_io.write_vec_flt(f, vec, key=key)
  """
  fd = open_or_fd(file_or_fd,output_folder, mode='wb')
  if sys.version_info[0] == 3: assert(fd.mode == 'wb')
  try:
    if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
    fd.write('\0B'.encode()) # we write binary!
    # Data-type,
    if v.dtype == 'float32': fd.write('FV '.encode())
    elif v.dtype == 'float64': fd.write('DV '.encode())
    else: raise UnsupportedDataType("'%s', please use 'float32' or 'float64'" % v.dtype)
    # Dim,
    fd.write('\04'.encode())
    fd.write(struct.pack(np.dtype('uint32').char, v.shape[0])) # dim
    # Data,
    fd.write(v.tobytes())
  finally:
    if fd is not file_or_fd : fd.close()


#################################################
# Float matrices (features, transformations, ...),

# Reading,
def read_mat_scp(file_or_fd,output_folder):
  """ generator(key,mat) = read_mat_scp(file_or_fd)
   Returns generator of (key,matrix) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the scp:
   for key,mat in kaldi_io.read_mat_scp(file):
     ...

   Read scp to a 'dictionary':
   d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    for line in fd:
      (key,rxfile) = line.decode().split(' ')
      mat = read_mat(rxfile,output_folder)
      yield key, mat
  finally:
    if fd is not file_or_fd : fd.close()

def read_mat_ark(file_or_fd,output_folder):
  """ generator(key,mat) = read_mat_ark(file_or_fd)
   Returns generator of (key,matrix) tuples, read from ark file/stream.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the ark:
   for key,mat in kaldi_io.read_mat_ark(file):
     ...

   Read ark to a 'dictionary':
   d = { key:mat for key,mat in kaldi_io.read_mat_ark(file) }
  """


  fd = open_or_fd(file_or_fd,output_folder)
  try:
    key = read_key(fd)
    while key:
      mat = read_mat(fd,output_folder)
      yield key, mat
      key = read_key(fd)   
  finally:
    if fd is not file_or_fd : fd.close()
  


def read_mat(file_or_fd,output_folder):
  """ [mat] = read_mat(file_or_fd)
   Reads single kaldi matrix, supports ascii and binary.
   file_or_fd : file, gzipped file, pipe or opened file descriptor.
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    binary = fd.read(2).decode()
    if binary == '\0B' :
      mat = _read_mat_binary(fd)
    else:
      assert(binary == ' [')
      mat = _read_mat_ascii(fd)
  finally:
    if fd is not file_or_fd: fd.close()
  return mat

def _read_mat_binary(fd):
  # Data type
  header = fd.read(3).decode()
  # 'CM', 'CM2', 'CM3' are possible values,
  if header.startswith('CM'): return _read_compressed_mat(fd, header)
  elif header == 'FM ': sample_size = 4 # floats
  elif header == 'DM ': sample_size = 8 # doubles
  else: raise UnknownMatrixHeader("The header contained '%s'" % header)
  assert(sample_size > 0)
  # Dimensions
  s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
  # Read whole matrix
  buf = fd.read(rows * cols * sample_size)
  if sample_size == 4 : vec = np.frombuffer(buf, dtype='float32')
  elif sample_size == 8 : vec = np.frombuffer(buf, dtype='float64')
  else : raise BadSampleSize
  mat = np.reshape(vec,(rows,cols))
  return mat

def _read_mat_ascii(fd):
  rows = []
  while 1:
    line = fd.readline().decode()
    if (len(line) == 0) : raise BadInputFormat # eof, should not happen!
    if len(line.strip()) == 0 : continue # skip empty line
    arr = line.strip().split()
    if arr[-1] != ']':
      rows.append(np.array(arr,dtype='float32')) # not last line
    else:
      rows.append(np.array(arr[:-1],dtype='float32')) # last line
      mat = np.vstack(rows)
      return mat


def _read_compressed_mat(fd, format):
  """ Read a compressed matrix,
      see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
      methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
  """
  assert(format == 'CM ') # The formats CM2, CM3 are not supported...

  # Format of header 'struct',
  global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')]) # member '.format' is not written,
  per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])

  # Read global header,
  globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]

  # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
  #                         {           cols           }{     size         }
  col_headers = np.frombuffer(fd.read(cols*8), dtype=per_col_header, count=cols)
  col_headers = np.array([np.array([x for x in y]) * globrange * 1.52590218966964e-05 + globmin for y in col_headers], dtype=np.float32)
  data = np.reshape(np.frombuffer(fd.read(cols*rows), dtype='uint8', count=cols*rows), newshape=(cols,rows)) # stored as col-major,

  mat = np.zeros((cols,rows), dtype='float32')
  p0 = col_headers[:, 0].reshape(-1, 1)
  p25 = col_headers[:, 1].reshape(-1, 1)
  p75 = col_headers[:, 2].reshape(-1, 1)
  p100 = col_headers[:, 3].reshape(-1, 1)
  mask_0_64 = (data <= 64)
  mask_193_255 = (data > 192)
  mask_65_192 = (~(mask_0_64 | mask_193_255))

  mat += (p0  + (p25 - p0) / 64. * data) * mask_0_64.astype(np.float32)
  mat += (p25 + (p75 - p25) / 128. * (data - 64)) * mask_65_192.astype(np.float32)
  mat += (p75 + (p100 - p75) / 63. * (data - 192)) * mask_193_255.astype(np.float32)

  return mat.T # transpose! col-major -> row-major,


# Writing,
def write_mat(output_folder,file_or_fd, m, key=''):
  """ write_mat(f, m, key='')
  Write a binary kaldi matrix to filename or stream. Supports 32bit and 64bit floats.
  Arguments:
   file_or_fd : filename of opened file descriptor for writing,
   m : the matrix to be stored,
   key (optional) : used for writing ark-file, the utterance-id gets written before the matrix.

   Example of writing single matrix:
   kaldi_io.write_mat(filename, mat)

   Example of writing arkfile:
   with open(ark_file,'w') as f:
     for key,mat in dict.iteritems():
       kaldi_io.write_mat(f, mat, key=key)
  """
  fd = open_or_fd(file_or_fd, output_folder, mode='wb')
  if sys.version_info[0] == 3: assert(fd.mode == 'wb')
  try:
    if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
    fd.write('\0B'.encode()) # we write binary!
    # Data-type,
    if m.dtype == 'float32': fd.write('FM '.encode())
    elif m.dtype == 'float64': fd.write('DM '.encode())
    else: raise UnsupportedDataType("'%s', please use 'float32' or 'float64'" % m.dtype)
    # Dims,
    fd.write('\04'.encode())
    fd.write(struct.pack(np.dtype('uint32').char, m.shape[0])) # rows
    fd.write('\04'.encode())
    fd.write(struct.pack(np.dtype('uint32').char, m.shape[1])) # cols
    # Data,
    fd.write(m.tobytes())
  finally:
    if fd is not file_or_fd : fd.close()


#################################################
# 'Posterior' kaldi type (posteriors, confusion network, nnet1 training targets, ...)
# Corresponds to: vector<vector<tuple<int,float> > >
# - outer vector: time axis
# - inner vector: records at the time
# - tuple: int = index, float = value
#

def read_cnet_ark(file_or_fd,output_folder):
  """ Alias of function 'read_post_ark()', 'cnet' = confusion network """
  return read_post_ark(file_or_fd,output_folder)

def read_post_rxspec(file_):
  """ adaptor to read both 'ark:...' and 'scp:...' inputs of posteriors,
  """
  if file_.startswith("ark:"):
      return read_post_ark(file_)
  elif file_.startswith("scp:"):
      return read_post_scp(file_)
  else:
      print("unsupported intput type: %s" % file_)
      print("it should begint with 'ark:' or 'scp:'")
      sys.exit(1)

def read_post_scp(file_or_fd,output_folder):
  """ generator(key,post) = read_post_scp(file_or_fd)
   Returns generator of (key,post) tuples, read according to kaldi scp.
   file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

   Iterate the scp:
   for key,post in kaldi_io.read_post_scp(file):
     ...

   Read scp to a 'dictionary':
   d = { key:post for key,post in kaldi_io.read_post_scp(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    for line in fd:
      (key,rxfile) = line.decode().split(' ')
      post = read_post(rxfile)
      yield key, post
  finally:
    if fd is not file_or_fd : fd.close()

def read_post_ark(file_or_fd,output_folder):
  """ generator(key,vec<vec<int,float>>) = read_post_ark(file)
   Returns generator of (key,posterior) tuples, read from ark file.
   file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

   Iterate the ark:
   for key,post in kaldi_io.read_post_ark(file):
     ...

   Read ark to a 'dictionary':
   d = { key:post for key,post in kaldi_io.read_post_ark(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    key = read_key(fd)
    while key:
      post = read_post(fd)
      yield key, post
      key = read_key(fd)
  finally:
    if fd is not file_or_fd: fd.close()

def read_post(file_or_fd,output_folder):
  """ [post] = read_post(file_or_fd)
   Reads single kaldi 'Posterior' in binary format.

   The 'Posterior' is C++ type 'vector<vector<tuple<int,float> > >',
   the outer-vector is usually time axis, inner-vector are the records
   at given time,  and the tuple is composed of an 'index' (integer)
   and a 'float-value'. The 'float-value' can represent a probability
   or any other numeric value.

   Returns vector of vectors of tuples.
  """
  fd = open_or_fd(file_or_fd,output_folder)
  ans=[]
  binary = fd.read(2).decode(); assert(binary == '\0B'); # binary flag
  assert(fd.read(1).decode() == '\4'); # int-size
  outer_vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of frames (or bins)

  # Loop over 'outer-vector',
  for i in range(outer_vec_size):
    assert(fd.read(1).decode() == '\4'); # int-size
    inner_vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of records for frame (or bin)
    data = np.frombuffer(fd.read(inner_vec_size*10), dtype=[('size_idx','int8'),('idx','int32'),('size_post','int8'),('post','float32')], count=inner_vec_size)
    assert(data[0]['size_idx'] == 4)
    assert(data[0]['size_post'] == 4)
    ans.append(data[['idx','post']].tolist())

  if fd is not file_or_fd: fd.close()
  return ans


#################################################
# Kaldi Confusion Network bin begin/end times,
# (kaldi stores CNs time info separately from the Posterior).
#

def read_cntime_ark(file_or_fd,output_folder):
  """ generator(key,vec<tuple<float,float>>) = read_cntime_ark(file_or_fd)
   Returns generator of (key,cntime) tuples, read from ark file.
   file_or_fd : file, gzipped file, pipe or opened file descriptor.

   Iterate the ark:
   for key,time in kaldi_io.read_cntime_ark(file):
     ...

   Read ark to a 'dictionary':
   d = { key:time for key,time in kaldi_io.read_post_ark(file) }
  """
  fd = open_or_fd(file_or_fd,output_folder)
  try:
    key = read_key(fd)
    while key:
      cntime = read_cntime(fd)
      yield key, cntime
      key = read_key(fd)
  finally:
    if fd is not file_or_fd : fd.close()

def read_cntime(file_or_fd,output_folder):
  """ [cntime] = read_cntime(file_or_fd)
   Reads single kaldi 'Confusion Network time info', in binary format:
   C++ type: vector<tuple<float,float> >.
   (begin/end times of bins at the confusion network).

   Binary layout is '<num-bins> <beg1> <end1> <beg2> <end2> ...'

   file_or_fd : file, gzipped file, pipe or opened file descriptor.

   Returns vector of tuples.
  """
  fd = open_or_fd(file_or_fd,output_folder)
  binary = fd.read(2).decode(); assert(binary == '\0B'); # assuming it's binary

  assert(fd.read(1).decode() == '\4'); # int-size
  vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of frames (or bins)

  data = np.frombuffer(fd.read(vec_size*10), dtype=[('size_beg','int8'),('t_beg','float32'),('size_end','int8'),('t_end','float32')], count=vec_size)
  assert(data[0]['size_beg'] == 4)
  assert(data[0]['size_end'] == 4)
  ans = data[['t_beg','t_end']].tolist() # Return vector of tuples (t_beg,t_end),

  if fd is not file_or_fd : fd.close()
  return ans


#################################################
# Segments related,
#

# Segments as 'Bool vectors' can be handy,
# - for 'superposing' the segmentations,
# - for frame-selection in Speaker-ID experiments,
def read_segments_as_bool_vec(segments_file):
  """ [ bool_vec ] = read_segments_as_bool_vec(segments_file)
   using kaldi 'segments' file for 1 wav, format : '<utt> <rec> <t-beg> <t-end>'
   - t-beg, t-end is in seconds,
   - assumed 100 frames/second,
  """
  segs = np.loadtxt(segments_file, dtype='object,object,f,f', ndmin=1)
  # Sanity checks,
  assert(len(segs) > 0) # empty segmentation is an error,
  assert(len(np.unique([rec[1] for rec in segs ])) == 1) # segments with only 1 wav-file,
  # Convert time to frame-indexes,
  start = np.rint([100 * rec[2] for rec in segs]).astype(int)
  end = np.rint([100 * rec[3] for rec in segs]).astype(int)
  # Taken from 'read_lab_to_bool_vec', htk.py,
  frms = np.repeat(np.r_[np.tile([False,True], len(end)), False],
                   np.r_[np.c_[start - np.r_[0, end[:-1]], end-start].flat, 0])
  assert np.sum(end-start) == np.sum(frms)
  return frms
