##########################################################
# pytorch-kaldi v.0.1                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# October 2018
##########################################################

import sys
import configparser
import os
from utils import is_sequential_dict,forward_model,progress
from data_io import load_counts, LoadData, fea_to_cuda
import numpy as np
import random
import torch
from distutils.util import strtobool
import time
import threading
import copy
from data_io import open_or_fd,write_mat
from utils import shift
import re
import torch.nn as nn
import model_computations
import optimizers


def run_nn(labs,cfg_file,processed_first,next_config_file):
    
    # This function processes the current chunk using the information in cfg_file. In parallel, the next chunk is load into the CPU memory
    
    # Reading chunk-specific cfg file (first argument-mandatory file) 
    if not(os.path.exists(cfg_file)):
         sys.stderr.write('ERROR: The config file %s does not exist!\n'%(cfg_file))
         sys.exit(0)
    else:
        config = configparser.ConfigParser()
        config.read(cfg_file)
    
    # Setting torch seed
    seed=int(config['exp']['seed'])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
    # Reading config parameters
    output_folder=config['exp']['out_folder']
    use_cuda=strtobool(config['exp']['use_cuda'])
    multi_gpu=strtobool(config['exp']['multi_gpu'])
    
    to_do=config['exp']['to_do']
    info_file=config['exp']['out_info']
    
    model_sec=config['model']['model'].split('\n')
    
    forward_outs=config['forward']['forward_out'].split(',')
    forward_normalize_post=list(map(strtobool,config['forward']['normalize_posteriors'].split(',')))
    forward_count_files=config['forward']['normalize_with_counts_from'].split(',')
    require_decodings=list(map(strtobool,config['forward']['require_decoding'].split(',')))
    
    use_cuda=strtobool(config['exp']['use_cuda'])
    save_gpumem=strtobool(config['exp']['save_gpumem'])


    if to_do=='train':
        batch_size=int(config['batches']['batch_size'])
    
    if to_do=='valid':
        batch_size=int(config['batches']['batch_size'])
    
    if to_do=='forward':
        batch_size=1
        
    
    # ***** Reading the Data********
    if processed_first:
        
        # Reading all the features and labels for this chunk
        # read_data is a list containing: features=read_data[0], fea_info=read_data[1],labels=sread_data[2],lab_info=read_data[3], arch_info=shared_list[4]
        read_data=[]
        
        p=threading.Thread(target=LoadData, args=(cfg_file,labs,read_data))
        p.start()
        p.join()
        

        # Initializing the neural networks specified in the model section (we do only the first time when process_first=True)
        model_nn=model(read_data,config)
        
        # initializing the optimizers from the architectures defined in the config file
        optimizer_nn=optim(model_nn,config)
        
        features_batch=read_data[0]
        features_batch['batch_0']
        
        model_nn(features_batch['batch_0'])
        # To do implement model_nn forward method. test it
        # then, implement optimin_step and test it.
        
        sys.exit(0)
        

        
        # optimizers initialization
        optimizers=optimizer_init(model_nn.nns,config,arch_info)
        

                                      
    # Reading all the features for the next chunk (while training the DNN for the current chunk)
    shared_list=[]
    p=threading.Thread(target=LoadData, args=(cfg_file,labs,shared_list))
    p.start()
    
    # ***** Neural network computations ***** 
    
    # pre-training (loading the parameters learned in the previous chunk)
    for net in model_nn.nns.keys():
      pt_file_arch=config[arch_info[net][0]]['arch_pretrain_file']
      
      if pt_file_arch!='none':        
          checkpoint_load = torch.load(pt_file_arch)
          model_nn.nns[net].load_state_dict(checkpoint_load['model_par'])
          optimizers[net].load_state_dict(checkpoint_load['optimizer_par'])
          optimizers[net].param_groups[0]['lr']=float(config[arch_info[net][0]]['arch_lr']) # loading lr of the cfg file for pt
 
       
#    if to_do=='forward':
#        
#        post_file={}
#        for out_id in range(len(forward_outs)):
#            if require_decodings[out_id]:
#                out_file=info_file.replace('.info','_'+forward_outs[out_id]+'_to_decode.ark')
#            else:
#                out_file=info_file.replace('.info','_'+forward_outs[out_id]+'.ark')
#            post_file[forward_outs[out_id]]=open_or_fd(out_file,output_folder,'wb')


    # check automatically if the model is sequential
    seq_model=is_sequential_dict(config,arch_info)

    # computing the number of batches in the chunk
    N_batches=len(list(features.keys()))
    
    loss_sum=0
    err_sum=0
    
    for i in range(N_batches):   
        
        max_len=0
    
        if seq_model:
         
         max_len=int(max(arr_snt_len[snt_index:snt_index+batch_size]))  
         inp= torch.zeros(max_len,batch_size,inp_dim).contiguous()
    
            
         for k in range(batch_size):
              
                  snt_len=data_end_index[snt_index]-beg_snt
                  N_zeros=max_len-snt_len
                  
                  # Appending a random number of initial zeros, tge others are at the end. 
                  N_zeros_left=random.randint(0,N_zeros)
                 
                  # randomizing could have a regularization effect
                  inp[N_zeros_left:N_zeros_left+snt_len,k,:]=data_set[beg_snt:beg_snt+snt_len,:]
                  
                  beg_snt=data_end_index[snt_index]
                  snt_index=snt_index+1
                
        else:
            # features and labels for batch i
            if to_do!='forward':
                inp= data_set[beg_batch:end_batch,:].contiguous()
            else:
                snt_len=data_end_index[snt_index]-beg_snt
                inp= data_set[beg_snt:beg_snt+snt_len,:].contiguous()
                beg_snt=data_end_index[snt_index]
                snt_index=snt_index+1
    
        # use cuda
        if use_cuda:
            inp=inp.cuda()
    
        if to_do=='train':
            # Forward input, with autograd graph active
            outs_dict=forward_model(fea_dict,lab_dict,arch_dict,model,nns,costs,inp,inp_out_dict,max_len,batch_size,to_do,forward_outs)
            
            for opt in optimizers.keys():
                optimizers[opt].zero_grad()
                
    
            outs_dict['loss_final'].backward()
            
            # Gradient Clipping (th 0.1)
            #for net in nns.keys():
            #    torch.nn.utils.clip_grad_norm_(nns[net].parameters(), 0.1)
            
            
            for opt in optimizers.keys():
                if not(strtobool(config[arch_dict[opt][0]]['arch_freeze'])):
                    optimizers[opt].step()
        else:
            with torch.no_grad(): # Forward input without autograd graph (save memory)
                outs_dict=forward_model(fea_dict,lab_dict,arch_dict,model,nns,costs,inp,inp_out_dict,max_len,batch_size,to_do,forward_outs)
    
                    
        if to_do=='forward':
            for out_id in range(len(forward_outs)):
                
                out_save=outs_dict[forward_outs[out_id]].data.cpu().numpy()
                
                if forward_normalize_post[out_id]:
                    # read the config file
                    counts = load_counts(forward_count_files[out_id])
                    out_save=out_save-np.log(counts/np.sum(counts))             
                    
                # save the output    
                write_mat(output_folder,post_file[forward_outs[out_id]], out_save, data_name[i])
        else:
            loss_sum=loss_sum+outs_dict['loss_final'].detach()
            err_sum=err_sum+outs_dict['err_final'].detach()
           
        # update it to the next batch 
        beg_batch=end_batch
        end_batch=beg_batch+batch_size
        
        # Progress bar
        if to_do == 'train':
          status_string="Training | (Batch "+str(i+1)+"/"+str(N_batches)+")"+" | L:" +str(round(loss_sum.cpu().item()/(i+1),3))
          if i==N_batches-1:
             status_string="Training | (Batch "+str(i+1)+"/"+str(N_batches)+")"

             
        if to_do == 'valid':
          status_string="Validating | (Batch "+str(i+1)+"/"+str(N_batches)+")"
        if to_do == 'forward':
          status_string="Forwarding | (Batch "+str(i+1)+"/"+str(N_batches)+")"
          
        progress(i, N_batches, status=status_string)
    
    elapsed_time_chunk=time.time() - start_time 
    
    loss_tot=loss_sum/N_batches
    err_tot=err_sum/N_batches
    
    # clearing memory
    del inp, outs_dict, data_set
    
    # save the model
    if to_do=='train':
     
    
         for net in nns.keys():
             checkpoint={}
             checkpoint['model_par']=nns[net].state_dict()
             checkpoint['optimizer_par']=optimizers[net].state_dict()
             
             out_file=info_file.replace('.info','_'+arch_dict[net][0]+'.pkl')
             torch.save(checkpoint, out_file)
         
    if to_do=='forward':
        for out_name in forward_outs:
            post_file[out_name].close()
         
    
         
    # Write info file
    with open(info_file, "w") as text_file:
        text_file.write("[results]\n")
        if to_do!='forward':
            text_file.write("loss=%s\n" % loss_tot.cpu().numpy())
            text_file.write("err=%s\n" % err_tot.cpu().numpy())
        text_file.write("elapsed_time_chunk=%f\n" % elapsed_time_chunk)
    
    text_file.close()
    
    
    # Getting the data for the next chunk (read in parallel)    
    p.join()
    data_name=shared_list[0]
    data_end_index=shared_list[1]
    fea_dict=shared_list[2]
    lab_dict=shared_list[3]
    arch_dict=shared_list[4]
    data_set=shared_list[5]
    
    
    # converting numpy tensors into pytorch tensors and put them on GPUs if specified
    if not(save_gpumem) and use_cuda:
       data_set=torch.from_numpy(data_set).float().cuda()
    else:
       data_set=torch.from_numpy(data_set).float()
       
       
    return [data_name,data_set,data_end_index,fea_dict,lab_dict,arch_dict]


class model(nn.Module):
    
    # This class defines the model specified in the model sections. 
    # It might contain combinations of neural networks and features. 
    # The initialization method initializes the neural networks.
    # The forward method propagates the input features and computes the required outputs.
    
    def __init__(self, read_data,config):
        super(model,self).__init__()
        # This method initializes the neural networks specified in the model section 
        
        # reading the needed information on input features and architectures 
        fea_info=read_data[1]
        arch_info=read_data[4]
        
        print(arch_info['MLP_layers2'])
        sys.exit(0)
        # reading the needed fields from the config file
        use_cuda=strtobool(config['exp']['use_cuda'])
        multi_gpu=strtobool(config['exp']['multi_gpu'])
        to_do=config['exp']['to_do']
        model_sec=config['model']['model'].split('\n')
        
         # This is a dictionary that will contain the dimensionality of all the input sand outputs specified in the model section
        input_info= copy.deepcopy(fea_info) 
    
        # search pattern
        pattern='(.*)=(.*)\((.*),(.*)\)'
         
        # initializing the nns dictionary. It will contain all the parameters of the neural networks defined in the model section.
        nns={}
        
        # initializing the cost dictionary. It will contain all the costs/losses defined in the model section.
        costs={}
        
        # initializing the model computation dictionary. It tracks all the operations/computations defined in the model section.
        model_computations_dict={}
           
        # initalizing the list that contains the sequence of operation specified in the model section
        computation_lst=[]
        
        # reading the model section line by line 
        for line in model_sec:
            
            # Parsing the current line of the model section
            [out_name,operation_id,inp1,inp2]=list(re.findall(pattern,line)[0])
            
            # definition of an unique computation id (string)
            computation_id=out_name+':_:'+operation_id+':_:'+inp1+':_:'+inp2
            
            # reading the targeted operation from model_computations.py
            load_class=operation_id
            
            # if the operation id is the name of a neural archirecture (e.g, MLP_layer1(mfcc)) load the class compute that will compute the output of the specified neural architectute given the input
            print(arch_info.keys())
            if operation_id in list(arch_info.keys()):
                load_class='compute'
                
            print(load_class)
            print(operation_id)
            sys.exit(0)

            
            operation = getattr(model_computations, load_class)
            
            # initializing the neural networks (when needed)
            model_computations_dict[computation_id]=operation(out_name,inp1,inp2,input_info,arch_info,config,use_cuda,multi_gpu,to_do,nns,costs)
        
            # updated the operation list
            computation_lst.append([out_name,inp1,inp2,computation_id])
            
        self.nns=nns
        self.costs=costs
        self.model_computations_dict=model_computations_dict
        self.input_info=input_info
        self.arch_info=arch_info
        self.config=config
        self.computation_lst=computation_lst
        


    def forward(self,features):
        # Initializing the output dictionary
        
        # conversion of the feature to cuda (if needed)
        use_cuda=strtobool(self.config['exp']['use_cuda'])
        
        if use_cuda:
            features=fea_to_cuda(features)
           
        # initializing the output_dict. 
        # the output_dict is a dictionary that will contain the values for all the output defined in the model section when in input is given the current batch of features.
        self.output_dict=features
        
        # processing all the computations specified in the model section
        for computation in self.computation_lst:
            
            # reading the computation 
            out_name=computation[0]
            input_name=computation[2]
            computation_id=computation[3]
            
            # reading the input of the computation
            input_dnn=self.output_dict[input_name]
            
            self.output_dict[out_name]=self.model_computations_dict[computation_id](input_dnn)
            print(self.output_dict[out_name].shape)
            
            sys.exit(0)
            
            # To do: if operation_name in architecture_list then use compute
            

        
        
        sys.exit(0)
            

                
        
        

         
    
    
    def compute_costs(self,labels):
        print("Hello")
        
        

class optim:
    
    # This class defines the optimizers for the architectures defined in the config file. 
    # The initialization method initializes all the optimizers
    # The step method updates the parameters of the neural networks.
    
    def __init__(self, model_nn,config):
        
        # This method initializes the optimizers of the architectures defined in the config file
        optimizer_dict={}
        
        nns=model_nn.nns
        arch_info=model_nn.arch_info
        
        for net in nns.keys():
            
            # detecting the name of the optimizer
            optimizer_id=config[arch_info[net][0]]['arch_opt']
            arch_id=config[arch_info[net][0]]['arch_name']
            
            # reading the targeted optimizer from optimizers.py
            optimizer_nn = getattr(optimizers, optimizer_id)
            
            # initializing the optimizer for the neural network net
            optimizer_dict[arch_id+':_:'+optimizer_id]=optimizer_nn(model_nn,net,config)
            
        self.optimizer_dict= optimizer_dict
        


    def step(self,features):
        print("Hello")
