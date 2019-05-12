# This library specifies the set of operations that can be defined in the model section.
# Users can define here customized operations not included in the original set of operations.

##########################################################
# pytorch-kaldi v.0.2                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# June 2019
##########################################################


import importlib
from distutils.util import strtobool
import torch.nn as nn


class compute(nn.Module):
    # The class manages initialization and forward phase of a neural network defined in the model section
    
    def __init__(self, out_name,inp1,inp2,input_info,arch_dict,config,use_cuda,multi_gpu,to_do,nns,costs):
        super(compute,self).__init__()
        # This method initializes the neural network specified within the “compute” line.
        # It also stores into the "inp_info" dictionary the output dimensinality.
      
        # computing input dimensionality (the last element of the input_info[inp2] list contains this info)
        inp_dim=input_info[inp2][-1]
        
        # import the class of the specified neural network
        module = importlib.import_module(config[arch_dict[inp1][0]]['arch_library'])
        nn_class=getattr(module, config[arch_dict[inp1][0]]['arch_class'])
        
        # add use cuda and todo options
        config.set(arch_dict[inp1][0],'use_cuda',config['exp']['use_cuda'])
        config.set(arch_dict[inp1][0],'to_do',config['exp']['to_do'])
        
        arch_freeze_flag=strtobool(config[arch_dict[inp1][0]]['arch_freeze'])
        
        # initialize the neural network
        net=nn_class(config[arch_dict[inp1][0]],inp_dim)
        
        # transfer the model to cuda is specified in the config file
        if use_cuda:
            net.cuda()
            if multi_gpu:
                # use multiple-gpus if specified in the config file
                net = nn.DataParallel(net)
                
        # activate train flag if we are under trainig modality
        if to_do=='train':
            if not(arch_freeze_flag):
                net.train()
            else:
               # Switch to eval modality if architecture is frozen (mainly for batch_norm/dropout functions)
               net.eval() 
        else:
            # activate eval flag if we are under test/eval modality
            net.eval()
        
        
        # adding the initialized neural network into the nns dict
        nns[arch_dict[inp1][1]]=net
        
        # compiute the output dimensionality
        if multi_gpu:
            out_dim=net.module.out_dim
        else:
            out_dim=net.out_dim
            
        # storing the output dimensionality in the input_info dictionary
        input_info[out_name]=[out_dim]
        
        self.net=net
                       

    def forward(self,input_dnn):
        
        # Compute the output of the neural network given the input
        return self.net(input_dnn)
        
        
 
class concatenate(nn.Module):
    # The class manages the concatenation operation between two tensors
    
    def __init__(self, out_name,inp1,inp2,input_info,arch_dict,config,use_cuda,multi_gpu,to_do,nns,costs):
        #The initialization classes just compute the output dimensionality given the two input dimensionalities
        inp_dim1=input_info[inp1][-1]
        inp_dim2=input_info[inp2][-1]
        input_info[out_name]=[inp_dim1+inp_dim2]
 

    def forward(self):
        print("Hello")   
        
        
class mult(nn.Module):
    # The class manages the element-wise multiplication operation between two tensors
    
    def __init__(self, out_name,inp1,inp2,input_info,arch_dict,config,use_cuda,multi_gpu,to_do,nns,costs):
        #The initialization classes just compute the output dimensionality given the input dimensionality
        input_info[out_name]=input_info[inp1]
 

    def forward(self):
        print("Hello") 
        
        
class mult_constant(nn.Module):
    # The class manages the multiplication operation between a tensor and a constant
    
    def __init__(self, out_name,inp1,inp2,input_info,arch_dict,config,use_cuda,multi_gpu,to_do,nns,costs):
        #The initialization classes just compute the output dimensionality given the input dimensionality
        input_info[out_name]=input_info[inp1]
 

    def forward(self):
        print("Hello")  
        
        
class sum(nn.Module):
    # The class manages the element-wise sum operation between two tensors
    
    def __init__(self, out_name,inp1,inp2,input_info,arch_dict,config,use_cuda,multi_gpu,to_do,nns,costs):
        #The initialization classes just compute the output dimensionality given the input dimensionality
        input_info[out_name]=input_info[inp1]
 

    def forward(self):
        print("Hello")   
        
        
            
class sum_constant(nn.Module):
    # The class manages the sum operation between a tensor and a constant
    
    def __init__(self, out_name,inp1,inp2,input_info,arch_dict,config,use_cuda,multi_gpu,to_do,nns,costs):
        #The initialization classes just compute the output dimensionality given the input dimensionality
        input_info[out_name]=input_info[inp1]
 

    def forward(self):
        print("Hello")  
        
class avg(nn.Module):
    # The class manages the elemet-wise average operation between two tensors
    
    def __init__(self, out_name,inp1,inp2,input_info,arch_dict,config,use_cuda,multi_gpu,to_do,nns,costs):
        #The initialization classes just compute the output dimensionality given the input dimensionality
        input_info[out_name]=input_info[inp1]
 

    def forward(self):
        print("Hello")  
        
        
        
class cost_nll(nn.Module):
    # The class computes the negative log-likelihood (cross-entropy) cost
    
    def __init__(self, out_name,inp1,inp2,input_info,arch_dict,config,use_cuda,multi_gpu,to_do,nns,costs):
        # cost initialization
        costs[out_name] = nn.NLLLoss()
        # the cost outputs a scalar (output_dim=1)
        input_info[out_name]=[1]
 

    def forward(self):
        print("Hello") 
        
        
class cost_err(nn.Module):
    # The class computes the classification error
    
    def __init__(self, out_name,inp1,inp2,input_info,arch_dict,config,use_cuda,multi_gpu,to_do,nns,costs):
        # the cost outputs a scalar (output_dim=1)
        input_info[out_name]=[1]
 

    def forward(self):
        print("Hello")
        

class cost_mse(nn.Module):
    # The class computes the mean squared error (MSE) 
    
    def __init__(self, out_name,inp1,inp2,input_info,arch_dict,config,use_cuda,multi_gpu,to_do,nns,costs):
        # the cost outputs a scalar (output_dim=1)
        input_info[out_name]=[1]
 

    def forward(self):
        print("Hello") 