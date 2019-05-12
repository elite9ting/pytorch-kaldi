# This library specifies the supported optimizers.
# Users can define here additional/customized optimizers not included in the original set.

##########################################################
# pytorch-kaldi v.0.2                                      
# Mirco Ravanelli, Titouan Parcollet
# Mila, University of Montreal
# June 2019
##########################################################

from distutils.util import strtobool
import torch.optim as optim

class sgd:
    
    # Initialization of Stochastic Gradient Descend (SGD) optimizer
    def __init__(self, model_nn,net,config):
        super(sgd,self).__init__()
        
        arch_info=model_nn.arch_info
        nns=model_nn.nns
        
        lr=float(config[arch_info[net][0]]['arch_lr'])
            
        opt_momentum=float(config[arch_info[net][0]]['opt_momentum'])
        opt_weight_decay=float(config[arch_info[net][0]]['opt_weight_decay'])
        opt_dampening=float(config[arch_info[net][0]]['opt_dampening'])
        opt_nesterov=strtobool(config[arch_info[net][0]]['opt_nesterov'])
        
        
        self.optimizer=(optim.SGD(nns[net].parameters(),
                  lr=lr,
                  momentum=opt_momentum,
                  weight_decay=opt_weight_decay,
                  dampening=opt_dampening,
                  nesterov=opt_nesterov))
                      

    def step(self):
        
        print("Hello")
        
        
        
class adam:
    
    # Initialization of Adam optimizer
    def __init__(self, model_nn,net,config):
        super(adam,self).__init__()
        arch_info=model_nn.arch_info
        nns=model_nn.nns
        
        lr=float(config[arch_info[net][0]]['arch_lr'])
        
            
        opt_betas=list(map(float,(config[arch_info[net][0]]['opt_betas'].split(','))))
        opt_eps=float(config[arch_info[net][0]]['opt_eps'])
        opt_weight_decay=float(config[arch_info[net][0]]['opt_weight_decay'])
        opt_amsgrad=strtobool(config[arch_info[net][0]]['opt_amsgrad'])
        
        self.optimizer=(optim.Adam(nns[net].parameters(),
                  lr=lr,
                  betas=opt_betas,
                  eps=opt_eps,
                  weight_decay=opt_weight_decay,
                  amsgrad=opt_amsgrad))
                      

    def step(self):
        print("Hello")
        
        
class rmsprop:
    
    # Initialization of rmsprop optimizer
    def __init__(self, model_nn,net,config):
        super(rmsprop,self).__init__()

        arch_info=model_nn.arch_info
        nns=model_nn.nns
        
        lr=float(config[arch_info[net][0]]['arch_lr'])
        
        opt_momentum=float(config[arch_info[net][0]]['opt_momentum'])
        opt_alpha=float(config[arch_info[net][0]]['opt_alpha'])
        opt_eps=float(config[arch_info[net][0]]['opt_eps'])
        opt_centered=strtobool(config[arch_info[net][0]]['opt_centered'])
        opt_weight_decay=float(config[arch_info[net][0]]['opt_weight_decay'])
        
        
        self.optimizer=(optim.RMSprop(nns[net].parameters(),
                  lr=lr,
                  momentum=opt_momentum,
                  alpha=opt_alpha,
                  eps=opt_eps,
                  centered=opt_centered,
                  weight_decay=opt_weight_decay))
            

    def step(self):
        print("Hello")
        
        
        
        
 
