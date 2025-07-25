from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "Light_HookNet"

__C.CUDA = True

__C.output_path = './checkpoints'

# MODEL
__C.in_channels = 1
__C.n_classes = 4               
__C.n_filters = 16               
__C.filter_size = 3              
__C.learning_rate = 0.001
__C.total_epoch = 300            
__C.steps = 2                    
__C.batch_size = 40             
__C.seed = 123
__C.load = False
__C.scale = 1
