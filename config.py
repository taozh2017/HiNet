#coding:utf8
import warnings
class DefaultConfig(object):
        
    data_path       = 'data/'
    save_path       = 'results/'

    use_gpu         = True # user GPU or not    
    
    lr              = 0.001 # initial learning rate
    lr_decay        = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay    = 1e-4 # 
    
    batch_size      = 1 # batch size
    epochs          = 400
    decay_epoch     = 100
    dropout         = 0.5
    
    fold            = 0
    lambda1         = 5
    lambda2         = 10
    
    gpu_id          = [0,1,2,3]
 
    checkpoint          = 'checkpointNEW'
    checkpoint_interval = 5  
    
    b1 = 0.5
    b2 = 0.999
    n_critic = 5

    task_id  = 1

            
def parse(self,kwargs):
        '''
        根据字典kwargs 更新 config参数
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self,k))

DefaultConfig.parse = parse
opt =DefaultConfig()

#parser = argparse.ArgumentParser()
#parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
#parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
#parser.add_argument('--;', type=int, default=2, help='size of the batches')
#parser.add_argument('--dataset_name', type=str, default='edges2shoes', help='name of the dataset')
#parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
#parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
#parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
#parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
#parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
#parser.add_argument('--channels', type=int, default=3, help='number of image channels')
#parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
#parser.add_argument('--sample_interval', type=int, default=200, help='interval betwen image samples')
#parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
#opt = parser.parse_args()
#print(opt)

