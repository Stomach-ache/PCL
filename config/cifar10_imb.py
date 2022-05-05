config = {}
config['data_path'] = './data/cifar10'
config['dataset'] = 'cifar10'
config['openset'] = False
config['noise_ratio'] = 0.5
config['noise_mode'] = 'imb'
config['imb_type'] = 'exp'
config['imb_ratio'] = 0.1
config['imb_file'] = 'imb_file/%s_%s_%s.json'%(config['dataset'],config['imb_type'],str(config['imb_ratio']))
config['noise_file'] = 'noise_file/%s_%s_%s_%s_%.1f.json'%(config['dataset'],config['imb_type'],str(config['imb_ratio']),config['noise_mode'],config['noise_ratio'])

data_train_opt = {} 
data_train_opt['batch_size'] = 128
data_train_opt['temperature'] = 0.3
data_train_opt['num_class'] = 10
data_train_opt['alpha'] = 8
data_train_opt['w_inst'] = 1
data_train_opt['w_proto'] = 5
data_train_opt['w_recon'] = 1
data_train_opt['low_dim'] = 50 
data_train_opt['warmup_iters'] = 100 
data_train_opt['ramp_epoch'] = 0

config['data_train_opt'] = data_train_opt
config['max_num_epochs'] = 200

config['test_knn'] = True
config['knn_start_epoch'] = 5
config['knn'] = True
config['n_neighbors'] = 10
config['low_th'] = 0.1
config['high_th'] = -0.4
config['gamma'] = 1.005
#config['gamma'] = 1.01

networks = {}
lr = 0.02
net_optim_params = {'optim_type': 'sgd', 'lr': lr, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov':False}
networks['model'] = {'name': 'presnet', 'pretrained': None, 'opt': {},  'optim_params': net_optim_params}
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':{}} 
criterions['loss_instance'] = {'ctype':'CrossEntropyLoss', 'opt':{}} 

config['criterions'] = criterions
config['algorithm_type'] = 'Model'

config['exp_directory'] = 'experiment/cifar10_%s_%s_%s_%.1f'%(config['imb_type'],str(config['imb_ratio']),config['noise_mode'],config['noise_ratio'])
config['checkpoint_dir'] = 'experiment/cifar10_%s_%s_%s_%.1f'%(config['imb_type'],str(config['imb_ratio']),config['noise_mode'],config['noise_ratio'])

#config['log_tag'] = 'no_gamma'
#config['log_tag'] = 'tune_gamma'
#config['log_tag'] = 'no_mixup'
#config['log_tag'] = 'no_augmix'
#config['log_tag'] = 'no_cc'
#config['log_tag'] = 'no_ce'
config['log_tag'] = 'scl'
#config['log_tag'] = 'wproto_6'
