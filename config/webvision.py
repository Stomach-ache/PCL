config = {}
config['data_path'] = '../DivideMix-master-0518/webvision/'
config['dataset'] = 'webvision'
config['imb_type'] = 'exp'
config['imb_ratio'] = 0.01

data_train_opt = {} 
data_train_opt['batch_size'] = 64
data_train_opt['temperature'] = 0.3
data_train_opt['num_class'] = 50
data_train_opt['alpha'] = 0.5
data_train_opt['w_inst'] = 1
data_train_opt['w_proto'] = 2
data_train_opt['w_recon'] = 1
data_train_opt['low_dim'] = 50 
data_train_opt['warmup_iters'] = 200 
data_train_opt['ramp_epoch'] = 4 #(optional ramp up)

config['data_train_opt'] = data_train_opt
config['max_num_epochs'] = 40

config['test_knn'] = True
config['knn_start_epoch'] = 15
config['knn'] = True
config['n_neighbors'] = 200
config['low_th'] = 0.05
config['high_th'] = -0.8
config['gamma'] = 1.005

networks = {}
lr = 0.04
net_optim_params = {'optim_type': 'sgd', 'lr': lr, 'momentum':0.9, 'weight_decay': 1e-4, 'nesterov':False}
networks['model'] = {'name': 'inception', 'pretrained': None, 'opt': {},  'optim_params': net_optim_params}
config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':{}} 
criterions['loss_instance'] = {'ctype':'CrossEntropyLoss', 'opt':{}} 

config['criterions'] = criterions
config['algorithm_type'] = 'Model'

#config['exp_directory'] = 'experiments/webvision' 
config['exp_directory'] = 'experiment/webvision_%s_%s'%(config['imb_type'],str(config['imb_ratio']))

config['checkpoint_dir'] = 'experiment/webvision_%s_%s'%(config['imb_type'],str(config['imb_ratio']))

config['imb_file'] = 'imb_file/%s_%s_%s.json'%(config['dataset'],config['imb_type'],str(config['imb_ratio']))

config['log_tag'] = ''
