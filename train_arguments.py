import warnings
warnings.filterwarnings("ignore")

import yaml

class TrainArgs:
    def __init__(self, arguments_yaml_path):
        # 从yaml加载参数
        args = None
        with open(arguments_yaml_path, 'r', encoding='utf-8') as f:
            args = yaml.safe_load(f)
        
        args_train = args['train']
        
        # 学习器参数
        self.ppo_clip_param = args_train['ppo_clip_param']
        self.ppo_epoch = args_train['ppo_epoch']
        self.ppo_num_mini_batch = args_train['ppo_num_mini_batch']
        self.ppo_value_loss_coef = args_train['ppo_value_loss_coef']
        self.ppo_entropy_coef = args_train['ppo_entropy_coef']
        self.ppo_adam_learning_rate = args_train['ppo_adam_learning_rate']
        self.ppo_adam_epsilon = args_train['ppo_adam_epsilon']
        self.ppo_max_grad_norm = args_train['ppo_max_grad_norm']
        
        # 总更新轮数
        self.num_updates = args_train['num_updates']
        
        # rollout收集步数
        self.storage_num_steps = args_train['storage_num_steps']
        
        # 影子模式，使用纯文本进行训练
        self.shadow_mode = args_train['shadow_mode']
        
        self.checkpoints_save_path = args_train['checkpoints_save_path']

        self.debug = args_train['debug']