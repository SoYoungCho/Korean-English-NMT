#import define

class TrainHyperparams():
    def __init__(self,
                 model_fn = 'lm.model.pt',
                 train = "C:/Users/Soyoung Cho/Desktop/Test/test",
                 valid = "C:/Users/Soyoung Cho/Desktop/Test/test",
                 lang = 'train_ko.csvtrain_en.csv',
                 gpu_id = -1,
                 batch_size =  5,
                 n_epochs = 2,
                 verbose = 2,
                 init_epoch = 1,
                 max_length = 80,
                 dropout = .2,
                 word_vec_size = 512,
                 hidden_size = 768,
                 n_layers = 2,
                 max_grad_norm = 5.,
                 use_adam = True,
                 lr = 1.,
                 lr_step = 1,
                 lr_gamma = .5,
                 lr_decay_start = 10,
                 use_noam_decay = True,
                 lr_n_warmup_steps = 48000,
                 #rl_lr = .01,
                 #rl_n_samples = 1,
                 #rl_n_epochs = 10,
                 # rl_init_epoch =  1,
                 # rl_n_gram = 6,
                 dsl = True,
                 lm_n_epochs = 2,
                 lm_batch_size = 32,
                 dsl_n_epochs = 2,
                 dsl_lambda = 1e-3,
                 use_transformer = False,
                 n_splits = 8,
                 use_cuda = -1
                 ):

        self.model_fn = model_fn
        self.train = train
        self.valid = valid
        self.lang = lang
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.init_epoch = init_epoch
        self.max_length = max_length
        self.dropout = dropout
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.max_grad_norm = max_grad_norm
        self.use_adam = use_adam
        self.lr = lr
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma
        self.lr_decay_start = lr_decay_start
        self.use_noam_decay = use_noam_decay
        self.lr_n_warmup_steps = lr_n_warmup_steps
        #self.rl_lr = rl_lr,
        #self.rl_n_samples = rl_n_samples,
        #self.rl_n_epochs = rl_n_epochs,
        #self.rl_init_epoch = rl_init_epoch
        #self.rl_n_gram = rl_n_gram
        self.dsl = dsl
        self.lm_n_epochs = lm_n_epochs
        self.lm_batch_size = lm_batch_size
        self.dsl_n_epochs = dsl_n_epochs
        self.dsl_lambda = dsl_lambda
        self.use_transformer = use_transformer
        self.n_splits = n_splits
        self.use_cuda = use_cuda

class TranslateHyperparams():
    def __init__(self,
                 model = 'dsl.model.pt',
                 gpu_id = -1,
                 batch_size = 2,
                 max_length = 255,
                 n_best = 1,
                 beam_size = 5,
                 lang = "korsample.csvengsample.csv",
                 length_penalty = 1.2
                 ):
        self.model = model
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.n_best = n_best
        self.beam_size = beam_size
        self.lang = lang
        self.length_penalty = length_penalty