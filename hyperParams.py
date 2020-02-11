import logging, sys
logger = logging.getLogger('root')
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT)
logger.setLevel(logging.INFO)

class HyperParams():
    """
    Set of Hyperparameters
    Hyperparameters:
        - **use_bidirectional**: if True, becomes a bidirectional encoder
        - **use_attention**: flag indication whether to use attention mechanism or not
        - **input_reverse**: flag indication whether to reverse input feature or not
        - **use_pickle**: flag indication whether to load data from pickle or not
        - **use_augment**: flag indication whether to use spec-augmentation or not
        - **augment_ratio**: ratio of spec-augmentation applied data
        - **encoder_layer_size**: num of encoder`s RNN cell
        - **decoder_layer_size**: num of decoder`s RNN cell
        - **hidden_size**: size of hidden state of RNN
        - **dropout**: dropout probability
        - **batch_size**: mini-batch size
        - **worker_num**: num of cpu core will be used
        - **max_epochs**: max epoch
        - **lr**: learning rate
        - **teacher_forcing**: The probability that teacher forcing will be used
        - **seed**: seed for random
        - **max_len**: a maximum allowed length for the sequence to be processed
        - **no_cuda**: if True, don`t use CUDA
        - **save_name**: save name of model
        - **mode**: train or test
        - **load_model**: flag indication whether to load weight file or not
        - **model_path**: path for weight file
    """
    def __init__(self):
        self.use_bidirectional = True
        self.use_attention = True
        self.input_reverse = True
        self.use_augment = False
        self.use_pickle = True
        #self.use_pyramidal = True
        self.use_cuda = False
        self.augment_ratio = 0.4
        self.hidden_size = 64
        self.dropout = 0.5
        self.encoder_layer_size = 3
        self.decoder_layer_size = 1
        self.batch_size = 4
        self.worker_num = 1
        self.max_epochs = 40
        self.lr = 0.0001
        self.teacher_forcing = 0.0
        self.seed = 1
        self.max_len = 80
        self.save_name = 'model'
        self.mode = 'train'
        self.load_model = False
        self.model_path = ""

    def logger_hparams(self):
        logger.info("use_bidirectional : %s" % str(self.use_bidirectional))
        logger.info("use_attention : %s" % str(self.use_attention))
        logger.info("use_pickle : %s" % str(self.use_pickle))
        logger.info("use_augment : %s" % str(self.use_augment))
        logger.info("use_pyramidal : %s" % str(self.use_pyramidal))
        logger.info("augment_ratio : %0.2f" % self.augment_ratio)
        logger.info("input_reverse : %s" % str(self.input_reverse))
        logger.info("hidden_size : %d" % self.hidden_size)
        logger.info("encoder_layer_size : %d" % self.encoder_layer_size_layer_size)
        logger.info("decoder_layer_size : %d" % self.decoder_layer_sizeer_layer_size)
        logger.info("dropout : %0.2f" % self.dropout)
        logger.info("batch_size : %d" % self.batch_size)
        logger.info("worker_num : %d" % self.worker_num)
        logger.info("max_epochs : %d" % self.max_epochs)
        logger.info("learning rate : %0.4f" % self.lr)
        logger.info("teacher_forcing_ratio : %0.2f" % self.teacher_forcing)
        logger.info("seed : %d" % self.seed)
        logger.info("max_len : %d" % self.max_len)
        logger.info("use_cuda : %s" % str(self.use_cuda))
        logger.info("save_name : %s" % self.save_name)
        logger.info("mode : %s" % self.mode)