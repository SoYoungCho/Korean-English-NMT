import torch
from torch import optim
import torch.nn as nn

from data_loader import DataLoader
import data_loader

from hyperparams import TrainHyperparams
from simple_nmt.seq2seq import Seq2Seq
#from simple_nmt.transformer import Transformer
from simple_nmt.rnnlm import LanguageModel

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def main(config, model_weight=None, opt_weight=None):
    print("==== train.py main =====")
    def print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    if config.dsl:
        print("==== train.py config.dsl => Dataloader 실행 =====")
        loader = DataLoader(
            config.train,
            config.valid,
            (config.lang[:12], config.lang[12:]),
            batch_size=config.lm_batch_size,
            device=config.gpu_id,
            max_length=config.max_length,
            dsl=config.dsl,
        )

        from simple_nmt.lm_trainer import LanguageModelTrainer as LMTrainer
        print("==== train.py language_model 2개 실행 =====")
        language_models = [
            LanguageModel(
                len(loader.tgt.vocab),
                config.word_vec_size,
                config.hidden_size,
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
            LanguageModel(
                len(loader.src.vocab),
                config.word_vec_size,
                config.hidden_size,
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
        ]
        print("==== train.py language_model 2개 실행 end =====")
        print("==== train.py models (seq2seq) 2개 실행 =====")
        models = [
            Seq2Seq(
                len(loader.src.vocab),
                config.word_vec_size,
                config.hidden_size,
                len(loader.tgt.vocab),
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
            Seq2Seq(
                len(loader.tgt.vocab),
                config.word_vec_size,
                config.hidden_size,
                len(loader.src.vocab),
                n_layers=config.n_layers,
                dropout_p=config.dropout,
            ),
        ]
        print("==== train.py models (seq2seq) 2개 실행 end =====")

        loss_weights = [
            torch.ones(len(loader.tgt.vocab)),
            torch.ones(len(loader.src.vocab)),
        ]
        loss_weights[0][data_loader.PAD] = .0
        loss_weights[1][data_loader.PAD] = .0

        crits = [
            nn.NLLLoss(weight=loss_weights[0], reduction='none'),
            nn.NLLLoss(weight=loss_weights[1], reduction='none'),
        ]

        print(language_models)
        print(models)
        print(crits)

        if model_weight is not None:
            print("train.py - if model_weight is not None: 에 걸렸다!")
            for model, w in zip(models + language_models, model_weight):
                model.load_state_dict(w)
            print("model의 정체는..? ", model)

        if config.gpu_id >= 0:
            for lm, seq2seq, crit in zip(language_models, models, crits):
                lm.cuda(config.gpu_id)
                seq2seq.cuda(config.gpu_id)
                crit.cuda(config.gpu_id)

        for lm, crit in zip(language_models, crits):
            print("==== train.py for문 lm 모델 하나씩 실행 =====")
            optimizer = optim.Adam(lm.parameters())
            lm_trainer = LMTrainer(config)

            lm_trainer.train(
                lm, crit, optimizer,
                train_loader=loader.train_iter,
                valid_loader=loader.valid_iter,
                src_vocab=loader.src.vocab if lm.vocab_size == len(loader.src.vocab) else None,
                tgt_vocab=loader.tgt.vocab if lm.vocab_size == len(loader.tgt.vocab) else None,
                n_epochs=config.lm_n_epochs,
            )
            print("==== train.py for문 lm 모델 하나씩 실행 =====")


        loader = DataLoader(
            config.train,
            config.valid,
            (config.lang[:12], config.lang[12:]),
            batch_size=config.batch_size,
            device=config.gpu_id,
            max_length=config.max_length,
            # dsl=config.dsl
        )

        from simple_nmt.dual_trainer import DualSupervisedTrainer as DSLTrainer
        dsl_trainer = DSLTrainer(config)

        optimizers = [
            optim.Adam(models[0].parameters()),
            optim.Adam(models[1].parameters()),
        ]

        if opt_weight is not None:
            for opt, w in zip(optimizers, opt_weight):
                opt.load_state_dict(w)

        dsl_trainer.train(
            models,
            language_models,
            crits,
            optimizers,
            train_loader=loader.train_iter,
            valid_loader=loader.valid_iter,
            vocabs=[loader.src.vocab, loader.tgt.vocab],
            n_epochs=config.n_epochs + config.dsl_n_epochs,
            lr_schedulers=None,
    )
    else:
        loader = DataLoader(
            config.train,
            config.valid,
            (config.lang[:12], config.lang[12:]),
            batch_size=config.batch_size,
            device=config.gpu_id,
            max_length=config.max_length,
            dsl=config.dsl
        )

        #from simple_nmt.trainer import MaximumLikelihoodEstimationTrainer as MLETrainer
        # Encoder's embedding layer input size
        input_size = len(loader.src.vocab)
        # Decoder's embedding layer input size and Generator's softmax layer output size
        output_size = len(loader.tgt.vocab)
        # Declare the model
        # if config.use_transformer:
        #     model = Transformer(
        #         input_size,
        #         config.hidden_size,
        #         output_size,
        #         n_splits=config.n_splits,
        #         n_enc_blocks=config.n_layers,
        #         n_dec_blocks=config.n_layers,
        #         dropout_p=config.dropout,
        #     )
        # else:
        model = Seq2Seq(input_size,
                        config.word_vec_size,  # Word embedding vector size
                        config.hidden_size,  # LSTM's hidden vector size
                        output_size,
                        n_layers=config.n_layers,  # number of layers in LSTM
                        dropout_p=config.dropout  # dropout-rate in LSTM
                        )

        # Default weight for loss equals to 1, but we don't need to get loss for PAD token.
        # Thus, set a weight for PAD to zero.
        loss_weight = torch.ones(output_size)
        loss_weight[data_loader.PAD] = 0.
        # Instead of using Cross-Entropy loss,
        # we can use Negative Log-Likelihood(NLL) loss with log-probability.
        crit = nn.NLLLoss(weight=loss_weight,
                          reduction='sum'
                          )

        print(model)
        print(crit)

        if model_weight is not None:
            model.load_state_dict(model_weight)

        # Pass models to GPU device if it is necessary.
        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
            crit.cuda(config.gpu_id)

        if config.use_adam:
            if config.use_transformer:
                # optimizer = optim.Adam(model.parameters(), lr=config.hidden_size**(-.5), betas=(.9, .98))
                optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))
            else: # case of rnn based seq2seq.
                optimizer = optim.Adam(model.parameters(), lr=config.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=config.lr)

        if opt_weight is not None and config.use_adam:
            optimizer.load_state_dict(opt_weight)

        if config.use_noam_decay:
            f = lambda step: min((step + 1)**(-.5), (step + 1) * config.lr_n_warmup_steps**(-1.5))
            lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
        else:
            if config.lr_step > 0:
                lr_scheduler = optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[i for i in range(max(0, config.lr_decay_start - 1),
                                                 (config.init_epoch - 1) + config.n_epochs,
                                                 config.lr_step)],
                    gamma=config.lr_gamma
                )

                for _ in range(config.init_epoch - 1):
                    lr_scheduler.step()
            else:
                lr_scheduler = None

        print(optimizer)

    # Start training. This function maybe equivalant to 'fit' function in Keras.
    # mle_trainer = MLETrainer(config)
    # mle_trainer.train(
    #     model,
    #     crit,
    #     optimizer,
    #     train_loader=loader.train_iter,
    #     valid_loader=loader.valid_iter,
    #     src_vocab=loader.src.vocab,
    #     tgt_vocab=loader.tgt.vocab,
    #     n_epochs=config.n_epochs,
    #     lr_scheduler=lr_scheduler,
    # )
"""
        if config.rl_n_epochs > 0:
            optimizer = optim.SGD(model.parameters(), lr=config.rl_lr)

            from simple_nmt.rl_trainer import MinimumRiskTrainer
            mrt_trainer = MinimumRiskTrainer(config)

            mrt_trainer.train(
                model,
                crit,
                optimizer,
                train_loader=loader.train_iter,
                valid_loader=loader.valid_iter,
                src_vocab=loader.src.vocab,
                tgt_vocab=loader.tgt.vocab,
                n_epochs=config.rl_n_epochs,
            )
"""

if __name__ == '__main__':
    #config = define_argparser()
    config = TrainHyperparams()
    main(config)
