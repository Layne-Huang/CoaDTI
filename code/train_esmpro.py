import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from utilts import *

import random
import logging
from sklearn import metrics
import gc
import os

from CoaDTI_pro import *
# from CoaDTI_pro_visualize import *
# from apex import amp
import esm


model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()  # esm1_t6_43M_UR50S esm_msa1b_t12_100M_UR50S esm1b_t33_650M_UR50S
batch_converter = alphabet.get_batch_converter()


class Trainer(object):
    def __init__(self, model, batch_size, num_training_steps):
        self.model = model
        self.num_training_steps = num_training_steps
        self.optimizer_inner = optim.SGD(self.model.parameters(),
                                         lr=lr, weight_decay=weight_decay)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)

        # self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=0.2*self.num_training_steps, num_training_steps=self.num_training_steps)
        # self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
        # num_warmup_steps=0.1 * self.num_training_steps, num_training_steps = self.num_training_steps)
        self.batch_size = batch_size
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        # self.amp_handle = amp.init(self.model, self.optimizer)

    def train(self, dataloader, epoch, es):
        # np.random.shuffle(dataloader)
        N = len(dataloader)
        train_labels = []
        train_preds = []
        loss_total = 0
        tk = tqdm(dataloader, desc="Training epoch: " + str(epoch))
        for i, data in enumerate(tk):

            # print(data.y.shape)
            # print(data.protein.shape)

            # proteins = data.protein.view(int(data.y.size()[0]),1000,-1) # word2vec
            # print(len(data.protein[0]))
            # proteins = tokenizer(data.protein, padding="longest", truncation=True, return_tensors='pt')

            o = data.protein[0]
            # print(len(o))
            # print(o)
            proteins = [('', data.protein[0])]
            # print(proteins)
            _, _, proteins = batch_converter(proteins)
            # print(proteins)
            # print(proteins.size())
            # print(proteins['input_ids'].size())
            # print(data.x)
            # exit()

        # try:

            data, proteins = data.to(device), proteins.to(device)
            loss, logits = self.model(data, proteins)
            preds = logits.max(1)[1]
            self.optimizer.zero_grad()
            # with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()

            loss.backward()
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=5)
            self.optimizer.step()
            # self.scheduler.step()
            loss_total += loss.item()
            tk.set_postfix(
                {'loss': '%.6f' % float(loss_total / (i + 1)), 'LR': self.optimizer.param_groups[0]['lr'], 'ES':es})

            train_labels.extend(data.y.cpu())
            train_preds.extend(preds.cpu())
            # except:
            if np.isnan(loss_total):
                print(proteins.size())
                print(data)
                print(data.x)
                print(data.y)
                print(data.edge_index)
                print(o)
                exit()

            if i % 1000 == 0:
                del loss
                del preds
                gc.collect()

            torch.cuda.empty_cache()
        train_accu = metrics.accuracy_score(train_labels, train_preds)
        return loss_total, train_accu


class Tester(object):
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def test(self, dataset):
        N = len(dataset)
        # print(N)
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                # proteins = data.protein.view(int(data.y.size()[0]),1000,-1) #w2v
                # print(proteins.size())
                # proteins = data.protein.view(int(data.y.size()[0]),  -1)
                # proteins = tokenizer(data.protein, padding="longest", truncation=True, return_tensors='pt')
                proteins = [('', data.protein[0])]
                _, _, proteins = batch_converter(proteins)

                # (correct_labels, predicted_labels,
                #  predicted_scores,_,_) = self.model(data.to(device), proteins.to(device), train=False)
                (correct_labels, predicted_labels,
                 predicted_scores) = self.model(data.to(device), proteins.to(device), train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        # print(T)
        # print(Y)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        train_accu = metrics.accuracy_score(T, Y)
        AUC = roc_auc_score(T, S)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, precision, recall,train_accu, PRC

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a+') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def random_shuffle(dataset, seed):
    random.seed(seed)  # 2021 2345 1234
    random.shuffle(dataset)
    return dataset

def train(DATASET, fold, save_auc, co_attention, GCN_pooling, random_seed, log_write=False):

    # read data
    dir_input = ('../dataset/' + DATASET + '/input/''radius' + str(
        radius) + '_ngram' + str(ngram) + '/final/')

    setup_seed(random_seed)

    if fold==0:
        train_dataset = torch.load(dir_input + 'drug-target_train_esmpro_{}.pt'.format(MAX_LENGTH))
        dev_dataset = torch.load(dir_input + 'drug-target_dev_esmpro_{}.pt'.format(MAX_LENGTH))
        test_dataset = torch.load(dir_input + 'drug-target_test_esmpro_{}.pt'.format(MAX_LENGTH))
    else:
        train_dataset = torch.load(dir_input + 'drug-target_train_esmpro_{}_{}.pt'.format(MAX_LENGTH, fold))
        dev_dataset = torch.load(dir_input + 'drug-target_dev_esmpro_{}_{}.pt'.format(MAX_LENGTH, fold))
        test_dataset = torch.load(dir_input + 'drug-target_test_esmpro_{}_{}.pt'.format(MAX_LENGTH, fold))

    traindata_length = len(train_dataset)
    testdata_length = len(test_dataset)
    batch_size = 1
    # train_dataset = random_shuffle(train_dataset, 1234)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # dicdir_input = ('../dataset/' + '/dict/''radius' + str(
    #     radius) + '_ngram' + str(ngram) + '/')
    dicdir_input = ('../dataset/' + DATASET + '/input/''radius' + str(
    radius) + '_ngram' + str(ngram) + '/final/')
    fingerprint_dict = load_pickle(dicdir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dicdir_input + 'word_dict.pickle')


    # exit()
    n_fingerprint = len(fingerprint_dict)
    # print(n_fingerprint)
    # exit()
    n_word = len(word_dict)  # 100 #len(word_dict)+1

    print('n_word:{}'.format(n_word))

    """Set a model."""
    num_training_steps = len(train_loader) * iteration
    # model = Dtis(n_fingerprint, dim, n_word, window, layer_output, layer_coa).to(device)
    model = Dtis(n_fingerprint, dim, n_word, window, layer_output, layer_coa, co_attention=co_attention, gcn_pooling=GCN_pooling).to(device)
    trainer = Trainer(model, batch_size, num_training_steps)
    tester = Tester(model, batch_size)

    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
            'AUC_test\tPrecision_test\tRecall_test\tAcc_test\tPRC_test')
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    # logging info
    MODEL_NAME = 'SageGNN+{}+Co-attention'.format(pretrained_model)
    CO_ATTENTION = co_attention

    logging.info('DATASET: {}'.format(DATASET))
    logging.info('TRAIN_DATASET_LENGTH: {}'.format(traindata_length))
    logging.info('TEST_DATASET_LENGTH: {}'.format(testdata_length))
    logging.info('MAX_LENGTH: {}'.format(MAX_LENGTH))
    logging.info('RADIUS: {}'.format(radius))
    logging.info('LEARNING RATE: {}'.format(lr))
    logging.info('MODEL: {}'.format(MODEL_NAME))
    logging.info('CO-ATTENTIOM: {}'.format(CO_ATTENTION))
    logging.info('OPTIMIZER: {}'.format(optimizer))
    logging.info('BATCH_SIZE: {}'.format(batch_size))
    logging.info('MAX_EPOCHS: {}'.format(iteration))
    logging.info('COA_LAYERS: {}'.format(layer_coa))
    logging.info('fold: {}'.format(fold))

    best_auc = 0
    best_auc_dev = 0
    es = 0  # early stopping counter

    if GCN_pooling:
        pooling = 'True'
    else:
        pooling = 'False'
    log_header = 'CoaDTI Version:\nDATASET={}\n 1.ngram={}, radius={}\n2. position embedding\n' \
                 '3. {} attention. In particular, we use protein as Query, drug as Key and Value to feed into the module.\n' \
                 '   Use {}({}) and SGD optimizer.\n' \
                 '4. optimizer={}\n' \
                 '5. batch={}\n' \
                 '   we cut the protein length to {} and set batch=1 since we want to get sequence-form out put by GraphSgae for the following co-attention.\n' \
                 '6. learning rate={}\n' \
                 '7. fold={}\n' \
                 '8. random_seed={}\n' \
                 '9. gcn pooling: {}\n'.format(DATASET, ngram, radius, CO_ATTENTION, pretrained_model, pretrain, optimizer, batch_size,
                                       MAX_LENGTH, lr, fold, random_seed, pooling)

    if log_write:
        log_dir = '../log/' + DATASET + '/CoaDTIpro/{}/final'.format(
            pretrained_model)
        if fold == 0:
            file_name = 'radius{}_ngram{}_crosscoa{}_esmpro_batch{}_{}'.format(radius, ngram, layer_coa, batch_size,
                                                                                       MAX_LENGTH) + '.log'
            # file_name = 'Ablation_radius{}_ngram{}_crosscoa{}esmpro_prot{}'.format(radius, ngram, layer_coa,
            #                                                                            MAX_LENGTH) + '.log'

        # else:
        #     file_name = 'coaDTI_radius{}_ngram{}_crosscoa{}esmpro_batch{}_prot{}_fold{}'.format(radius, ngram, layer_coa, batch_size,
        #                                                                                       MAX_LENGTH,
        #                                                                                       fold) + '.log'
            # f.write('CoaDTI Version:\n1.ngram=1, radius=2\n2. position embedding\n'
            #         '3. encoder style cross attention. In particular, we use protein as Query, drug as Key and Value to feed into the module.\n'
            #         '4. batch=16 we do not cut the sequence\n'
            #         '5. fold={}\n'.format(fold))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        f_path = os.path.join(log_dir, file_name)
        with open(f_path, 'a+') as f:
            f.write(log_header)
    save_dir = '../output/model/' + DATASET + '/{}/coaDTI_radius{}_ngram{}_prot{}/final/'.format(pretrained_model, radius, ngram, MAX_LENGTH)
    for epoch in range(0, iteration):

        # if (epoch+1) % decay_interval == 0:
        #     trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train, train_accu = trainer.train(train_loader, epoch, es)
        # print(loss_train, train_accu)

        # if (epoch+1)%10==0:
        #     model_filename = ('{}-{}--{:.4f}.pkl'.format(DATASET, batch_size, epoch))
        #     model_path = os.path.join(save_dir, model_filename)
        #     torch.save(model.state_dict(), model_path)
        #     print(r'Saved the  model (epoch {}) to {}'.format(epoch, model_path))


        AUC_dev = tester.test(dev_loader)
        AUC_test, precision_test, recall_test, acc_test, PRC_test = tester.test(test_loader)

        end = timeit.default_timer()
        time = end - start
        #
        AUCs = [epoch, time, loss_train, train_accu, AUC_dev,
                AUC_test, precision_test, recall_test, acc_test, PRC_test]

        if log_write:
            tester.save_AUCs(AUCs, f_path)

        print('\t'.join(map(str, AUCs)))

        if AUC_test > best_auc:
            best_auc = AUC_test
            if best_auc > save_auc:
                save_dir = '../output/model/' + DATASET + '/{}/coaDTI_radius{}_ngram{}_prot{}/final/'.format(
                    pretrained_model, radius, ngram, MAX_LENGTH)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                if fold == 0:
                    model_filename = ('{}-{}--{}--{:.4f}.pkl'.format(DATASET, batch_size, co_attention, best_auc))

                else:
                    model_filename = ('{}-{}--fold{}--{:.4f}.pkl'.format(DATASET, batch_size, fold, best_auc))
                model_path = os.path.join(save_dir, model_filename)
                torch.save(model.state_dict(), model_path)
                print(r'Saved the new best model (valid auc: {}; test auc: {}) to {}'.format(AUC_dev, AUC_test,
                                                                                             model_path))
        # early stop mechanism
        if AUC_dev[0]>best_auc_dev:
            best_auc_dev = AUC_dev[0]
            es = 0
        elif AUC_dev[0]<=best_auc_dev:
            es+=1
            if es>15:
                print('Early stopping counter reaches to 90, the training will stop')
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')

    """Hyperparameters."""

    # AUCs=[1,2,3,4,5]
    # log_f.write('\t'.join(map(str, AUCs)))
    # exit()
    # (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
    #  lr, lr_decay, decay_interval, weight_decay, iteration,
    #  setting) = sys.argv[1:]
    # DATASET = 'human'
    DATASET = 'human'  # 'celegans' #'binding_DB'#'human' #'binding_DB'
    # DATASET=yourdata
    # log_f = open('../log/' + DATASET + '/sageGNN+Prot_Bert_coa_900.log', 'a+')
    # radius=1
    radius = 2
    # radius=3

    # ngram=2
    ngram = 1
    MAX_LENGTH = 4000
    # dim = 90
    dim = 512  # 90 # 512 # For co-attention
    layer_gnn = 3
    side = 5
    window = 2 * side + 1
    layer_output = 3
    layer_coa = 1
    lr = 5e-2  # 4e-3
    lr_decay = 0.5
    decay_interval = 20
    weight_decay = 1e-4
    iteration = 50
    optimizer = 'lookahead-SGD'
    pretrain = 'Yes'
    pretrained_model = 'esm1_t6_43M_UR50S'
    (dim, layer_gnn, window, layer_output, layer_coa, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_output, layer_coa,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    config = {'radius':radius, 'ngram':ngram, 'MAX_LENGTH':MAX_LENGTH, 'dim':dim, 'layer_gnn':layer_gnn
                 , 'window':window, 'layer_output':layer_output, 'layer_coa':layer_coa, 'decay_interval':decay_interval,
     'iteration':iteration, 'lr':lr, 'lr_decay':lr_decay, 'weight_decay':weight_decay, 'optimizer':optimizer, 'pretrain':pretrain, 'pretrained_model':pretrained_model}
    """CPU or GPU."""
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        device = torch.device('cuda')
        #device = torch.device('cpu')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')


    seed_list = [1, 2021, 1234]
    attention_list = ['stack', 'encoder']
    datasets = ['human', 'celegans']
    DATASET = 'binding_DB'
    # train('binding_DB', 0, 0.93, 'inter', False, 1234, False)
    # exit()
    for dataset in datasets[1:]:
        for attention in attention_list:
            if 'inter' in attention:
                train(dataset, 0, 0.98, attention, False, 1, True)
            else:
                train(dataset, 0, 0.98, attention, False, 1, True)

    for dataset in datasets:
        for attention in attention_list:
            if 'inter' in attention:
                train(dataset, 0, 0.98, attention, False, 1, True)
            else:
                train(dataset, 0, 0.98, attention, True, 1, True)

    for dataset in datasets[1:]:
        for attention in attention_list:
            if 'inter' in attention:
                train(dataset, 0, 0.98, attention, False, 1234, True)
            else:
                train(dataset, 0, 0.98, attention, False, 1234, True)


