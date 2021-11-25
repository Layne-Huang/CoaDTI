import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
from tqdm import tqdm
# from model import *
from coaDTI import *
import random
import logging
from sklearn import metrics
import gc
import os
from utilts import *
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup


class Trainer(object):
    def __init__(self, model, batch_size, num_training_steps):
        self.model = model
        self.num_training_steps = num_training_steps
        self.optimizer_inner = optim.SGD(self.model.parameters(),
                                         lr=lr, weight_decay=weight_decay)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5)

        # self.scheduler = get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=0.2*self.num_training_steps, num_training_steps=self.num_training_steps)
        # self.scheduler = get_constant_schedule_with_warmup(self.optimizer,
        #                                                    num_warmup_steps=0.1 * self.num_training_steps)
        self.batch_size = batch_size
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
        # self.amp_handle = amp.init(self.model, self.optimizer)

    def train(self, dataloader,epoch, es):
        # np.random.shuffle(dataloader)
        N = len(dataloader)
        train_labels=[]
        train_preds=[]
        loss_total = 0
        tk = tqdm(dataloader, desc="Training epoch: " + str(epoch))

        for i, data in enumerate(tk):

            # print(data.y.shape)
            # print(data.protein.shape)


            # proteins = data.protein.view(int(data.y.size()[0]),1000,-1) # word2vec
            proteins = data.protein.view(int(data.y.size()[0]), -1)

            # print(proteins.size())
            # print(proteins['input_ids'].size())

            # try:

            data, proteins = data.to(device), proteins.to(device)
            loss, logits = self.model(data, proteins)
            preds = logits.max(1)[1]

            # with self.amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()

            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters=self.model.parameters(), max_norm=5)
            self.optimizer.step()
            # self.scheduler.step()
            loss_total += loss.item()
            tk.set_postfix(
                {'loss': '%.6f' % float(loss_total / (i + 1)), 'LR': self.optimizer.param_groups[0]['lr'], 'ES': es})

            train_labels.extend(data.y.cpu())
            train_preds.extend(preds.cpu())
            # except:
            #     print(proteins.size())

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
                proteins = data.protein.view(int(data.y.size()[0]),  -1)
                # proteins = data.protein
                (correct_labels, predicted_labels,
                 predicted_scores) = self.model(data.to(device), proteins.to(device), train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)

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

def train(DATASET, fold, save_auc, co_attention, ablation, GCN_pooling, random_seed, scarce, log_write=False):

    # read data
    setup_seed(random_seed)
    if 'binding' in DATASET:
        dir_input = ('../dataset/' + DATASET + '/input/''radius' + str(
            radius) + '_ngram' + str(ngram) + '/final/')
        dicdir_input = ('../dataset/' + '/dict/radius' + str(
        radius) + '_ngram' + str(ngram) + '/')
    else:
        dir_input = ('../dataset/' + DATASET + '/input/final/radius' + str(
            radius) + '_ngram' + str(ngram)+'/')
        dicdir_input = dir_input

    # if batch and fold
    # train_dataset = torch.load(dir_input+'drug-target_train_{}_{}.pt'.format(MAX_LENGTH,fold))
    # dev_dataset = torch.load(dir_input + 'drug-target_dev_{}_{}.pt'.format(MAX_LENGTH,fold))
    # test_dataset = torch.load(dir_input + 'drug-target_test_{}_{}.pt'.format(MAX_LENGTH,fold))


    # if no fold
    # train_dataset = torch.load(dir_input+'drug-target_train_{}.pt'.format(MAX_LENGTH))
    # dev_dataset = torch.load(dir_input + 'drug-target_dev_{}.pt'.format(MAX_LENGTH))
    # test_dataset = torch.load(dir_input + 'drug-target_test_{}.pt'.format(MAX_LENGTH))
    # train_dataset = torch.load(dir_input+'drug-target_train_4000.pt')
    # dev_dataset = torch.load(dir_input + 'drug-target_dev_4000.pt')
    # test_dataset = torch.load(dir_input + 'drug-target_test_4000.pt')

    # if batch==1
    # train_dataset = torch.load(dir_input+'drug-target_train_'+str(fold)+'.pt')
    # dev_dataset = torch.load(dir_input + 'drug-target_dev_'+str(fold)+'.pt')
    # test_dataset = torch.load(dir_input + 'drug-target_test_'+str(fold)+'.pt')

    # if use w2v
    # train_dataset = torch.load(dir_input+'drug-target_train_w2v_'+str(fold)+'.pt')
    # dev_dataset = torch.load(dir_input + 'drug-target_dev_w2v_'+str(fold)+'.pt')
    # test_dataset = torch.load(dir_input + 'drug-target_test_w2v_'+str(fold)+'.pt')

    # no-radius+no batch
    if DATASET:
        train_dataset = torch.load(dir_input + 'drug-target_train_{}.pt'.format(MAX_LENGTH))
        dev_dataset = torch.load(dir_input + 'drug-target_dev_{}.pt'.format(MAX_LENGTH))
        test_dataset = torch.load(dir_input + 'drug-target_test_{}.pt'.format(MAX_LENGTH))
    # else:
    #     train_dataset = torch.load(dir_input + 'drug-target_train_esmpro_{}_{}.pt'.format(MAX_LENGTH, fold))
    #     dev_dataset = torch.load(dir_input + 'drug-target_dev_esmpro_{}_{}.pt'.format(MAX_LENGTH, fold))
    #     test_dataset = torch.load(dir_input + 'drug-target_test_esmpro_{}_{}.pt'.format(MAX_LENGTH, fold))

    traindata_length = len(train_dataset)
    testdata_length = len(test_dataset)
    batch_size = 1
    # scarce_slice = int(len(train_dataset)/int(scarce*len(train_dataset)))
    train_dataset = random_shuffle(train_dataset, 1234)
    train_loader = DataLoader(train_dataset[:int(scarce*len(train_dataset))], batch_size=batch_size, shuffle=True) # [0:18000:18]
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False) # [:2000:20]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # [:5000:25]

    fingerprint_dict = load_pickle(dicdir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dicdir_input + 'word_dict.pickle')

    n_fingerprint = len(fingerprint_dict)

    n_word = len(word_dict)  # 100 #len(word_dict)+1

    print('n_word:{}'.format(n_word))

    """Set a model."""
    CO_ATTENTION =  co_attention #'stack' #'encoder-decoder'  # 'intersection'

    num_training_steps = len(train_loader) * iteration
    if 'False' in ablation:
        print('----------')
        model = Dtis(n_fingerprint, dim, n_word, layer_output, layer_coa, co_attention=CO_ATTENTION, gcn_pooling=GCN_pooling).to(device)
    else:
        if 'attention' in ablation:
            model = Dtis_ablation(n_fingerprint, dim, n_word, layer_output, layer_coa, gcn_pooling=GCN_pooling).to(
                device)
        elif 'trans' in ablation:
            model = Dtis_ablation_trans(n_fingerprint, dim, n_word, layer_output, layer_coa, gcn_pooling=GCN_pooling, co_attention=CO_ATTENTION).to(
                device)
        elif 'gcn' in ablation:
            model = Dtis_ablation_gnn(n_fingerprint, dim, n_word, layer_output, layer_coa, co_attention=CO_ATTENTION).to(device)
    trainer = Trainer(model, batch_size, num_training_steps)
    tester = Tester(model, batch_size)

    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\t'
            'AUC_test\tPrecision_test\tRecall_test\tAcc_test\tPRC_test')
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    # logging info
    MODEL_NAME = 'GraphSage+{}+Co-attention'.format('transformer')

    logging.info('DATASET: {}'.format(DATASET))
    logging.info('TRAIN_DATASET_LENGTH: {}'.format(traindata_length))
    logging.info('TEST_DATASET_LENGTH: {}'.format(testdata_length))
    logging.info('MAX_LENGTH: {}'.format(MAX_LENGTH))
    logging.info('LEARNING RATE: {}'.format(lr))
    logging.info('MODEL: {}'.format(MODEL_NAME))
    logging.info('CO-ATTENTIOM: {}'.format(CO_ATTENTION))
    logging.info('OPTIMIZER: {}'.format(optimizer))
    logging.info('BATCH_SIZE: {}'.format(batch_size))
    logging.info('MAX_EPOCHS: {}'.format(iteration))
    logging.info('COA_LAYERS: {}'.format(layer_coa))
    logging.info('fold: {}'.format(fold))
    logging.info('random_seed: {}'.format(random_seed))

    best_auc = 0
    best_auc_dev = 0
    es = 0  # early stopping counter

    log_header = 'CoaDTI Version:\nDATASET={}\n 1.ngram={}, radius={}\n2. position embedding\n' \
                 '3. {} attention. In particular, we use protein as Query, drug as Key and Value to feed into the module.\n' \
                 '   Use {}({}) and SGD optimizer.\n' \
                 '4. optimizer={}\n' \
                 '5. batch={}\n' \
                 '   we cut the protein length to {} and set batch=1 since we want to get sequence-form out put by GraphSgae for the following co-attention.\n' \
                 '6. learning rate={}\n' \
                 '7. random_seed={}\n' \
                 '8. ablation: {}\n'\
                 '9. gcn pooling: False\n'.format(DATASET, ngram, radius, pretrained_model, CO_ATTENTION, pretrain, optimizer, batch_size,
                                       MAX_LENGTH, lr, random_seed, ablation)

    if log_write:
        log_dir = '../log/' + DATASET + '/CoaDTI/final/scarce/'
        if fold == 0:
            # file_name = 'coaDTI_radius2_ngram1_{}_crosscoa{}esmpro_batch{}_prot{}'.format(CO_ATTENTION, layer_coa, batch_size,
            #                                                                            MAX_LENGTH) + '.log'
            file_name = 'scarce_radius{}_ngram{}_{}_{}'.format(radius, ngram, MAX_LENGTH,str((1-scarce)*100)) + '.log'

        else:
            file_name = 'coaDTI_radius2_ngram1_crosscoa{}esmpro_batch{}_prot{}_fold{}'.format(layer_coa, batch_size,
                                                                                              MAX_LENGTH,
                                                                                              fold) + '.log'
            # f.write('CoaDTI Version:\n1.ngram=1, radius=2\n2. position embedding\n'
            #         '3. encoder style cross attention. In particular, we use protein as Query, drug as Key and Value to feed into the module.\n'
            #         '4. batch=16 we do not cut the sequence\n'
            #         '5. fold={}\n'.format(fold))
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        f_path = os.path.join(log_dir, file_name)
        with open(f_path, 'a+') as f:
            f.write(log_header)

    for epoch in range(0, iteration):

        # if (epoch+1) % decay_interval == 0:
        #     trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train, train_accu = trainer.train(train_loader, epoch, es)

        # print(loss_train)
        # print(train_dataset[0])
        # print(len(dev_loader))
        # print(dev_dataset[0])
        # print(len(test_loader))
        # print(test_dataset[0])
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
            print('Get higher performance')
            if best_auc > save_auc:
                save_dir = '../output/model/' + DATASET + '/CoaDTI/coaDTI_radius2_ngram1_{}_{}'.format(CO_ATTENTION, MAX_LENGTH)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

                if fold == 0:
                    model_filename = ('{}-{}--{:.4f}.pkl'.format(DATASET, batch_size, best_auc))

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
            if es>5:
                print('Early stopping counter reaches to 5, the training will stop')
                break

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')
    # random_seed = 1

    """Hyperparameters."""


    DATASET = 'celegans'  # 'celegans' #'binding_DB'#'human' #'binding_DB'

    radius = 2
    ngram = 1
    MAX_LENGTH = 1000
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
    iteration = 20
    optimizer = 'lookahead-SGD'
    pretrain = 'None'
    pretrained_model = 'None'
    co_attention = 'intersection' #'encoder-decoder'  # 'intersection'
    GCN_pooling=False


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
        # device = torch.device('cpu')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')




        # log_f.write('\t'.join(map(str, AUCs)).join('\n'))

        # -------------------------------------------------


    # for fold in range(2,4):
    #     DATASET='celegans'
    #     train(DATASET,fold)

    # for fold in range(3,4):
    #     DATASET='celegans'
    #     train(DATASET,fold,1, False)
    # range(1, 4)
    # for fold in range(1, 4):
    #     DATASET = 'celegans'
    #     train(DATASET, fold, 1, log_write=False)
    #     # train('celegans', fold, 1, log_write=True)
    #     # train('binding_DB', fold, 1, log_write=True)


    seed_list = [1,2021, 1234]
    ablation = ['co-attention','transformer','gcn']
    scarce_list = [0.05,0.01]
    # train('celegans', 0, 0.97, 'inter', 'transformer', GCN_pooling, 2021, True)
    train('binding_DB', 0, 0.97, 'inter', 'False', GCN_pooling, 1234, 0.01, True)
    # for scarce in scarce_list[2:]:
    #     for random_seed in seed_list[2:]:
    #         print(scarce)
    #         train('binding_DB', 0, 0.97, 'inter', 'False', GCN_pooling, random_seed, scarce, True)





