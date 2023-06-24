import math
import random
import time

import nltk
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import sklearn
from nltk.translate.meteor_score import meteor_score
from torch import nn
import torch
from torch.optim import Adam,SGD,RMSprop,Adagrad,Adadelta,Adamax,AdamW
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils import shuffle
import DataPreprocessing
from EarlyStopper import EarlyStopper
import torch.cuda.amp.autocast_mode
import torch.cuda.amp.grad_scaler
import nltk
nltk.download('wordnet')

#from torchmetrics import BLEUScore


class Seq2Seq(nn.Module):


    def init_optimizer(self, optimizer_name=None, learning_rate=None, momentum=None, weight_decay=None):
        if optimizer_name == "adam":
            self.opt = Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "SGD":
            self.opt = SGD(self.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "RMSprop":
            self.opt = RMSprop(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "Adagrad":
            self.opt = Adagrad(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "AdanW":
            self.opt = AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "Adamax":
            self.opt = Adamax(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def init_loss_function(self, function_name):
        if function_name == "cross-entropy":
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token)
        elif function_name == "mse":
            self.criterion = nn.MSELoss(ignore_index=self.pad_token)
        elif function_name == "L1":
            self.criterion = nn.L1Loss(ignore_index=self.pad_token)
        elif function_name == "binary-cross-entropy":
            self.criterion = nn.BCELoss(ignore_index=self.pad_token)
        elif function_name == "neg-log-likelihood":
            self.criterion = nn.NLLLoss(ignore_index=self.pad_token)


    def __init__(self, encoder, decoder, device, MAX_LENGTH=100,teacher_forcing_ratio=0.5,optimizer="RMSprop",learning_rate=0.001,momentum=0.9,weight_decay=1e-05,loss_function="cross-entropy",batch_size=32,early_stop=False,waiting=10,min_delta=0):
        super().__init__()

        self.sos_token = 0
        self.eos_token = 1
        self.pad_token = 2
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.init_optimizer(optimizer_name=optimizer, learning_rate=learning_rate, momentum=momentum,
                            weight_decay=weight_decay)
        self.init_loss_function(loss_function)
        self.batch_size = batch_size
        self.verbose_levels = [0, 1, 10, 100, 1000]
        self.History = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "epoch_time": [],"perplexity":[],"val_perplexity":[],"meteor":[],"val_meteor":[]}

        self.early_stop = early_stop
        if self.early_stop:
            self.early_stopper = EarlyStopper(waiting=waiting, mind=min_delta)



    def forward(self, source, target):

        batch_size = source.shape[1]
        target_length = target.shape[0]
        vocab_size = self.decoder.output_size

        # initialize a variable to hold the predicted outputs
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        hidden,cell = self.encoder(source)


        x = target[0]
        for t in range(1,target_length):
            decoder_output,decoder_hidden,decoder_cell = self.decoder(x,hidden,cell)

            outputs[t] = decoder_output
            best_guess = decoder_output.argmax(1)

            x = target[t] if random.random() < self.teacher_forcing_ratio else best_guess

        return outputs

    def index_sentences(self,pairs,en_w2i,rus_w2i):
        indexed_pairs = []
        for pair in pairs:
            indexed_pairs.append(DataPreprocessing.transform_to_index_tensor(pair,rus_w2i,en_w2i,self.device))
        return indexed_pairs

    def FindMaxLength(self,lst):
        maxLength = max(len(x) for x in lst)
        return maxLength

    def pad_zero(self,l, content, width):
        l.extend([content] * (width - len(l)))
        return l

    def create_batches(self,x):
        x = np.array(x)
        m = x.shape[0]
        num_batches = m / self.batch_size
        batches = []
        for i in range(int(num_batches + 1)):
            batch_x = x[i * self.batch_size:(i + 1) * self.batch_size]
            batches.append(batch_x.tolist())

        if m % self.batch_size == 0:
            batches.pop(-1)

        return batches

    def print_progress(self, phase, accuracy, loss,perplexity,meteor, current_epoch, verbose):
        if current_epoch % self.verbose_levels[verbose] == 0:
            print("Phase " + str(phase) + " - Bleu: " + str(round(accuracy*100,2))+" - Meteor: "+str(round(meteor*100,2)) +" - Perplexity: "+str(perplexity)+ " - " + "loss: " + str(loss))

    def pad(self,batches):
        target_padded = []
        source_padded = []
        for batch in batches:
            temp_source = []
            temp_target = []
            for pair in batch:
                temp_source.append(pair[0])
                temp_target.append(pair[1])
            source_max_len = self.FindMaxLength(temp_source)
            target_max_len = self.FindMaxLength(temp_target)

            max_len = max(source_max_len,target_max_len)

            target_padded_batch = []
            source_padded_batch = []
            for pair in batch:
                source_padded_batch.append(self.pad_zero(pair[0],2,max_len))
                target_padded_batch.append(self.pad_zero(pair[1], 2, max_len))
            target_padded.append(target_padded_batch)
            source_padded.append(source_padded_batch)

        return source_padded,target_padded

    def remove_tokens(self,target,pred):

        #based on <pad> tokens of the target sentence remove the same indexes from the predicted
        target_itemindex_pad = np.array(np.where(target == '<PAD>'))
        if len(target_itemindex_pad) > 0:
            target = np.delete(target, target_itemindex_pad)
            pred = np.delete(pred, target_itemindex_pad)


        target_itemindex_eos = np.array(np.where(target == '<EOS>'))
        target_itemindex_sos = np.array(np.where(target == '<SOS>'))
        if len(target_itemindex_eos) > 0:
            target = np.delete(target, target_itemindex_eos)
        if len(target_itemindex_sos) > 0:
            target = np.delete(target, target_itemindex_sos)

        pred_itemindex_eos = np.array(np.where(pred == '<EOS>'))
        pred_itemindex_sos = np.array(np.where(pred == '<SOS>'))
        if len(pred_itemindex_eos) > 0:
            pred = np.delete(pred, pred_itemindex_eos)
        if len(pred_itemindex_sos) > 0:
            pred = np.delete(pred, pred_itemindex_sos)

        return target,pred

    def get_ngram_order(self,sentence):
        length = len(sentence)
        if length <= 10:
            ngram_order = 1
        elif length <= 20:
            ngram_order = 2
        elif length <= 30:
            ngram_order = 3
        else:
            ngram_order = math.ceil(length / 5)
        return ngram_order

    def calculate_metric(self,pred,target,index_to_word_target,debug):
        pred = torch.transpose(pred, 0, 1)
        target = torch.transpose(target, 0, 1)
        pred = pred.detach().cpu().numpy()
        target = target.cpu().numpy()
        number_of_sentences = target.shape[0]
        score_bleu = 0
        score_meteor = 0
        for i in range(len(target)):
            pred_sentence = np.array([])
            target_sentence = np.array([])
            for j in range(len(target[i])):
                target_sentence = np.append(target_sentence,[index_to_word_target[target[i][j]]],axis=0)
                pred_sentence = np.append(pred_sentence, [index_to_word_target[pred[i][j].argmax(axis=0)]], axis=0)
            target_sentence,pred_sentence = self.remove_tokens(target_sentence,pred_sentence)
            ngram_order = max(self.get_ngram_order(target_sentence), self.get_ngram_order(pred_sentence))
            score_bleu += sentence_bleu([target_sentence], pred_sentence, weights=[1 / ngram_order] * ngram_order)
            score_meteor += nltk.translate.meteor_score.meteor_score([target_sentence], pred_sentence)
            if debug:
                print("The predicted sentence: "+str(pred_sentence))
                print("The correct sentence: "+str(target_sentence))
                print("Bleu socre: "+str(score_bleu))
                print("Meteor score: "+str(score_meteor))
        return score_bleu / number_of_sentences , score_meteor/number_of_sentences


    def model_evaluation(self,test_set,source2in,target2in,in2target):
        test_pairs = self.index_sentences(test_set, dict(source2in), dict(target2in))
        test_pairs = self.create_batches(test_pairs)
        padded_source_test , padded_target_test = self.pad(test_pairs)
        total_score = 0
        total_meteor = 0
        total_loss = 0
        with torch.no_grad():
            for batch in range(len(padded_source_test)):
                source_tensor = torch.tensor(padded_source_test[batch], dtype=torch.long, device=self.device)
                target_tensor = torch.tensor(padded_target_test[batch], dtype=torch.long, device=self.device)
                source_tensor = torch.transpose(source_tensor, 0, 1)
                target_tensor = torch.transpose(target_tensor, 0, 1)

                output = self(source_tensor, target_tensor)

                score,meteor = self.calculate_metric(output, target_tensor, dict(in2target), False)
                output = output[1:].reshape(-1, output.shape[2])
                target_tensor = target_tensor[1:].reshape(-1)
                loss = self.criterion(output, target_tensor)
                total_loss += loss
                total_score += score
                total_meteor += meteor

        return total_score / len(padded_source_test),math.exp(total_loss/len(padded_source_test)) , total_meteor/len(padded_source_test)

    def save_model(self):
        torch.save(self.state_dict(), "saved_seq2seq_model.pth")

    def translate(self,sentence_source,sentence_target,source2index,target2index,index2target):
        sentence_source_original = np.array(sentence_source)

        source_sentence_idx = []
        target_sentence_idx = []

        for word in sentence_source:
            source_sentence_idx.append(source2index[word])
        for word in sentence_target:
            target_sentence_idx.append(target2index[word])

        source_sentence_idx = [source_sentence_idx]
        target_sentence_idx = [target_sentence_idx]

        source_tensor = torch.tensor(source_sentence_idx, dtype=torch.long, device=self.device)
        target_tensor = torch.tensor(target_sentence_idx, dtype=torch.long, device=self.device)
        source_tensor = torch.transpose(source_tensor, 0, 1)
        target_tensor = torch.transpose(target_tensor, 0, 1)

        with torch.no_grad():
            output = self(source_tensor, target_tensor)

        pred = torch.transpose(output, 0, 1)
        target = torch.transpose(target_tensor, 0, 1)
        pred = pred.detach().cpu().numpy()
        target = target.cpu().numpy()

        for i in range(len(target)):
            pred_sentence = np.array([])
            target_sentence = np.array([])
            for j in range(len(target[i])):
                target_sentence = np.append(target_sentence,[index2target[target[i][j]]],axis=0)
                pred_sentence = np.append(pred_sentence, [index2target[pred[i][j].argmax(axis=0)]], axis=0)
            target_sentence,pred_sentence = self.remove_tokens(target_sentence,pred_sentence)
        sentence_source_original = sentence_source_original[1:]
        sentence_source_original = sentence_source_original[:-1]
        target_sentence = target_sentence.tolist()
        pred_sentence = pred_sentence.tolist()
        sentence_source_original = sentence_source_original.tolist()

        print("The given sentense is: " + " ".join(sentence_source_original))
        print("The translated sentence is: " + " ".join(target_sentence))
        print("The translated prediction is: " + " ".join(pred_sentence))


    def fit(self, source_lan, trans_lan, train_pairs,val_pairs, epochs=20000,verbose=1,shuffle=True):
        scaler = torch.cuda.amp.grad_scaler.GradScaler()

        train_pairs = self.index_sentences(train_pairs, dict(source_lan[0]), dict(trans_lan[0]))
        train_pairs = self.create_batches(train_pairs)
        padded_source_train , padded_target_train = self.pad(train_pairs)

        val_pairs = self.index_sentences(val_pairs, dict(source_lan[0]), dict(trans_lan[0]))
        val_pairs = self.create_batches(val_pairs)
        padded_source_val , padded_target_val = self.pad(val_pairs)


        for epoch in range(1, epochs + 1):
            start = time.time()
            total_train_loss = 0
            total_val_loss = 0
            total_train_score = 0
            total_val_score = 0
            total_train_meteor = 0
            total_val_meteor = 0

            if (verbose != 0):
                if (epoch + 1) % self.verbose_levels[verbose] == 0:
                    print("\n")
                    print("Epoch: " + str(epoch) + "/" + str(epochs) + " - ║{0:20s}║ {1:.1f}%".format(
                            '🟩' * int((epoch) / epochs * 20), (epoch) / epochs * 100))
            if shuffle:
                padded_source_train, padded_target_train = sklearn.utils.shuffle(np.array(padded_source_train), np.array(padded_target_train))
                padded_target_train = padded_target_train.tolist()
                padded_source_train = padded_source_train.tolist()
            self.train()
            for batch in range(len(padded_source_train)):
                source_tensor = torch.tensor(padded_source_train[batch], dtype=torch.long, device=self.device)
                target_tensor = torch.tensor(padded_target_train[batch], dtype=torch.long, device=self.device)
                source_tensor = torch.transpose(source_tensor, 0, 1)
                target_tensor = torch.transpose(target_tensor, 0, 1)

                with torch.cuda.amp.autocast_mode.autocast(dtype=torch.float16):
                    output = self(source_tensor, target_tensor)

                    score,met_score = self.calculate_metric(output,target_tensor,dict(trans_lan[1]),False)

                    output = output[1:].reshape(-1,output.shape[2])
                    target_tensor = target_tensor[1:].reshape(-1)
                    loss = self.criterion(output,target_tensor)

                self.opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                scaler.step(optimizer=self.opt)
                scaler.update()

                total_train_loss += loss
                total_train_score += score
                total_train_meteor += met_score
            train_perplexity = math.exp((total_train_loss / len(padded_source_train)).item())
            self.print_progress("Train", (total_train_score / len(padded_source_train)),(total_train_loss / len(padded_source_train)).item(),train_perplexity,(total_train_meteor / len(padded_source_train)) ,epoch + 1, verbose)


            with torch.no_grad():
                self.eval()
                for batch in range(len(padded_source_val)):

                    source_tensor = torch.tensor(padded_source_val[batch], dtype=torch.long, device=self.device)
                    target_tensor = torch.tensor(padded_target_val[batch], dtype=torch.long, device=self.device)
                    source_tensor = torch.transpose(source_tensor, 0, 1)
                    target_tensor = torch.transpose(target_tensor, 0, 1)
                    with torch.cuda.amp.autocast_mode.autocast(dtype=torch.float16):
                        output = self(source_tensor, target_tensor)

                        score,met_score = self.calculate_metric(output,target_tensor,dict(trans_lan[1]),False)

                        output = output[1:].reshape(-1,output.shape[2])
                        target_tensor = target_tensor[1:].reshape(-1)

                        loss = self.criterion(output,target_tensor)

                    total_val_loss += loss
                    total_val_score += score
                    total_val_meteor += met_score

                val_perplexity = math.exp((total_val_loss / len(padded_source_val)).item())
                self.print_progress("Val", (total_val_score / len(padded_source_val)),(total_val_loss / len(padded_source_val)).item(),val_perplexity,(total_val_meteor / len(padded_source_val)), epoch + 1, verbose)


            end = time.time()
            print("Epoch time elapsed: " + str((end - start)))
            self.History["loss"].append((total_train_loss.cpu().detach().numpy() / len(padded_source_train)))
            self.History["accuracy"].append((total_train_score / len(padded_source_train)))
            self.History["val_loss"].append((total_val_loss.cpu().detach().numpy() / len(padded_source_val)))
            self.History["val_accuracy"].append((total_val_score / len(padded_source_val)))
            self.History["epoch_time"].append(end - start)
            self.History["perplexity"].append(train_perplexity)
            self.History["val_perplexity"].append(val_perplexity)
            self.History["meteor"].append((total_train_meteor / len(padded_source_train)))
            self.History["val_meteor"].append((total_val_meteor / len(padded_source_val)))

            if self.early_stop:
                if self.early_stopper.early_stop((total_val_loss / len(padded_source_val))):
                    break

        self.save_model()