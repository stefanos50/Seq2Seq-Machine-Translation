import random
import time
import DataPreprocessing
import DecoderRNN
import EncoderRNN
import HelperMethods
from Seq2Seq import Seq2Seq
import torch

def plot_results(history,type):
    HelperMethods.plot_result(history['loss'], history['val_loss'], "Loss Plot "+str(type), "Epochs", "Loss",
                              "Train Loss", "Val Loss")
    HelperMethods.plot_result(history['accuracy'], history['val_accuracy'], "Bleu Plot "+str(type), "Epochs",
                              "Bleu", "Train Bleu", "Val Bleu")
    HelperMethods.plot_result(history['meteor'], history['val_meteor'], "Meteor Plot "+str(type), "Epochs",
                              "Meteor", "Train Meteor", "Val Meteor")
    HelperMethods.plot_result(history['perplexity'], history['val_perplexity'], "Perplexity Plot "+str(type), "Epochs",
                              "Perplexity", "Train Perplexity", "Val Perplexity")

rus_words,en_words,rus_w2i,rus_i2w,en_w2i,en_i2w,pairs,russian_sentences,english_sentences = DataPreprocessing.get_data('rus.txt',max_words=100000,plot_res=True)

en_unique_words = len(list(en_i2w.keys()))
rus_unique_words = len(list(rus_i2w.keys()))
#350 500
embedded_size = 350
hidden_neurons = 500
hidden_layers = 2
epochs = 18
dropout = 0.5
batch_size = 32
eval_number = 5000
hardware = 'cuda'
phase = 'train'
test_param = 'AdamW'

#init encoder and decoder classes
rnn_encoder = EncoderRNN.EncoderRNN(en_unique_words, hidden_neurons, embedded_size, hidden_layers,dropout)
rnn_decoder = DecoderRNN.DecoderRNN(rus_unique_words,rus_unique_words, hidden_neurons, embedded_size, hidden_layers,dropout)

#init gpu
device = HelperMethods.initialize_hardware(hardware)

model = Seq2Seq(rnn_encoder, rnn_decoder, device,batch_size=batch_size,weight_decay=0.0001,optimizer='RMSprop',teacher_forcing_ratio=0.5,learning_rate=0.001,momentum=0.9).to(device)
train_set, val_set, test_set = DataPreprocessing.split_dataset(pairs, 2000, 2000)

if phase == 'train':
    start = time.time()
    model.fit([en_w2i,en_i2w],[rus_w2i,rus_i2w],train_set,val_set,epochs)
    end = time.time()
    bleu_score_train,perplexity_train,meteor_train = model.model_evaluation(train_set,en_w2i,rus_w2i,rus_i2w)
    bleu_score_test,perplexity_test,meteor_test = model.model_evaluation(test_set,en_w2i,rus_w2i,rus_i2w)

    history = model.History
    history['train_accuracy'] = bleu_score_train
    history['test_accuracy'] = bleu_score_test
    history['train_meteor'] = meteor_train
    history['test_meteor'] = meteor_test
    history['train_perplexity'] = perplexity_train
    history['test_perplexity'] = perplexity_test
    history['time'] = end-start
    history['param'] = test_param
    HelperMethods.save_history(history)

    for i in range(len(val_set)):
        print("------")
        model.translate(val_set[i][0], val_set[i][1], en_w2i, rus_w2i,rus_i2w)

    print("Bleu train: "+str(round(bleu_score_train*100,2)))
    print("Bleu test: "+str(round(bleu_score_test*100,2)))
    print("Meteor train: "+str(round(meteor_train*100,2)))
    print("Meteor test: "+str(round(meteor_test*100,2)))
    print("Perplexity train: "+str(perplexity_train))
    print("Perplexity test: "+str(perplexity_test))
    print("Training time: "+str(end-start))
    plot_results(model.History,"")
elif phase == 'eval':
    model.load_state_dict(torch.load("saved_seq2seq_model.pth"))
    model.eval()
    bleu_score_train,perplexity_score_train,meteor_train = model.model_evaluation(train_set,en_w2i,rus_w2i,rus_i2w)
    bleu_score_test,perplexity_score_test,meteor_test = model.model_evaluation(test_set,en_w2i,rus_w2i,rus_i2w)
    print("Bleu train: "+str(round(bleu_score_train*100,2)))
    print("Bleu test: "+str(round(bleu_score_test*100,2)))
    print("Meteor train: "+str(round(meteor_train*100,2)))
    print("Meteor test: "+str(round(meteor_test*100,2)))
    print("Perplexity train: "+str(perplexity_score_train))
    print("Perplexity test: "+str(perplexity_score_test))
    for i in range(eval_number):
        print("--------------")
        random_sentence = random.randint(0,len(english_sentences))
        model.translate(english_sentences[random_sentence],russian_sentences[random_sentence],en_w2i,rus_w2i,rus_i2w)