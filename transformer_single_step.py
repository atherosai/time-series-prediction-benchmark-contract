from datetime import date
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from base_config import config
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from utils import print_metrics
from matplotlib import pyplot

transformer_config = config['methods_hyper_config']['transformer']

torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

#src = torch.rand((10, 32, 512)) # (S,N,E) 
#tgt = torch.rand((20, 32, 512)) # (T,N,E)
#out = transformer_model(src, tgt)
#
#print(out)

input_window = transformer_config['input_window']
output_window = transformer_config['output_window']
lr_definition = transformer_config['lr']
loopback = transformer_config['loopback']
number_of_epochs = transformer_config['num_epochs']
batch_size = transformer_config['batch_size'] # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = MinMaxScaler(feature_range=(-1, 1)) 
criterion = nn.MSELoss()

lr = lr_definition
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)

best_val_loss = float("inf")
epochs = number_of_epochs # The number of epochs
best_model = None


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
       

class TransAm(nn.Module):
    def __init__(self,feature_size=transformer_config['hidden_dim'], num_layers=transformer_config['num_layers'], dropout=transformer_config['dropout']):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=transformer_config['nhead'], dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

model = TransAm().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        print('train_seq: ', train_seq);
        print('train_label: ', train_label);
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

def prepare_data(series_train, series_test):

    price_train = scaler.fit_transform(
        series_train['open'].to_numpy().reshape(-1, 1)
    ).reshape(-1)

    price_test = scaler.fit_transform(
        series_test['open'].to_numpy().reshape(-1, 1)
    ).reshape(-1)

    # amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    # convert our train data into a pytorch train tensor
    #train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment.. 
    train_sequence = create_inout_sequences(price_train, input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack?

    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_inout_sequences(price_test,input_window)
    test_data = test_data[:-output_window] #todo: fix hack?

    return train_sequence.to(device),test_data.to(device)

def get_batch(source, i,batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target


def train(train_data, epoch):
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

def plot_and_loss(eval_model, data_source, epoch, series):
    eval_model.eval()
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)

    # print('date source', data_source)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i,1)
            # look like the model returns static values for the output window
            output = eval_model(data)                
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0) #todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)


    #test_result = test_result.cpu().numpy()
    print('train', series['series_train'])

    print('test', series['series_test'])
    # train_dates = dates['train_dates']
    # test_dates = dates['test_dates'][:145]


    truth_result = scaler.inverse_transform(truth[:len(test_result)].numpy().reshape(1, -1)).reshape(-1)
    test_result = scaler.inverse_transform(test_result.numpy().reshape(1, -1)).reshape(-1)
    dates_train = series['series_train'].iloc[::5, :]
    
    print('dates', dates_train.shape)
    print('train dates', series['series_train'].shape)
    print('test dates', series['series_test'].shape)
    print('test results', test_result.shape)
    print('truth results', truth_result.shape)

    pyplot.plot(test_result, color="red")
    pyplot.plot(truth_result, color="blue")
    # pyplot.plot(test_result-truth,color="green")
    # pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    # pyplot.show()
    # pyplot.savefig('graphs/transformer/transformer-single-epoch%d.png'%epoch)
    pyplot.close()
    
    return total_loss / i

def predict_future(eval_model, data_source, original_series, epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)

    test_series = original_series['series_test']
    training_series = original_series['series_train']
    steps = 151

    # _, full_data = get_batch(data_source, 0, 1)
    data, _ = get_batch(data_source, 0, 1)
    with torch.no_grad():
        for i in range(0, steps):            
            output = eval_model(data[-input_window:])                        
            data = torch.cat((data, output[-1:]))
    
    data = data.cpu().view(-1)

    # all_data = training_series.append(test_series)
    # testing_data = test_series['open'].to_numpy()
    # training_data = training_series['open'].to_numpy()

    # predictions_plot = np.empty_like(all_data)
    # predictions_plot[:] = np.nan
    # testing_plot = np.empty_like(all_data)
    # testing_plot[:] = np.nan
    # training_plot = np.empty_like(all_data)
    # training_plot[:] = np.nan


    # shift train predictions for plotting
    # trainPredictPlot = np.empty_like(all_data)
    # trainPredictPlot[:, :] = np.nan
    # trainPredictPlot[loopback:len(training_data)+loopback, :] = training_data

    # # shift test predictions for plotting
    # testPredictPlot = np.empty_like(all_data)
    # testPredictPlot[:, :] = np.nan
    # testPredictPlot[len(training_data)+loopback-1:len(all_data)-1, :] = scaler.inverse_transform(data.numpy().reshape(1, -1)).reshape(-1)

    # # shift predictions for plotting
    # predictionsPlot = np.empty_like(all_data)
    # predictionsPlot[:, :] = np.nan
    # predictionsPlot[len(training_data)+loopback-1:len(all_data)-1, :] = scaler.inverse_transform(data.numpy().reshape(1, -1)).reshape(-1)


    testing_plot = test_series['open'].to_numpy()
    training_plot = training_series['open'].to_numpy()
    predictions_plot = scaler.inverse_transform(data.numpy().reshape(1, -1)).reshape(-1)

    print_metrics(test_series['open'].to_numpy(), scaler.inverse_transform(data.numpy().reshape(1, -1)).reshape(-1));

    print('training_plot: ', training_plot)
    print('training_plot size: ', training_plot.size)
    print('testing_plot: ', testing_plot)
    print('testing_plot size: ', testing_plot.size)
    print('predictions_plot: ', predictions_plot)
    print('predictions_plot size: ', predictions_plot.size)

    # pyplot.plot(all_data, color="blue")
    pyplot.plot(testing_plot, color="red")
    pyplot.plot(predictions_plot, color="violet")
    pyplot.xlabel('Predikce vs realná data na části testovacího souboru')
    pyplot.ylabel(f"{config['ticker']} cena akcie")
    pyplot.legend(
        [
            'Predikce',
            'Testovací datový soubor',
            'Trénovací datový soubor',
        ]
    )

    pyplot.show()
    # pyplot.savefig(f"graphs/transformer/transformer_predicted_{config['ticker']}_{str(steps)}_{str(epoch)}.png")
    pyplot.close()
 
def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = transformer_config['eval_batch_size']
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)            
            total_loss += len(data[0])* criterion(output, targets).cpu().item()
    return total_loss / len(data_source)

def predict_with_transformer_single(series_train, series_test):

    train_data, val_data = prepare_data(series_train, series_test)

    print('train_data: ', train_data)
    print('series_test: ', series_test)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(train_data, epoch)
        
        if(epoch % 10 is 0):
            val_loss = plot_and_loss(model, val_data, epoch, {
                "series_train": series_train,
                "series_test": series_test 
            })
            predict_future(model, val_data, {
                "series_train": series_train,
                "series_test": series_test
            }, epoch)
        else:
            val_loss = evaluate(model, val_data)
            predict_future(model, val_data, {
                "series_train": series_train,
                "series_test": series_test
            }, epoch)
            
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),val_loss, math.exp(val_loss)))
        print('-' * 89)

        # if val_loss < best_val_loss:
        #    best_val_loss = val_loss
        #    best_model = model

        scheduler.step() 

#src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number) 
#out = model(src)
#
#print(out)
#print(out.shape)