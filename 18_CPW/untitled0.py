# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 18:05:04 2020

@author: osama
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 22:38:49 2020

"""

from platform import python_version
import copy
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
import scipy.io as io
import matplotlib.pyplot as plt
print("python version==%s" % python_version())
print("pandas==%s" % pd.__version__)
print("numpy==%s" % np.__version__)
print("sklearn==%s" % sklearn.__version__)
print("torch==%s" % torch.__version__)
print("matplotlib==%s" % matplotlib.__version__)


plt.close(fig='all')

# %% my transformation
    
def preprocessing_transformation(x):
    return torch.sin(x) #torch.log(x)  #torch.log(x)#1/(1+torch.exp(x))

def postprocessing_transformation(x):
    return torch.arcsin(x) #torch.exp(x)   #torch.log(1.0/((1/x)-1.0))
# %%

###############################################################################
# %%
#FILE_PATH = "C:/Users/obhatti3/OneDrive - Georgia Institute of Technology/Frequency Extrapolation/Data/Random_data"

# the new file path after the new laptop haha :p 


FILE_PATH = 'C:/Users/osama/OneDrive - Georgia Institute of Technology/Research/3A5_Design_Space_Exploration_using_Neural_networks/Frequency Extrapolation/18_third_application/Data/'

f_real = open(FILE_PATH + 'TL_data_S21_real.txt')
f_imag = open(FILE_PATH + 'TL_data_S21_imag.txt')

g_real = f_real.read()
g_imag = f_imag.read()

d_real = g_real.split()
d_imag = g_imag.split()

freq = d_real[4::2]
s_21_real = d_real[5::2]
s_21_imag = d_imag[5::2]

frequency = np.zeros((len(freq),))
S21_real = np.zeros((len(freq),))
S21_imag = np.zeros((len(freq),))

for i in range(len(freq)):
    frequency[i] = float(freq[i])
    S21_real[i] = float(s_21_real[i])
    S21_imag[i] = float(s_21_imag[i])

plt.figure()
plt.subplot(2, 1, 1), plt.plot(frequency, S21_real, label='S21 real', color='b'), plt.xlabel('frequency [GHz] -----> '), plt.ylabel('Re{S_21}')
plt.subplot(2, 1, 2), plt.plot(frequency, S21_imag, label='S21 imag', color='b'), plt.xlabel('frequency [GHz] -----> '), plt.ylabel('Im{S_21}') 
plt.grid()

S_21 = S21_real + 1j*S21_real

S_21_abs = np.absolute(S_21)
S_21_phase = np.angle(S_21)

plt.figure()
plt.subplot(2, 1, 1), plt.plot(frequency, S_21_abs, label='|S21|', color='b'), plt.xlabel('frequency [GHz] -----> '), plt.ylabel('|S_21|'), plt.grid()
plt.subplot(2, 1, 2), plt.plot(frequency, S_21_phase, label='angle(S21)', color='b'), plt.xlabel('frequency [GHz] -----> '), plt.ylabel('<S_2 [rad]}'), plt.grid()



# %%
'''############################ preparing dataset #############################'''


total_ratio = 1
totalSize = int(total_ratio*len(freq))


test_to_train_data_ratio = 0.5
a = int(test_to_train_data_ratio * totalSize)

trained_frequency = torch.Tensor(frequency[0:a])
testing_frequency = torch.Tensor(frequency[a:totalSize])

# %%

frequency_response_real = torch.Tensor(S21_real)
trained_frequency_response_real = torch.Tensor(S21_real[0:a])
testing_frequency_response_real = torch.Tensor(S21_real[a:totalSize])


frequency_response_imag = torch.Tensor(S21_imag)
trained_frequency_response_imag = torch.Tensor(S21_imag[0:a])
testing_frequency_response_imag = torch.Tensor(S21_imag[a:totalSize])

x_train = trained_frequency
y_train = trained_frequency_response_real

x_test = testing_frequency
y_test = testing_frequency_response_real

x_train = x_train.numpy().reshape(-1, 1)
y_train = y_train.numpy().reshape(-1, 1)

x_test = x_test.numpy().reshape(-1, 1)
y_test = y_test.numpy().reshape(-1, 1)



'''###########################################################################'''


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_arr = scaler.fit_transform(y_train)
test_arr = scaler.transform(y_test)

# %%
def transform_data(arr, seq_len):
    x, y = [], []
    for i in range(len(arr) - seq_len):
        x_i = arr[i : i + seq_len]
        y_i = arr[i + 1 : i + seq_len + 1]
        x.append(x_i)
        y.append(y_i)
    x_arr = np.array(x).reshape(-1, seq_len)
    y_arr = np.array(y).reshape(-1, seq_len)
    x_var = Variable(torch.from_numpy(x_arr).float())
    y_var = Variable(torch.from_numpy(y_arr).float())
    return x_var, y_var


from torch.autograd import Variable

seq_len = 20

x_train, y_train = transform_data(train_arr, seq_len)
x_test, y_test = transform_data(test_arr, seq_len)
x_val, y_val = x_test, y_test

# %%

def plot_sequence(axes, i, x_train, y_train):
    axes[i].set_title("%d. Sequence" % (i + 1))
    axes[i].set_xlabel("Time Bars")
    axes[i].set_ylabel("Scaled VWAP")
    axes[i].plot(range(seq_len), x_train[i].cpu().numpy(), color="r", label="Feature")
    axes[i].plot(range(1, seq_len + 1), y_train[i].cpu().numpy(), color="b", label="Target")
    axes[i].legend()



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
plot_sequence(axes, 0, x_train, y_train)
plot_sequence(axes, 1, x_train, y_train)

# %%

import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, output_size):
        
        super(Model, self).__init__()

        self.input_size = input_size

        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.hidden_size_3 = hidden_size_3
        self.hidden_size_4 = hidden_size_4
        
        self.output_size = output_size
        
        self.lstm1 = nn.LSTMCell(self.input_size, self.hidden_size_1)
        self.lstm2 = nn.LSTMCell(self.hidden_size_1, self.hidden_size_2)
        self.lstm3 = nn.LSTMCell(self.hidden_size_2, self.hidden_size_3)
        self.lstm4 = nn.LSTMCell(self.hidden_size_3, self.hidden_size_4)

        self.linear = nn.Linear(self.hidden_size_4, self.output_size)
        
    def forward(self, input, future=0, y=None):
        outputs = []

        # reset the state of LSTM
        # the state is kept till the end of the sequence
        h_t_1 = torch.zeros(input.size(0), self.hidden_size_1, dtype=torch.float32)
        c_t_1 = torch.zeros(input.size(0), self.hidden_size_1, dtype=torch.float32)

        h_t_2 = torch.zeros(input.size(0), self.hidden_size_2, dtype=torch.float32)
        c_t_2 = torch.zeros(input.size(0), self.hidden_size_2, dtype=torch.float32)

        h_t_3 = torch.zeros(input.size(0), self.hidden_size_3, dtype=torch.float32)
        c_t_3 = torch.zeros(input.size(0), self.hidden_size_3, dtype=torch.float32)

        h_t_4 = torch.zeros(input.size(0), self.hidden_size_4, dtype=torch.float32)
        c_t_4 = torch.zeros(input.size(0), self.hidden_size_4, dtype=torch.float32)




        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):

            h_t_1, c_t_1 = self.lstm1(input_t, (h_t_1, c_t_1) )
            h_t_1 = F.dropout(h_t_1, p=0.5, training=True)
            
            h_t_2, c_t_2 = self.lstm2(h_t_1,   (h_t_2, c_t_2) )
            h_t_2 = F.dropout(h_t_2, p=0.5, training=True)

            h_t_3, c_t_3 = self.lstm3(h_t_2,   (h_t_3, c_t_3) )
            h_t_3 = F.dropout(h_t_3, p=0.5, training=True)

            h_t_4, c_t_4 = self.lstm4(h_t_3,   (h_t_4, c_t_4) )
            h_t_4 = F.dropout(h_t_4, p=0.5, training=True)
           
            output = self.linear(h_t_4)

            outputs += [output]
            

        for i in range(future):
            if y is not None and random.random() > 0.5:
                output = y[:, [i]]  # teacher forcing
 
            h_t_1, c_t_1 = self.lstm1(input_t, (h_t_1, c_t_1) )
            h_t_2, c_t_2 = self.lstm2(h_t_1,   (h_t_2, c_t_2) )
            h_t_3, c_t_3 = self.lstm3(h_t_2,   (h_t_3, c_t_3) )
            h_t_4, c_t_4 = self.lstm4(h_t_3,   (h_t_4, c_t_4) )
            
            output = self.linear(h_t_4)
            
            outputs += [output]
        
        outputs = torch.stack(outputs, 1).squeeze(2)
        
        return outputs

# %%


import time
import random


class Optimization:
    """ A helper class to train, test and diagnose the LSTM"""

    def __init__(self, model, loss_fn, optimizer, scheduler):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_losses = []
        self.val_losses = []
        self.futures = []

    @staticmethod
    def generate_batch_data(x, y, batch_size):
        for batch, i in enumerate(range(0, len(x) - batch_size, batch_size)):
            x_batch = x[i : i + batch_size]
            y_batch = y[i : i + batch_size]
            yield x_batch, y_batch, batch

    def train(
        self,
        x_train,
        y_train,
        x_val=None,
        y_val=None,
        batch_size=50,
        n_epochs=50,
        do_teacher_forcing=None,
    ):
        seq_len = x_train.shape[1]
        for epoch in range(n_epochs):
            start_time = time.time()
            self.futures = []

            train_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_train, y_train, batch_size):
                y_pred = self._predict(x_batch, y_batch, seq_len, do_teacher_forcing)
                self.optimizer.zero_grad()
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            self.scheduler.step()
            train_loss /= batch
            self.train_losses.append(train_loss)

            self._validation(x_val, y_val, batch_size)

            elapsed = time.time() - start_time
            print(
                "Epoch %d Train loss: %.4f. Validation loss: %.4f. Avg future: %.4f. Elapsed time: %.4fs."
                % (epoch + 1, train_loss, self.val_losses[-1], np.average(self.futures), elapsed)
            )

    def _predict(self, x_batch, y_batch, seq_len, do_teacher_forcing):
        if do_teacher_forcing:
            future = random.randint(1, int(seq_len) / 2)
            limit = x_batch.size(1) - future
            y_pred = self.model(x_batch[:, :limit], future=future, y=y_batch[:, limit:])
        else:
            future = 0
            y_pred = self.model(x_batch)
        self.futures.append(future)
        return y_pred

    def _validation(self, x_val, y_val, batch_size):
        if x_val is None or y_val is None:
            return
        with torch.no_grad():
            val_loss = 0
            for x_batch, y_batch, batch in self.generate_batch_data(x_val, y_val, batch_size):
                y_pred = self.model(x_batch)
                loss = self.loss_fn(y_pred, y_batch)
                val_loss += loss.item()
            if batch==0: batch = 1
            val_loss /= batch
            self.val_losses.append(val_loss)

    def evaluate(self, x_test, y_test, batch_size, future=1):
        with torch.no_grad():
            test_loss = 0
            actual, predicted = [], []
            for x_batch, y_batch, batch in self.generate_batch_data(x_test, y_test, batch_size):
                y_pred = self.model(x_batch, future=future)
                y_pred = (
                    y_pred[:, -len(y_batch) :] if y_pred.shape[1] > y_batch.shape[1] else y_pred
                )
                loss = self.loss_fn(y_pred, y_batch)
                test_loss += loss.item()
                actual += torch.squeeze(torch.Tensor(y_batch[:, -1])).data.cpu().numpy().tolist()
                predicted += torch.squeeze(y_pred[:, -1]).data.cpu().numpy().tolist()
            if batch == 0: batch = 1
            test_loss /= batch
            return actual, predicted, test_loss

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")

#%%
def generate_sequence(scaler, model, x_sample, future=100):
    """ Generate future values for x_sample with the model """
    y_pred_tensor = model(x_sample, future=future)
    y_pred = y_pred_tensor.cpu().tolist()
    y_pred = scaler.inverse_transform(y_pred)
    return y_pred
# %%


def to_dataframe(actual, predicted):
    return pd.DataFrame({"actual": actual, "predicted": predicted})


def inverse_transform(scalar, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df

# %%
batch_size = 20
# %%
''' the real bayesian stuff here! '''

n_experiments = 1
test_uncertainty_df = np.array([])
truth = np.zeros(shape=(n_experiments, len(x_test)-batch_size))
predicted = np.zeros(shape=(n_experiments, len(x_test)-batch_size))

for i in range(n_experiments):
    print("Experiment # %d", (i))
    model_1 = Model(input_size=1, hidden_size_1=50, hidden_size_2=200, hidden_size_3=150, hidden_size_4=50, output_size=1)
    loss_fn_1 = nn.MSELoss(reduction='sum')
    optimizer_1 = optim.Adam(model_1.parameters(), lr=1e-2)
    scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size=10, gamma=0.9)
    optimization_1 = Optimization(model_1, loss_fn_1, optimizer_1, scheduler_1)
    optimization_1.train(x_train, y_train, x_val, y_val, do_teacher_forcing=False)
    actual_1, predicted_1, test_loss_1 = optimization_1.evaluate(x_test, y_test, future=50, batch_size=20)
    truth[i, :] = np.array(actual_1)
    predicted[i, :] = np.array(predicted_1)
    

# %%
test_uncertainty_df_mean = np.mean(predicted, axis=0)
test_uncertainty_df_std = np.std(predicted, axis=0)

test_uncertainty_df_lower_bound = test_uncertainty_df_mean - 10 * test_uncertainty_df_std
test_uncertainty_df_upper_bound = test_uncertainty_df_mean + 10 * test_uncertainty_df_std
#%%
x = nn.Sigmoid()
plt.figure()
plt.plot(actual_1, label='actual', color='blue')
plt.plot(test_uncertainty_df_mean, 'k*', label='mean' )
plt.plot(test_uncertainty_df_upper_bound, label='upper bound', color='green')
plt.plot(test_uncertainty_df_lower_bound, label='lower bound', color='green')
plt.plot(y_test[24:, 0], label='y_test')

plt.legend() 
plt.grid(b=True)

plt.figure()
plt.plot(trained_frequency.numpy(), trained_frequency_response_real.numpy(), label='training_data')
plt.plot(testing_frequency[:-seq_len-batch_size].numpy(), np.array(actual_1), label='actual')
plt.plot(testing_frequency[:-seq_len-batch_size-1].numpy(), test_uncertainty_df_mean[1:], label='mean')
plt.fill_between(testing_frequency[:-seq_len-batch_size-1].numpy(), test_uncertainty_df_lower_bound[1:], test_uncertainty_df_upper_bound[1:], color='lightblue', label='95% Confidence region')
plt.legend()
plt.grid(b=True)
# %%

''' TRANSFORMING BACK from the scaler transform '''


transformed_back_mean = scaler.inverse_transform(test_uncertainty_df_mean)
transformed_back_lower_bound = scaler.inverse_transform(test_uncertainty_df_lower_bound)
transformed_back_upper_bound = scaler.inverse_transform(test_uncertainty_df_upper_bound)


plt.figure(figsize=(16, 10))
plt.plot(frequency.squeeze(), S21_real)
plt.plot(testing_frequency[seq_len:-batch_size], scaler.inverse_transform(test_uncertainty_df_mean), label='mean')
plt.fill_between(testing_frequency[seq_len:-batch_size], scaler.inverse_transform(test_uncertainty_df_lower_bound), scaler.inverse_transform(test_uncertainty_df_upper_bound), color='lightblue', label='95% Confidence region')
plt.legend(loc='best')
plt.grid(b=True)



# %%

''' from the scaler transform to the windowing transform '''
yo_mama_mean = torch.zeros(transformed_back_mean.shape[0],)
yo_mama_lower_bound = torch.zeros(transformed_back_lower_bound.shape[0],)
yo_mama_upper_bound = torch.zeros(transformed_back_upper_bound.shape[0],)


for i in range((transformed_back_mean.shape[0])):
    yo_mama_mean[i] = transformed_back_mean[i] #+ window_min_test[divmod(i, smoothing_window_size)[0]])
    yo_mama_lower_bound[i] = (transformed_back_lower_bound[i])
    yo_mama_upper_bound[i] = (transformed_back_upper_bound[i])

# %%
plt.figure()
plt.plot(frequency.squeeze(), S21_real, label='actual')
plt.plot(testing_frequency[seq_len-1:-batch_size-1], yo_mama_mean, label='predicted mean')
plt.fill_between(testing_frequency[seq_len-1:-batch_size-1], yo_mama_lower_bound, yo_mama_upper_bound, color='lightblue', label='95% Confidence region')
plt.grid(b=True)
plt.legend(loc='best')
plt.show()

plt.xlabel('Frequency [GHz] ---->')
plt.ylabel('|Z| [ohms] ---->')

# %% hilbert thing adding here


def hilbert_V2(x):
    # zeros = torch.zeros((x.shape[0],x.shape[1],1)).to(device)
    # x = torch.cat((zeros, x), 2)
    S_flipped = torch.flip(x, [2])
    S_doubleSided = torch.cat((x, S_flipped[:,:,:-1]), 2)
    freq_size = S_doubleSided.shape[-1]

    M=1

    F_S2 = torch.rfft(S_doubleSided, 1)
    F_S2[:, :, 1:, :] = 2 * F_S2[:, :, 1:, :]

    zero_vector = torch.zeros((F_S2.shape[0], F_S2.shape[1], M * freq_size - (freq_size+1)// 2, F_S2.shape[3]))

    F_S2 = torch.cat((F_S2, zero_vector), 2)
    # F_S2 = F_S2[:, :, :-2, :].clone()
    HS = M * torch.ifft(F_S2, 1)
    if (M*freq_size)//2 == (M*freq_size)/2:
        HS = HS[:, :, :(M * freq_size // 2), :]
    else:
        HS = HS[:,:, :(M*freq_size+1)//2, :]
    hS_re = HS[:, :, :, 0]
    hS_im = -1*HS[:, :, :, 1]
    HS_tot = torch.cat((hS_re, hS_im), 1)
    return HS_tot

y_real_extrapolated = yo_mama_mean
y_real_extrapolated_lower_bound = yo_mama_lower_bound
y_real_extrapolated_upper_bound = yo_mama_upper_bound


ajeeb_na_change_hone_wali_cheez = torch.Tensor(frequency_response_real[0:a])

y_real_hat = torch.cat((ajeeb_na_change_hone_wali_cheez, yo_mama_mean), dim=0)
y_real_hat_lower_bound = torch.cat((ajeeb_na_change_hone_wali_cheez, yo_mama_lower_bound), dim=0)
y_real_hat_upper_bound = torch.cat((ajeeb_na_change_hone_wali_cheez, yo_mama_upper_bound), dim=0)

y_imag_hat = hilbert_V2(y_real_hat.view(1, 1, -1))[:,1, :].squeeze()
y_imag_hat_lower_bound = hilbert_V2(y_real_hat_lower_bound.view(1, 1, -1))[:,1, :].squeeze()
y_imag_hat_upper_bound = hilbert_V2(y_real_hat_upper_bound.view(1, 1, -1))[:,1, :].squeeze()


y_imag_train = torch.Tensor(frequency_response_imag[0:a])
y_imag_test = torch.Tensor(frequency_response_imag[a:totalSize])

y_imag = torch.cat((y_imag_train, y_imag_test), dim=0)


#%%
plt.figure()

plt.subplot(2, 1, 1)
plt.plot(frequency.squeeze(), S21_real, label='actual_real')
plt.plot(testing_frequency[seq_len-1:-batch_size-1], y_real_extrapolated, label='predicted mean')
plt.fill_between(testing_frequency[seq_len-1:-batch_size-1], yo_mama_lower_bound, yo_mama_upper_bound, color='lightblue', label='95% Confidence region')
plt.grid(b=True)
plt.xlabel('frequency [GHz] ------>')
plt.ylabel('Re{S_21} ----->')
plt.legend(loc='best')
plt.show()

plt.subplot(2, 1, 2)
plt.plot(frequency.squeeze(), y_imag, label='actual_real')
plt.plot(testing_frequency[seq_len-1:-batch_size-1], y_imag_hat[a:], label='predicted mean')
plt.fill_between(testing_frequency[seq_len-1:-batch_size-1], y_imag_hat_lower_bound[a:], y_imag_hat_upper_bound[a:], color='lightblue', label='95% Confidence region')
plt.grid(b=True)
plt.legend(loc='best')
plt.xlabel('frequency [GHz] ------>')
plt.ylabel('Imag{S_21} ----->')

plt.show()


# %% comparison to vector fitting

my_poles = all_poles[physical_design_point_index, :]
my_residues = all_residues[physical_design_point_index, :]
my_constant_term = all_constant_terms[0, physical_design_point_index]
my_proportional_term = all_proportional_terms[0, physical_design_point_index]
my_S_parameter_response_abs = torch.from_numpy(np.absolute(all_S_parameter_responses[physical_design_point_index, :])).type(dtype=torch.float32)
my_frequency_response_abs = frequency_response_abs[physical_design_point_index, :]

f = frequency#.numpy();
vector_fitted_response = np.array(my_constant_term + my_proportional_term*2*np.pi*1j*f, dtype=complex)

for i in range(0, my_poles.shape[-1]):
    vector_fitted_response += my_residues[i]/(2*np.pi*1j*f - my_poles[i])
        
vector_fitted_response = 50*(1+vector_fitted_response) / (1-vector_fitted_response)

plt.figure();
plt.plot(f.squeeze(), np.absolute(vector_fitted_response.squeeze()))
plt.grid()
plt.legend()
plt.figure();
plt.subplot(2, 1, 1)
plt.plot(f.squeeze(), vector_fitted_response.real.squeeze())
plt.subplot(2, 1, 2)
plt.plot(f.squeeze(), vector_fitted_response.imag.squeeze())
plt.grid()

# %%

physical_design_point_index = 0;

poles_till_a = io.loadmat(FILE_PATH + '/poles_till_a.mat')
residues_till_a = io.loadmat(FILE_PATH+'/Residues_till_a.mat')
constant_terms_till_a = io.loadmat(FILE_PATH+'/Constant_terms_till_a.mat')
proportional_terms_till_a = io.loadmat(FILE_PATH+'/Proportional_terms_till_a.mat')

my_poles_till_a = poles_till_a['Poles'][physical_design_point_index, :]
my_residues_till_a = residues_till_a['Residues'][physical_design_point_index, :]
my_constant_term_till_a = constant_terms_till_a['Constant_terms'][0, physical_design_point_index]
my_proportional_term_till_a = proportional_terms_till_a['Proportional_terms'][0, physical_design_point_index]


vector_fitted_response_till_a = np.array(my_constant_term_till_a + my_proportional_term_till_a*2*np.pi*1j*f, dtype=complex)

for i in range(0, my_poles_till_a.shape[-1]):
    vector_fitted_response_till_a += my_residues_till_a[i]/(2*np.pi*1j*f - my_poles_till_a[i])

vector_fitted_response_till_a = 50*(1+vector_fitted_response_till_a) / (1-vector_fitted_response_till_a)

plt.figure();
plt.plot(f.squeeze(), np.absolute(vector_fitted_response_till_a.squeeze()), 'k-', label='VF extrapolated')
plt.plot(f.squeeze(), np.absolute(vector_fitted_response.squeeze()), 'r-.', label='actual')
plt.grid()
plt.legend()
plt.figure();
plt.subplot(2, 1, 1)
plt.plot(f.squeeze(), vector_fitted_response_till_a.real.squeeze())
plt.subplot(2, 1, 2)
plt.plot(f.squeeze(), vector_fitted_response_till_a.imag.squeeze())
plt.grid()


# %%

V = torch.Tensor(frequency_response_real[physical_design_point_index, :])

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(frequency.squeeze(), X, 'r-', label='True')
plt.plot(frequency.squeeze()[a:-batch_size], torch.cat((V[a:a+batch_size], y_real_extrapolated)), 'k', label='Hybrid Hilbert+RNN')
#plt.fill_between(testing_frequency[seq_len-1:-batch_size-1], yo_mama_lower_bound, yo_mama_upper_bound, color='lightblue', label='95% Confidence region')
plt.plot(f.squeeze(), vector_fitted_response_till_a.real.squeeze(), 'b-.', label='VF extrapolated')
plt.grid(b=True)
plt.legend(loc='best')
plt.xlabel('Frequency [GHz] ---->')
plt.ylabel('real(Z) [ohms] ---->')
plt.show()


#y_imag_hat[a:] = torch.from_numpy(0.2*np.arange(y_imag_hat[a:].shape[-1]))*y_imag_hat[a:]

plt.subplot(2, 1, 2)
plt.plot(frequency.squeeze(), y_imag, 'r-', label='True')
#plt.plot(frequency.squeeze()[:-batch_size-seq_len], y_imag_hat, 'g', label='poora wala')
plt.plot(frequency.squeeze()[a:-batch_size], torch.cat((y_imag[a:a+batch_size], y_imag_hat[a:])), 'k', label='Hybrid Hilbert+RNN')
#plt.plot(testing_frequency[:-batch_size-1], y_imag_hat[a-batch_size:-1], 'k', label='Hybrid Hilbert+RNN')
plt.plot(f.squeeze(), vector_fitted_response_till_a.imag.squeeze(), 'b-.', label='VF extrapolated')
#plt.fill_between(testing_frequency[seq_len-1:-batch_size-1], y_imag_hat_lower_bound[a:], y_imag_hat_upper_bound[a:], color='lightblue', label='95% Confidence region')
plt.grid(b=True)
plt.legend(loc='best')
plt.xlabel('Frequency [GHz] ---->')
plt.ylabel('Imag(Z) [ohms] ---->')
plt.show()



plt.figure()
plt.plot(y_imag[a+batch_size:-batch_size])
plt.plot(y_imag_hat[a:])
diff = y_imag[a+batch_size:-batch_size] - y_imag_hat[a:]
plt.plot(diff)






# %%


