#%%
import os 
import pickle 
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import defaultdict
# import sklearn
import seaborn as sns

#%% data loading
data_path = '/Users/ivan_zorin/Documents/AIRI/data/batt'
pkl_file = 'batt_data.pkl'
image_path = './latex_images/'

with open(os.path.join(data_path, pkl_file), 'rb') as f:
    bat_data = pickle.load(f)
    

#%% extract life cycles
cycle_lifes = np.array([bat_data[bat]['cycle_life'].item() for bat in bat_data.keys()])

cycle_lifes_known = cycle_lifes[~np.isnan(cycle_lifes)]

print(f'nan cycle life: {cycle_lifes[np.isnan(cycle_lifes)].shape}')

print(f'available cycle life: {cycle_lifes[~np.isnan(cycle_lifes)].shape}')

print(f'min cycle life: {cycle_lifes_known.min()}')
print(f'max cycle life: {cycle_lifes_known.max()}')


plt.figure()
sns.histplot(cycle_lifes_known)
plt.xlabel('cycle life')
plt.title("Histogramm plot of batteries' with known cycle life ")
plt.show()

# plt.savefig(image_path + 'cycle_life_plot.png', format='png')

#%% specific battery
bat_ids = list(bat_data.keys())
bat_id = bat_ids[0]
bat = bat_data[bat_id]
#%% the battery summary
print(f'nunber of batteries: {len(bat_ids)}')

print('each ids has the following attributes')
print(bat.keys())

print(f"label (life of a batt in cycles): {bat['cycle_life']}")

print('summary statistics contained in the summary key')
print(bat['summary'].keys())

for key in bat['summary'].keys():
    plt.figure()
    y = bat['summary'][key]
    plt.plot(y)
    plt.title(f"{key} of len: {len(y)}, label: {bat['cycle_life']}")
    
    
    
#%%
feature = 'Qd'
max_qd = []
max_qd_idx = []

plt.figure()
for cycle in bat['cycles'].keys():
    qd = bat['cycles'][cycle][feature]
    max_idx = np.argmax(qd)
    max_qd_idx += [max_idx]
    max_qd += [qd[max_idx]]
    
    plt.plot(qd)
    plt.plot(max_idx, qd[max_idx], '*', 'r')
    

plt.show()



#%%
plt.figure()
plt.plot(bat['summary']['QD'], 'b')
plt.plot(max_qd, 'r', alpha=0.5)
plt.show()
    
#%%

feature = 'QD'
start = 5
end = 10
plt.figure()
for bat_idx in list(bat_data.keys())[start:end]:
    plt.plot(bat_data[bat_idx]['summary'][feature])
plt.show()

#%%
print(bat['cycles']['0'].keys())
features = bat['cycles']['0']

#%%
plt.figure()
plt.plot(features['Qd'])
plt.plot(features['Qdlin'])
plt.show()


plt.figure()
plt.plot(features['t'], features['I'])
plt.show()

plt.figure()
plt.plot(features['t'], features['Qd'])
plt.show()

# plt.figure()
# plt.plot(features['t'])

#%%
t_max = []

for cycle in bat['cycles'].keys():
    t_max += [bat['cycles'][cycle]['t'].max()]
    
plt.figure()
plt.plot(np.log10(t_max))
plt.show()
#%%  plot some features for selected cycles 
def plot_cycle_signals(bat_data, batt_id, cycle_ids, keys_to_plot=['I', 'V'], save=False):
    n_subplots = 2  # len(keys_to_plot)
    m_subplots = 2
    fig, axs = plt.subplots(n_subplots, m_subplots, figsize=(m_subplots * 10, n_subplots * 10))
    for idx, key in enumerate(keys_to_plot):
        i = idx // 2
        j = idx % 2
        # print(i,j)
        for cycle_id in cycle_ids:
            y = bat_data['cycles'][cycle_id][key]
            axs[i,j].plot(y, label=cycle_id)
            
        axs[i,j].set_title(f"Signal of {key}")
    
    fig.suptitle(f'Measured signals of Battery {batt_id}')
    axs[0,0].legend(title='Cycles')
    plt.show() 
    if save:
        fig.savefig(image_path + f"battery_{batt_id}.png", format='png')


keys_to_plot = ['I', 'V', 'Qc', 'Qd']
bat_cycles = [cycle for cycle in bat['cycles'].keys()]
cycle_ids = [bat_cycles[0], bat_cycles[-1]]

plot_cycle_signals(bat, bat_id, cycle_ids, keys_to_plot, save=False)


    
#%% all batteries 

print(bat_ids[:10])

print('cycle life info')
labels = {bat_id: bat_data[bat_id]['cycle_life'].item() for bat_id in bat_ids}
labels_nan = [label for label in labels.keys() if math.isnan(labels[label])]
print(f"{len(labels_nan)} batteries with nan values in cycle_life")

labels_good = {label: labels[label] for label in labels.keys() if not math.isnan(labels[label])}

print(f"{len(labels_good.keys())} batteries with correct values in cycle_life")

print(f"average cycle_life for batteries with correct value is {np.mean(list(labels_good.values()))}")

#%% plotting all features of random battery (among correct)

rand_bat_id = np.random.choice(list(labels_good.keys()))

for cycle in bat_data[rand_bat_id]['cycles'].keys():
    
    for signal, value in bat_data[rand_bat_id]['cycles'][cycle].items():
        plt.figure(signal)
        plt.title(f"battery {rand_bat_id}, {signal}")
        plt.plot(value, alpha=0.3)
        

for signal in bat_data[rand_bat_id]['cycles']['0'].keys():
    fig = plt.figure(signal)
    fig.savefig(f'./pics/{rand_bat_id}_{signal}.png', format='png')
        
#%% FFT of features 

figure_names = list(bat_data[rand_bat_id]['cycles']['0'].keys())

for bat_id in labels_good.keys():   
    signals_list = defaultdict(list)
    signals_dft_list = defaultdict(list)
    for cycle in bat_data[bat_id]['cycles'].keys():
        for signal, value in bat_data[bat_id]['cycles'][cycle].items():
            signals_list[signal].append(value)
            signals_dft_list[signal].append(np.fft.fft(value, n=512))
            
    
    for signal in signals_list.keys():
        avg_dft_signal = np.mean(signals_dft_list[signal], axis=0)        
        avg_signal = np.fft.ifft(avg_dft_signal, n=512)
        
        plt.figure(signal)
        plt.plot(avg_signal, alpha=0.3)
        plt.title(f'average signal (dft) of {signal}')
        plt.show()

#%%  
for signal in signals_list.keys():
    fig = plt.figure(signal)
    fig.savefig(f'./pics/bat_avg_{signal}.png', format='png')

# signals of each cycle has different lengths therefore it can not be averaged directly. Try to use DFT then average and inverve DFT after that. 


#%% feature generator as is in the paper
plt.figure()
plt.plot(bat['cycles']['10']['Qc'], label='10')
plt.plot(bat['cycles']['100']['Qc'], label='100')
plt.plot(bat['cycles']['500']['Qc'], label='500')
plt.plot(bat['cycles']['1301']['Qc'], label='1301')
plt.title('Qc')

plt.legend()
plt.show()

plt.figure()
plt.plot(bat['cycles']['10']['Qd'], label='10')
plt.plot(bat['cycles']['100']['Qd'], label='100')
plt.plot(bat['cycles']['500']['Qd'], label='500')
plt.plot(bat['cycles']['1301']['Qd'], label='1301')
plt.title('Qd')
plt.legend()
plt.show()
    
#%% features 
def build_feature_df(batch_dict):
    """Returns a pandas DataFrame with all originally used features out of a loaded batch dict"""

    print("Start building features ...")

    from scipy.stats import skew, kurtosis
    from sklearn.linear_model import LinearRegression
    
    # 124 cells (3 batches)
    n_cells = len(batch_dict.keys())

    ## Initializing feature vectors:
    # numpy vector with 124 zeros
    cycle_life = np.zeros(n_cells)
    # 1. delta_Q_100_10(V)
    minimum_dQ_100_10 = np.zeros(n_cells)
    variance_dQ_100_10 = np.zeros(n_cells)
    skewness_dQ_100_10 = np.zeros(n_cells)
    kurtosis_dQ_100_10 = np.zeros(n_cells)

    # 2. Discharge capacity fade curve features
    slope_lin_fit_2_100 = np.zeros(n_cells)  # Slope of the linear fit to the capacity fade curve, cycles 2 to 100
    intercept_lin_fit_2_100 = np.zeros(n_cells)  # Intercept of the linear fit to capavity face curve, cycles 2 to 100
    discharge_capacity_2 = np.zeros(n_cells)  # Discharge capacity, cycle 2
    diff_discharge_capacity_max_2 = np.zeros(n_cells)  # Difference between max discharge capacity and cycle 2

    # 3. Other features
    mean_charge_time_2_6 = np.zeros(n_cells)  # Average charge time, cycle 2 to 6
    minimum_IR_2_100 = np.zeros(n_cells)  # Minimum internal resistance
   
    diff_IR_100_2 = np.zeros(n_cells)  # Internal resistance, difference between cycle 100 and cycle 2

    # Classifier features
    minimum_dQ_5_4 = np.zeros(n_cells)
    variance_dQ_5_4 = np.zeros(n_cells)
    cycle_550_clf = np.zeros(n_cells)
    
    # iterate/loop over all cells.
    for i, cell in enumerate(batch_dict.values()):
        cycle_life[i] = cell['cycle_life'] 
        # 1. delta_Q_100_10(V)
        c10 = cell['cycles']['10']
        c100 = cell['cycles']['100']
        dQ_100_10 = c100['Qdlin'] - c10['Qdlin']

        minimum_dQ_100_10[i] = np.log10(np.abs(np.min(dQ_100_10)))
        variance_dQ_100_10[i] = np.log(np.abs(np.var(dQ_100_10)))
        skewness_dQ_100_10[i] = np.log(np.abs(skew(dQ_100_10)))
        kurtosis_dQ_100_10[i] = np.log(np.abs(kurtosis(dQ_100_10)))

        # 2. Discharge capacity fade curve features
        # Compute linear fit for cycles 2 to 100:
        q = cell['summary']['QD'][1:100].reshape(-1, 1)  # discharge cappacities; q.shape = (99, 1);
        X = cell['summary']['cycle'][1:100].reshape(-1, 1)  # Cylce index from 2 to 100; X.shape = (99, 1)
        linear_regressor_2_100 = LinearRegression()
        linear_regressor_2_100.fit(X, q)

        slope_lin_fit_2_100[i] = linear_regressor_2_100.coef_[0]
        intercept_lin_fit_2_100[i] = linear_regressor_2_100.intercept_
        discharge_capacity_2[i] = q[0][0]
        diff_discharge_capacity_max_2[i] = np.max(q) - q[0][0]

        # 3. Other features
        mean_charge_time_2_6[i] = np.mean(cell['summary']['chargetime'][1:6])
        minimum_IR_2_100[i] = np.min(cell['summary']['IR'][1:100])
        diff_IR_100_2[i] = cell['summary']['IR'][100] - cell['summary']['IR'][1]

        # Classifier features
        c4 = cell['cycles']['4']
        c5 = cell['cycles']['5']
        dQ_5_4 = c5['Qdlin'] - c4['Qdlin']
        minimum_dQ_5_4[i] = np.log10(np.abs(np.min(dQ_5_4)))
        variance_dQ_5_4[i] = np.log10(np.var(dQ_5_4))
        cycle_550_clf[i] = cell['cycle_life'] >= 550

    # combining all featues in one big matrix where rows are the cells and colums are the features
    # note last two variables below are labels/targets for ML i.e cycle life and cycle_550_clf
    features_df = pd.DataFrame({
        "cell_key": np.array(list(batch_dict.keys())),
        "minimum_dQ_100_10": minimum_dQ_100_10,
        "variance_dQ_100_10": variance_dQ_100_10,
        "skewness_dQ_100_10": skewness_dQ_100_10,
        "kurtosis_dQ_100_10": kurtosis_dQ_100_10,
        "slope_lin_fit_2_100": slope_lin_fit_2_100,
        "intercept_lin_fit_2_100": intercept_lin_fit_2_100,
        "discharge_capacity_2": discharge_capacity_2,
        "diff_discharge_capacity_max_2": diff_discharge_capacity_max_2,
        "mean_charge_time_2_6": mean_charge_time_2_6,
        "minimum_IR_2_100": minimum_IR_2_100,
        "diff_IR_100_2": diff_IR_100_2,
        "minimum_dQ_5_4": minimum_dQ_5_4,
        "variance_dQ_5_4": variance_dQ_5_4,
        "cycle_life": cycle_life,
        "cycle_550_clf": cycle_550_clf
    })

    print("Done building features")
    return features_df
    

#%%
features_df = build_feature_df(bat_data)

features_df = features_df.dropna(subset=['cycle_life'])

#%% 
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(features_df, test_size=0.2)

#%%
x_train = df_train.minimum_dQ_100_10.to_numpy().reshape(-1, 1)
y_train = df_train.cycle_life.to_numpy().reshape(-1, 1)

x_test = df_test.minimum_dQ_100_10.to_numpy().reshape(-1, 1)
y_test = df_test.cycle_life.to_numpy().reshape(-1, 1)

#%%
from sklearn.linear_model import LinearRegression

reg_model = LinearRegression(normalize=True)

# %%
reg_model.fit(x_train, y_train)

# %%
y_predict = reg_model.predict(x_test)
# %%
from sklearn.metrics import accuracy_score

# accuracy = accuracy_score(y_test, y_predict)
print(np.abs(y_predict - y_test).mean())

threshold = 10
accuracy = 0
for y, y_true in zip(y_predict, y_test): 
    if np.abs(y - y_true) < threshold: 
        accuracy += 1

# print(accuracy / len(y_test))
print(y_predict - y_test)
print((y_predict - y_true).mean())


# %%
