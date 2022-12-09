import h5py
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")
# from data import load_battery_data


def load_battery_data(batch_path, save_name='batt_data.pkl', is_save=True):
    batch1 = pickle.load(open(batch_path+'batch1.pkl', 'rb'))
    #remove batteries that do not reach 80% capacity
    del batch1['b1c8']
    del batch1['b1c10']
    del batch1['b1c12']
    del batch1['b1c13']
    del batch1['b1c22']

    batch2 = pickle.load(open(batch_path+'batch2.pkl','rb'))
    # There are four cells from batch1 that carried into batch2, we'll remove the data from batch2
    # and put it with the correct cell from batch1
    batch2_keys = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    batch1_keys = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    add_len = [662, 981, 1060, 208, 482]

    for i, bk in enumerate(batch1_keys):
        batch1[bk]['cycle_life'] = batch1[bk]['cycle_life'] + add_len[i]
        for j in batch1[bk]['summary'].keys():
            if j == 'cycle':
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j] + len(batch1[bk]['summary'][j])))
            else:
                batch1[bk]['summary'][j] = np.hstack((batch1[bk]['summary'][j], batch2[batch2_keys[i]]['summary'][j]))
        last_cycle = len(batch1[bk]['cycles'].keys())
        for j, jk in enumerate(batch2[batch2_keys[i]]['cycles'].keys()):
            batch1[bk]['cycles'][str(last_cycle + j)] = batch2[batch2_keys[i]]['cycles'][jk]

    del batch2['b2c7']
    del batch2['b2c8']
    del batch2['b2c9']
    del batch2['b2c15']
    del batch2['b2c16']

    batch3 = pickle.load(open(batch_path+'batch3.pkl','rb'))
    # remove noisy channels from batch3
    del batch3['b3c37']
    del batch3['b3c2']
    del batch3['b3c23']
    del batch3['b3c32']
    del batch3['b3c42']
    del batch3['b3c43']

    bat_dict = {**batch1, **batch2, **batch3}

    if is_save:
        pkl_file = os.path.join(batch_path, save_name)
        if not os.path.exists(pkl_file):
            with open(pkl_file, 'wb') as f:
                pickle.dump(bat_dict, f)

    return bat_dict

def build_batch(matFilename, batch_name, key_name, save_path=None):
    if save_path is None:
        save_path = os.path.split(matFilename)[1]
    
    f = h5py.File(matFilename)
    batch = f['batch']
    # batch 1 
    num_cells = batch['summary'].shape[0]
    bat_dict = {}
    for i in range(num_cells):
        cl = f[batch['cycle_life'][i,0]].value
        policy = f[batch['policy_readable'][i,0]].value.tobytes()[::2].decode()
        summary_IR = np.hstack(f[batch['summary'][i,0]]['IR'][0,:].tolist())
        summary_QC = np.hstack(f[batch['summary'][i,0]]['QCharge'][0,:].tolist())
        summary_QD = np.hstack(f[batch['summary'][i,0]]['QDischarge'][0,:].tolist())
        summary_TA = np.hstack(f[batch['summary'][i,0]]['Tavg'][0,:].tolist())
        summary_TM = np.hstack(f[batch['summary'][i,0]]['Tmin'][0,:].tolist())
        summary_TX = np.hstack(f[batch['summary'][i,0]]['Tmax'][0,:].tolist())
        summary_CT = np.hstack(f[batch['summary'][i,0]]['chargetime'][0,:].tolist())
        summary_CY = np.hstack(f[batch['summary'][i,0]]['cycle'][0,:].tolist())
        summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
                    summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
                    'cycle': summary_CY}
        cycles = f[batch['cycles'][i,0]]
        cycle_dict = {}
        for j in range(cycles['I'].shape[0]):
            I = np.hstack((f[cycles['I'][j,0]].value))
            Qc = np.hstack((f[cycles['Qc'][j,0]].value))
            Qd = np.hstack((f[cycles['Qd'][j,0]].value))
            Qdlin = np.hstack((f[cycles['Qdlin'][j,0]].value))
            T = np.hstack((f[cycles['T'][j,0]].value))
            Tdlin = np.hstack((f[cycles['Tdlin'][j,0]].value))
            V = np.hstack((f[cycles['V'][j,0]].value))
            dQdV = np.hstack((f[cycles['discharge_dQdV'][j,0]].value))
            t = np.hstack((f[cycles['t'][j,0]].value))
            cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V':V, 'dQdV': dQdV, 't':t}
            cycle_dict[str(j)] = cd
            
        cell_dict = {'cycle_life': cl, 'charge_policy':policy, 'summary': summary, 'cycles': cycle_dict}
        key = key_name + str(i)
        bat_dict[key] = cell_dict

    # save_path = '/Users/ivan_zorin/Documents/AIRI/data/batt'
    with open(os.path.join(save_path, batch_name),'wb') as fp:
            pickle.dump(bat_dict,fp)
            print(batch_name, 'saved')

def main(path):
    # save_path = path
    matFilenames = [
        '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
        '2017-06-30_batchdata_updated_struct_errorcorrect.mat',
        '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
    ]

    matFilema_pathes = [os.path.join(path, matFilename) for matFilename in matFilenames]


    batch_names = ['batch1.pkl', 'batch2.pkl', 'batch3.pkl']
    key_names = ['b1c', 'b2c', 'b3c']


    for matFilename, batch_name, key_name in zip(matFilema_pathes, batch_names, key_names):
        print('building', batch_name)
        build_batch(matFilename, batch_name, key_name, path)

    # combine batches 
    load_battery_data(path)
    print('bathces were combined and saved')


if __name__ == '__main__':
    path = '/Users/ivan_zorin/Documents/AIRI/data/batt_2/'
    main(path)


