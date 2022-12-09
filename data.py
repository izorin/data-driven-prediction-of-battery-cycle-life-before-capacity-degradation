import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os 
import pickle
from collections import defaultdict
from functools import partial
from build_batches import load_battery_data
from tqdm.auto import tqdm
from scipy import interpolate

# load_battery_data
'''
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
'''


class PolyApproximator:
    def __init__(self, degree=2):
        self.degree = degree
    
    def fit(self, x, y):
        self.poly = np.polyfit(x, y, deg=self.degree)
        return self

    def __call__(self, x):
        return np.polyval(self.poly, x)


class SplineApproximator:
    def __init__(self, degree=2, knot_step=10):
        self.degree = degree
        self.knot_step = knot_step

    def fit(self, x, y):
        t = x[x[1]:x[-1]:self.knot_step]
        t = np.r_[(x[0],) * (self.degree+1), t, (x[-1],) * (self.degree+1)]

        self.spl = interpolate.make_lsq_spline(x, y, t, self.degree)
        return self

    def __call__(self, x):
        return self.spl(x)

class BattData(Dataset):

    def __init__(self, path, data_name='batt_data.pkl', hi_approximator=None, hi_approximator_kwargs={},
                feature_len_threshold=2, is_save_data=True):

        # CONSTANTS 
        self._NOMINAL_CAPACITY = 1.1 # Ah set by the battery manufacturer
        self._DEATH_PERCENT_CAPACITY = 0.8 # industry standard -- reaching 80% of nominal capacity means death of a battery
        self._DEATH_CAPACITY = self._DEATH_PERCENT_CAPACITY * self._NOMINAL_CAPACITY # Ah 
        self.REFERENCE_BATTERIES = ['b1c0', 'b1c1', 'b1c2'] # these batteries were charged with settings recommended by their manufacturer
        self.REF_BAT = 'b1c1' # the one without outliers
        # ============

        self.path = path
        self.data_name = data_name
        self.hi_approximator = hi_approximator if hi_approximator is not None else PolyApproximator
        self.hi_approximator_kwargs = hi_approximator_kwargs
        self.feature_len_threshold = feature_len_threshold
        self.is_save_data = is_save_data

        self.length = None

        self.data = []
        self.batt_names = []
        self.batt_life_cycles = {}
        self.batt_charge_policy = {}
        self.batt_summary = {}
        self.batt_to_idxs = defaultdict(list)

        self.build_dataset()
        self.feature_names = list(self.data[0][-1].keys())
        _ = self.capacity_fade_approximation()


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.get_item(idx)


    def load_dataset(self):
        if os.path.exists(os.path.join(self.path, self.data_name)):
            with open(os.path.join(self.path, self.data_name), 'rb') as f:
                batt_data = pickle.load(f)
                print('data loaded from pickle file')

        else:
            batt_data = load_battery_data(self.path, save_name=self.data_name, is_save=self.is_save_data)

        return batt_data

    
    def build_dataset(self):
        batt_data = self.load_dataset()
        print('start building dataset')
        self.length = 0
        idx = 0
        bad_batt_names = []
        for batt_name in batt_data.keys():
            self.batt_names += [batt_name]
            self.batt_life_cycles[batt_name] = batt_data[batt_name]['cycle_life'].item()
            self.batt_charge_policy[batt_name] = batt_data[batt_name]['charge_policy']
            self.batt_summary[batt_name] = batt_data[batt_name]['summary']

            for cycle in batt_data[batt_name]['cycles'].keys():

                features = {}
                lengths = []
                for key, value in batt_data[batt_name]['cycles'][cycle].items():
                    features[key] = np.array(value, dtype=np.float64)
                    lengths.append(len(value))

                max_len = max(lengths)
                if max_len == self.feature_len_threshold:
                # if max_len == 2:
                    bad_batt_names.append(batt_name)
                else:
                    sample_tuple = [batt_name, int(cycle), max_len, features]

                    self.data.append(sample_tuple)
                    self.batt_to_idxs[batt_name].append(idx)
                    idx += 1
                    self.length += 1
        
        self.fix_first_cycles(bad_batt_names)

    def fix_first_cycles(self, batt_names):
        for batt_name in batt_names:
            self.batt_summary[batt_name]['QC'] = self.batt_summary[batt_name]['QC'][1:]
            self.batt_summary[batt_name]['QD'] = self.batt_summary[batt_name]['QD'][1:]



    def scale_fn(self, x):
        x = (x - self._DEATH_CAPACITY) / (self._NOMINAL_CAPACITY - self._DEATH_CAPACITY)
        return np.clip(x, a_min=0, a_max=1)


    def capacity_fade_approximation(self):

        self.batt_fade_approx = {}
        for batt_name, value in self.batt_summary.items():
            QC = self.scale_fn(value['QC'])
            QD = self.scale_fn(value['QD'])

            qc_approx = self.hi_approximator(**self.hi_approximator_kwargs).fit(np.arange(len(QC)), QC)
            qd_approx = self.hi_approximator(**self.hi_approximator_kwargs).fit(np.arange(len(QD)), QD)

            self.batt_fade_approx[batt_name] = {'QC': qc_approx,
                                                'QD': qd_approx
                                                }
        return self.batt_fade_approx

    def get_reference_hi(self, cycle):
        return self.batt_fade_approx[self.REF_BAT]['QD'](cycle)

    def get_item(self, idx): 
        return self.data[idx]

    def get_indices_by_batt(self, batt_list=[]):
        # returns data indices of batteries in `batt_list`
        pass


    def train_test_split(self, batts=[]):
        # return indices of train and test splits such that batteries in `batts` are all in test subset
        pass


                


class BattDataloader():
    def __init__(self, dataset, load_features=[], batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, generator=None, prefetch_factor=2, persistent_workers=False):

        self.dataset = dataset
        self.load_features = load_features if len(load_features) > 0 else self.dataset.feature_names

        collate_fn = partial(self._collate_fn, **{}) 

        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, timeout=timeout, worker_init_fn=worker_init_fn, generator=generator, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)


    def pad(self, features, pad_len):
        features_array = []
        for sample in features:
            sample_array = np.stack([np.pad(sample[key], (0, pad_len - sample[key].shape[0]), 'constant', constant_values=0) for key in self.load_features])
            features_array.append(sample_array)

        return np.stack(features_array)
                

    def _collate_fn(self, batch):
        batt_names = [sample[0] for sample in batch]
        cycles = [sample[1] for sample in batch]
        max_lengths = [sample[2] for sample in batch]
        features = [sample[3] for sample in batch]
        ruls = [dataset.batt_life_cycles[batt_name] - cycle for batt_name, cycle in zip(batt_names, cycles)]
        his_approx = [dataset.batt_fade_approx[batt_name]['QD'](cycle).item() for batt_name, cycle in zip(batt_names, cycles)]
        reference_his = dataset.get_reference_hi(cycles)

        ruls = torch.tensor(ruls)
        his_approx = torch.tensor(his_approx)
        cycles = torch.tensor(cycles)
        reference_his = torch.tensor(reference_his)

        max_len = max(max_lengths)
        features = self.pad(features, max_len)
        features = torch.tensor(features)


        return batt_names, cycles, ruls, his_approx, reference_his, features

    def __iter__(self):
        return self.dataloader.__iter__()

    def __len__(self):
        return len(self.dataloader)




if __name__ == '__main__':
    data_path = '/Users/ivan_zorin/Documents/AIRI/data/batt/' # change the path 

    # polynomial approximation 
    # hi_approximator = PolyApproximator
    # hi_approximator_kwargs = {'degree': 3}

    # LSQ B-Spline approximation 
    hi_approximator = SplineApproximator
    hi_approximator_kwargs = {'degree': 3, 'knot_step': 100}
    # dataset init
    dataset = BattData(data_path, 'batt_data.pkl', hi_approximator=hi_approximator, hi_approximator_kwargs=hi_approximator_kwargs,  is_save_data=False)



    
    # dataloader init
    shuffle = True
    dataloader = BattDataloader(dataset, load_features=['I', 'V', 'T'], batch_size=2, shuffle=shuffle)
    
    batch = next(iter(dataloader))
    batt_names, cycles, ruls, his_approx, reference_his, features = batch
    print(f'''
    Batch structure
        batteries: {batt_names},
        cycles: {cycles},
        RULs: {ruls},
        approximation of HIs: {his_approx},
        HI of reference battery: {reference_his}, 
        features shape: {features.shape}
    ''')

    t = tqdm(dataloader)
    for i, batch in enumerate(t):
        pass

