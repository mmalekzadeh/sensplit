# SenSplit
This python package helps to splits a sensor dataset (in Pandas DataFrame format) into train and test sets.

-----------------------------------------
## Homepage:
https://github.com/mmalekzadeh/sensplit

-----------------------------------------
## Installation 
Install SenSplit from PyPI (recommended):
```sh
pip install sensplit
```
- **Alternatively: install SenSplit from the GitHub source:**

First, clone SenSplit using `git`:

```sh
git clone https://github.com/mmalekzadeh/sensplit
```

 Then, `cd` to the SenSplit folder and run the install command:
```sh
cd sensplit
python setup.py install
```
-----------------------------------------

## Example:


```python
#%pip install sensplit==0.0.5
#%pip install pandas==0.25.1
```

We assume you already have a dataset of human activity recognition in a Pandas DataFrame format.

> For example, here I use the code of MotionSense dataset to create the dataset.
source: https://github.com/mmalekzadeh/motion-sense

#### dataset


```python

import numpy as np
import pandas as pd

def get_ds_infos():
    """
    Read the file includes data subject information.
    
    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]
    
    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """ 

    dss = pd.read_csv("data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")
    
    return dss

def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.
    
    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x",t+".y",t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])
    print(dt_list)
    return dt_list


def creat_time_series(folder_name, dt_list, act_labels, trial_codes, mode="mag", labeled=True):
    """
    Args:
        folder_name: one of 'A_DeviceMotion_data', 'B_Accelerometer_data', or C_Gyroscope_data
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be 'raw' which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be 'mag' which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.
    
    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
    else:
        dataset = np.zeros((0,num_data_cols))
        
    ds_list = get_ds_infos()
    
    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = folder_name+'/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                    else:
                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                    vals = vals[:,:num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                            sub_id-1,
                            ds_list["weight"][sub_id-1],
                            ds_list["height"][sub_id-1],
                            ds_list["age"][sub_id-1],
                            ds_list["gender"][sub_id-1],
                            trial          
                           ]]*len(raw_data), dtype=int)
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset,vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]
            
    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
    
    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset
#________________________________


ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
    ACT_LABELS[5]:[5,13]
}

## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
## attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
folder_name = 'A_DeviceMotion_data'
sdt = ["attitude", "gravity", "rotationRate", "userAcceleration"]
print("[INFO] -- Selected sensor data types: "+str(sdt))    
act_labels = ACT_LABELS [0:6]
print("[INFO] -- Selected activites: "+str(act_labels))    
trial_codes = [TRIAL_CODES[act] for act in act_labels]
dt_list = set_data_types(sdt)
dataset = creat_time_series(folder_name, dt_list, act_labels, trial_codes, mode="raw", labeled=True)
print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))    
dataset.head()
```

    [INFO] -- Selected sensor data types: ['attitude', 'gravity', 'rotationRate', 'userAcceleration']
    [INFO] -- Selected activites: ['dws', 'ups', 'wlk', 'jog', 'std', 'sit']
    [['attitude.roll', 'attitude.pitch', 'attitude.yaw'], ['gravity.x', 'gravity.y', 'gravity.z'], ['rotationRate.x', 'rotationRate.y', 'rotationRate.z'], ['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z']]
    [INFO] -- Data subjects' information is imported.
    [INFO] -- Creating Time-Series
    [INFO] -- Shape of time-Series dataset:(1412865, 19)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>attitude.roll</th>
      <th>attitude.pitch</th>
      <th>attitude.yaw</th>
      <th>gravity.x</th>
      <th>gravity.y</th>
      <th>gravity.z</th>
      <th>rotationRate.x</th>
      <th>rotationRate.y</th>
      <th>rotationRate.z</th>
      <th>userAcceleration.x</th>
      <th>userAcceleration.y</th>
      <th>userAcceleration.z</th>
      <th>act</th>
      <th>id</th>
      <th>weight</th>
      <th>height</th>
      <th>age</th>
      <th>gender</th>
      <th>trial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.528132</td>
      <td>-0.733896</td>
      <td>0.696372</td>
      <td>0.741895</td>
      <td>0.669768</td>
      <td>-0.031672</td>
      <td>0.316738</td>
      <td>0.778180</td>
      <td>1.082764</td>
      <td>0.294894</td>
      <td>-0.184493</td>
      <td>0.377542</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>102.0</td>
      <td>188.0</td>
      <td>46.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.527992</td>
      <td>-0.716987</td>
      <td>0.677762</td>
      <td>0.753099</td>
      <td>0.657116</td>
      <td>-0.032255</td>
      <td>0.842032</td>
      <td>0.424446</td>
      <td>0.643574</td>
      <td>0.219405</td>
      <td>0.035846</td>
      <td>0.114866</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>102.0</td>
      <td>188.0</td>
      <td>46.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.527765</td>
      <td>-0.706999</td>
      <td>0.670951</td>
      <td>0.759611</td>
      <td>0.649555</td>
      <td>-0.032707</td>
      <td>-0.138143</td>
      <td>-0.040741</td>
      <td>0.343563</td>
      <td>0.010714</td>
      <td>0.134701</td>
      <td>-0.167808</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>102.0</td>
      <td>188.0</td>
      <td>46.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.516768</td>
      <td>-0.704678</td>
      <td>0.675735</td>
      <td>0.760709</td>
      <td>0.647788</td>
      <td>-0.041140</td>
      <td>-0.025005</td>
      <td>-1.048717</td>
      <td>0.035860</td>
      <td>-0.008389</td>
      <td>0.136788</td>
      <td>0.094958</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>102.0</td>
      <td>188.0</td>
      <td>46.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.493941</td>
      <td>-0.703918</td>
      <td>0.672994</td>
      <td>0.760062</td>
      <td>0.647210</td>
      <td>-0.058530</td>
      <td>0.114253</td>
      <td>-0.912890</td>
      <td>0.047341</td>
      <td>0.199441</td>
      <td>0.353996</td>
      <td>-0.044299</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>102.0</td>
      <td>188.0</td>
      <td>46.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Splitting 
Now, I have the dataset in the right format and I want to split it into two subsets: training and testing.

> here I select `trials` setting and choose those trials with a code number less than 10 as the training data


```python
sorted(dataset['trial'].unique())
```




    [1.0,
     2.0,
     3.0,
     4.0,
     5.0,
     6.0,
     7.0,
     8.0,
     9.0,
     11.0,
     12.0,
     13.0,
     14.0,
     15.0,
     16.0]



> Here, I want data of each user (`id`) to be splitted into train (`trial<10.`) and test ('trial>10.`).
So, I choose `labels = `("id","trial")`. The order matters!

### Main Part


```python
from sensplit.dataframe_splitter import DataFrameSplitter
dfs = DataFrameSplitter(method="trials")

train_data, test_data = dfs.train_test_split(dataset = dataset,
                                             labels = ("id","trial"), 
                                             trial_col='trial', 
                                             train_trials=[1.,2.,3.,4.,5.,6.,7.,8.,9.],
                                             verbose=2)
train_data.shape, test_data.shape
```

    Seg_Shape:(790, 19) | TrainData:(1081446, 19) | TestData:(331419, 19) | ('id', 'trial'):(23.0, 16.0) | progress:100%.




    ((1081446, 19), (331419, 19))




```python
Features = dataset.columns[:-7]
labels_or_info = dataset.columns[-7:]
print("Features are {} \n Labels or Info are {}".format(Features, labels_or_info))
```

    Features are Index(['attitude.roll', 'attitude.pitch', 'attitude.yaw', 'gravity.x',
           'gravity.y', 'gravity.z', 'rotationRate.x', 'rotationRate.y',
           'rotationRate.z', 'userAcceleration.x', 'userAcceleration.y',
           'userAcceleration.z'],
          dtype='object') 
     Labels or Info are Index(['act', 'id', 'weight', 'height', 'age', 'gender', 'trial'], dtype='object')



```python
x_train = train_data[Features]
y_train = train_data[labels_or_info]

x_test = test_data[Features]
y_test = test_data[labels_or_info]

print("Train: x={}, y={}\nTest:  x={}, y={}".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
```

    Train: x=(1081446, 12), y=(1081446, 7)
    Test:  x=(331419, 12), y=(331419, 7)



```python
dataset_name = "MotionSense" 
x_train.to_csv(dataset_name+"_x_train.csv", index=False)
x_test.to_csv(dataset_name+"_x_test.csv", index=False)
y_train.to_csv(dataset_name+"_y_train.csv", index=False)
y_test.to_csv(dataset_name+"_y_test.csv", index=False)
```

It's all done :)

