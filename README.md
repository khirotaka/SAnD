# SAnD
## AAAI 2018 Attend and Diagnose: Clinical Time Series Analysis Using Attention Models

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/29dd2dce4acd401e9d8554a4d834c8f3)](https://www.codacy.com/manual/KawashimaHirotaka/SAnD?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=KawashimaHirotaka/SAnD&amp;utm_campaign=Badge_Grade)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Warning** This code is **UNOFFICIAL**.

Paper: [Attend and Diagnose: Clinical Time Series Analysis Using Attention Models](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16325/16790)

If you want to run this code,
you need download some dataset and write experimenting code.

```python
from comet_ml import Experiment
from SAnD.core.model import SAnD
from SAnD.utils.trainer import NeuralNetworkClassifier

model = SAnD( ... )
clf = NeuralNetworkClassifier( ... )
clf.fit( ... )
```

## Installation
`git clone https://github.com/khirotaka/SAnD.git`

## Requirements
*   Python 3.6
*   Comet.ml
*   PyTorch v1.1.0 or later

## Simple Usage
Here's a brief overview of how you can use this project to help you solve the classification task.

### Download this project
First, create an empty directory.  
In this example, I'll call it "playground".  
Run the `git init` & `git submodule add` command to  register SAnD project as a submodule.

```shell script
$ mkdir playground/
$ cd playground/
$ git init
$ git submodule add https://github.com/khirotaka/SAnD.git
```

Now you're ready to use `SAnD` in your project.

### Preparing the Dataset
Prepare the data set of your choice.  
Remember that the input dimension to the SAnD model is basically three dimensions of `[N, seq_len, features]`.

This example shows how to use `torch.randn()` as a pseudo dataset.

```python
from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from SAnD.core.model import SAnD
from SAnD.utils.trainer import NeuralNetworkClassifier


x_train = torch.randn(1024, 256, 23)    # [N, seq_len, features]
x_val = torch.randn(128, 256, 23)       # [N, seq_len, features]
x_test =  torch.randn(512, 256, 23)     # [N, seq_len, features]

y_train = torch.randint(0, 9, (1024, ))
y_val = torch.randint(0, 9, (128, ))
y_test = torch.randint(0, 9, (512, ))


train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=128)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)
```

Note:  
In my experience, I have a feeling that `SAnD` is better at problems with a large number of `features`.

### Training SAnD model using Trainer
Finally, train the SAnD model using the included `NeuralNetworkClassifier`.  
Of course, you can also have them use a well-known training tool such as [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/).  
The included `NeuralNetworkClassifier` depends on the [comet.ml](https://www.comet.ml)'s logging service.

```python
in_feature = 23
seq_len = 256
n_heads = 32
factor = 32
num_class = 10
num_layers = 6

clf = NeuralNetworkClassifier(
    SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
    nn.CrossEntropyLoss(),
    optim.Adam, optimizer_config={"lr": 1e-5, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4},
    experiment=Experiment()
)

# training network
clf.fit(
    {"train": train_loader,
     "val": val_loader},
    epochs=200
)

# evaluating
clf.evaluate(test_loader)

# save
clf.save_to_file("save_params/")
```

For the actual task, choose the appropriate hyperparameters for your model and optimizer.

### Regression Task
There are two ways to use SAnD in a regression task.

1.  Specify the number of output dimensions in `num_class`.
2.  Inherit class SAnD and overwrite `ClassificationModule` with `RegressionModule`.

I would like to introduce a second point.

```python
from SAnD.core.model import SAnD
from SAnD.core.modules import RegressionModule


class RegSAnD(SAnD):
    def __init__(self, *args, **kwargs):
        super(RegSAnD, self).__init__(*args, **kwargs)
        d_model = kwargs.get("d_model")
        factor = kwargs.get("factor")
        output_size = kwargs.get("n_class")    # output_size

        self.clf = RegressionModule(d_model, factor, output_size)


model = RegSAnD(
    input_features=..., seq_len=..., n_heads=..., factor=...,
    n_class=..., n_layers=...
)
```

The contents of both ClassificationModule and RegressionModule are almost the same, so the 1st is recommended.

Please let me know when my code has been used to bring products or research results to the world.   
It's very encouraging :)

## Author
Hirotaka Kawashima (川島 寛隆)

## License
Copyright (c) 2019 Hirotaka Kawashima  
Released under the MIT license
