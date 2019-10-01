# SAnD
## AAAI 2018 Attend and Diagnose: Clinical Time Series Analysis Using Attention Models

**Warning** This code is **UNOFFICIAL**.


[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16325/16790)


If you want to run this code.
You need download some dataset and write experimenting code.

```python
from comet_ml import Experiment
from SAnD.core.model import SAnD
from SAnD.utils.trainer import NeuralNetworkClassifier

model = SAnD( ... )
clf = NeuralNetworkClassifier( ... )
clf.fit( ... )
```
