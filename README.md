# SAnD
## AAAI 2018 Attend and Diagnose: Clinical Time Series Analysis Using Attention Models

---
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/29dd2dce4acd401e9d8554a4d834c8f3)](https://www.codacy.com/manual/KawashimaHirotaka/SAnD?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=KawashimaHirotaka/SAnD&amp;utm_campaign=Badge_Grade)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Warning** This code is **UNOFFICIAL**.

[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/16325/16790)

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
