"""
dataset
---
Read data, make observations into trajectories, and split train samples
from trajectory prediction datasets.

Public Classes
---
```python
# Control structure of one training sample
(class) Agent

# Control structure of one video clip
(class) VideoClip

# Manager to directly make training datasets
(class) AgentManager

# Manager to help sample training data into `AgentManager`s
(class) DatasetManager
```
"""

from . import maps
from .__agent import Agent
from .__agentManager import AgentManager
from .__picker import AnnotationManager
from .__videoClip import VideoClip
from .__videoClipManager import VideoClipManager
from .__videoDataset import Dataset
from .__videoDatasetManager import DatasetManager
