
#%%

import os
import numpy as np
import pandas as pd


data_dir = "C:\\Users\\user\\.keras\\datasets\\cora"
data = pd.read_csv(os.path.join(data_dir, "cora.cites"),
                sep = "\t",
                names = ["target", "source"])



print(data)
print(f"shape={data.shape}")
# %%
print(data["target"])
# %%
# class_idx = {str(target): source for target,source in enumerate(sorted(data["target"].unique()))}
class_idx = {str(target): source for target,source in enumerate(data["target"])}
print(class_idx)
# %%
print(class_idx.__len__())
# %%
class_idx["5"]
# %%
