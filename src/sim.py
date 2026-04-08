# %%
from scmrepo.git import Git
import os

PACKAGE_ROOT = Git(root_dir=".").root_dir

# make sure output folder exists
os.makedirs(f"{PACKAGE_ROOT}/outputs", exist_ok=True)
# %%
print("Hello world!")

# %%