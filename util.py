import yaml
import sys
import os
import numpy as np
import platform

proj_dir = os.path.dirname(os.path.abspath((__file__)))
sys.path.append(proj_dir)
conf_fp = os.path.join(proj_dir, 'config.yaml')

with open(conf_fp)  as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
nodename = platform.uname().node
print(nodename)
file_dir = config['filepath'][nodename]