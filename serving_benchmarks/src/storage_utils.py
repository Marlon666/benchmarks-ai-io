import os
from .outputs import write_csv_local, write_yaml_local

def write_csv(base_dir: str, object_name: str, rows):
    out = os.path.join(base_dir, object_name)
    write_csv_local(out, rows)

def write_yaml(base_dir: str, object_name: str, data):
    out = os.path.join(base_dir, object_name)
    write_yaml_local(out, data)