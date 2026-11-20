"""Atomic JSON receipt writer shared by simulation drivers.

Output formatting is intentionally identical to the frozen campaign.
"""
import json
import os
from pathlib import Path

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n")
    os.replace(tmp,path)
