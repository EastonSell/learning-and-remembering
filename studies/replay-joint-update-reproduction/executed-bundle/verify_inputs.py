#!/usr/bin/env python3
"""Hash-only package/TRAIN input check. No NumPy, decompression, or inference."""
import argparse
import json
from pathlib import Path
import portable

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-assets',type=Path)
    args=p.parse_args()
    files=portable.verify_package()
    report={'package_valid':True,'package_files_checked':len(files),
        'scientific_execution_performed':False,'checkpoint_arrays_loaded':False,
        'image_arrays_decoded':False,'training_assets_checked':False}
    if args.training_assets is not None:
        catalog=json.loads((portable.ROOT/'input-catalog.json').read_text())
        report['training_sha256']=portable.verify_training(args.training_assets,catalog['external_training_assets'])
        report['training_assets_checked']=True
    print(json.dumps(report,sort_keys=True,indent=2))

if __name__=='__main__': main()
