#!/usr/bin/env python3
"""Stdlib admission precedes NumPy import and all scientific input verification."""
import time
COMMAND_STARTED=(time.process_time(),time.monotonic())
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description='New reproduction of post-hoc JOINT diagnostic; no inference without --execute-reproduction.')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-inputs',action='store_true')
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--execute-reproduction',action='store_true')
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--training-assets',type=Path,required=True)
    args=parser.parse_args()
    if args.execute_reproduction:
        if args.output is None: parser.error('--output is required for reproduction execution')
        from jointaudit.admission import Admission
        admission=Admission(args.output,started=COMMAND_STARTED)
        try:
            # Earliest safe stdlib boundary: before output reservation, NumPy,
            # reading freeze/input files, source hashing, or scientific arrays.
            admission.install_limits(); admission.reserve_output()
            from jointaudit import runner
            runner.RAW=args.training_assets
            frozen=json.loads(args.freeze.read_text()); runner.verify_freeze(frozen)
            if frozen!=runner.make_review_freeze(): raise ValueError('portable source/config/input freeze incomplete or changed')
            runner.execute(frozen,args.output,admission=admission)
            print('Diagnostic artifacts finalized; success also requires observed process exit 0.',flush=True)
            admission.check()
        except BaseException as exc:
            if not admission.record_failure(exc):
                sys.stderr.write('Failure receipt unavailable; preserve nonzero exit and all partial files.\n')
            raise
    else:
        # Read-only review modes do not run the scientific command.
        from jointaudit import runner
        runner.RAW=args.training_assets
        if args.freeze_inputs:
            runner.write_once(args.freeze,runner.make_review_freeze())
            print('Source/input freeze saved. NO checkpoint arrays loaded, images decoded, or inference run.')
        else:
            frozen=json.loads(args.freeze.read_text()); runner.verify_freeze(frozen)
            if frozen!=runner.make_review_freeze(): raise ValueError('portable source/config/input freeze incomplete or changed')
            print('Source/input hashes match. NO checkpoint arrays loaded, images decoded, or inference run.')

if __name__=='__main__': main()
