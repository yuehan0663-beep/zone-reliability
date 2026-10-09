"""Recompute workflow outputs from an explicitly authorized local dataset."""
import argparse
from pathlib import Path
import time
import json
from zonerel.io import read_dataset
from zonerel.pipeline import run_workflow

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--sheet',default=None)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--well-col',default='well')
    parser.add_argument('--group-col',default='zone')
    parser.add_argument('--attribution',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        raise ValueError('Confidential output must be outside this repository')
    start=time.perf_counter()
    data=read_dataset(args.input,args.sheet,args.group_col,args.well_col)
    # Supply original aliases in the private source when exact bootstrap ordering is needed.
    tables=run_workflow(data,{'run_attribution':args.attribution,
        'progress':lambda ds,m:print(ds,m,flush=True)})
    args.output.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():table.to_csv(args.output/(name+'.csv'),index=False)
    (args.output/'runtime.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'input_rows':len(data)}),encoding='utf-8')

if __name__=='__main__':main()
