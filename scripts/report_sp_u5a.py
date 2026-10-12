"""Copyable text rendering of generated values, without document delivery."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.sp_u5a_common import *
def f(v):return f'{float(v):.6g}'
def main(section):
    if section=='D':
        for df in (8,16,32):
            q=readjson(OUT/f'sp_u5a_construction_{df}.json');print(df,q['wall_seconds'],q['statistics_seconds'],q['warnings'],q['statistics_sha256'],q['operators_sha256'])
    if section=='J':
        rr=csvread(OUT/'sp_u5a_stability.csv')
        for q in rr:
            print(q['condition'],q['snr_db'],q['dF'],'condition median/P05/P95/max', '/'.join(f(q['condition_number_'+k]) for k in ('median','p05','p95','maximum')),
                'resmax',q['normal_equation_residual_maximum'])
            for field in ('alpha','beta','coordinate_norm'):print(field,'mean/median/IQR/P05/P95/max','/'.join(f(q[field+'_'+k]) for k in ('mean','median','iqr','p05','p95','maximum')))
    if section=='L':
        for r in csvread(OUT/'sp_u5a_tails.csv'):print(r['condition'],r['snr_db'],r['contrast'],r['method'],r['top5_ids'],r['top5_differences'])
    if section=='cost':
        print(readjson(OUT/'sp_u5a_evaluation_gate.json'))
if __name__=='__main__':main(sys.argv[1])
