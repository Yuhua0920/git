"""Read-only A-MMSE source/interface audit; no neural model is implemented or run.

The resource bound is derived from parsed author source before allocation.
Only the explicitly permitted historical inputs listed below are read.
"""
import ast
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs'
SOURCES = OUT / 'sp_u6a_sources'
ALLOWED = {
    'src/pilots.py', 'src/cdl_benchmark.py', 'src/spatial_information.py',
    'outputs/sp_u5a_geometry_16.npz', 'outputs/sp_u4a_clean_gate.json',
    'docs/sp_u1_closest_work_matrix.md', 'docs/sp_u1_modern_baseline_gate.md',
    'outputs/sp_u1_author_code_access.json',
    'outputs/sp_u2a_sources/A-MMSE__AMMSE.py.txt',
    'outputs/sp_u2a_sources/A-MMSE__main.py.txt',
    'outputs/sp_u2a_sources/A-MMSE__data.py.txt',
    'outputs/sp_u2a_sources/A-MMSE__train_ra.py.txt',
    'outputs/sp_u2a_sources/README.md.txt',
}
READS = set()

def audit_read(path):
    path = Path(path).resolve()
    rel = path.relative_to(ROOT).as_posix()
    if not (rel in ALLOWED or rel.startswith('outputs/sp_u6a_') or rel.startswith('scripts/sp_u6a_') or rel.startswith('tests/test_sp_u6a_')):
        raise PermissionError('SP-U6A firewall rejected: ' + rel)
    READS.add(rel)
    return path.read_bytes()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def save(name, data):
    (OUT / ('sp_u6a_' + name + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def source(name):
    return audit_read(SOURCES / name).decode('utf-8')

def dense_decoder_bound(text, L, output_dim, temporal_dim):
    """Count the actual seven Dense kernels, biases, and three layer norms."""
    tree = ast.parse(text)
    decoder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ResidualFC')
    init = next(n for n in decoder.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    dense = [n for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'Dense']
    norms = [n for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'LayerNormalization']
    expression = ast.dump(ast.parse('2 * num_pilot * d_model_out', mode='eval').body)
    if len(dense) != 7 or len(norms) != 3 or any(ast.dump(n.args[0]) != expression for n in dense):
        raise ValueError('Author decoder no longer matches the frozen bound')
    call = next(n for n in decoder.body if isinstance(n, ast.FunctionDef) and n.name == 'call')
    flat = next(n for n in ast.walk(call) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'x_flat' for t in n.targets))
    if 'self.seq_length * self.d_model' not in ast.unparse(flat.value):
        raise ValueError('Decoder flatten contract changed')
    q, u = 2 * L * output_dim, 2 * L * temporal_dim
    kernels = 2 * u * q + 5 * q * q
    parameters = kernels + 7 * q + 2 * len(norms) * q
    return {'L': L, 'output_dim': output_dim, 'temporal_dim': temporal_dim, 'q': q, 'u': u,
            'dense_kernels': len(dense), 'layer_norms': len(norms), 'kernel_parameters': kernels,
            'decoder_parameters_including_bias_and_norm': parameters,
            'float32_kernel_bytes': kernels * 4, 'float32_decoder_bytes': parameters * 4,
            'float32_decoder_GiB': parameters * 4 / 2**30,
            'Adam_weight_gradient_m_v_bytes_lower_bound': parameters * 16,
            'unconditional_five_square_kernel_bytes': 5 * q * q * 4,
            'batch_independent': True, 'measured_allocation': False}

def packing_identity():
    """Independent algebra calibration, explicitly NOT an author-network smoke."""
    from scipy.io import loadmat, savemat
    L, D = 3, 5
    g = (np.arange(2 * L * D, dtype=np.float32).reshape(1, 2, L, D) - 7) / 13
    pilots = np.array([[1+2j, -3+4j, 2-1j], [-1+3j, 4-2j, 2+5j]], dtype=np.complex64)
    packed = np.stack([pilots.real, pilots.imag], axis=1)[..., None]
    w = (g[0, 0] + 1j * g[0, 1]).T
    path = OUT / 'sp_u6a_algebra_filter.mat'
    savemat(path, {'global_filter': g})
    loaded = loadmat(path)['global_filter']
    w_loaded = (loaded[0, 0] + 1j * loaded[0, 1]).T
    direct = pilots @ w_loaded.T
    nested = np.array([[sum(complex(g[0, 0, l, d], g[0, 1, l, d]) * complex(row[l]) for l in range(L)) for d in range(D)] for row in pilots])
    mismatch = float(np.max(np.abs(direct - nested)))
    targets = direct + np.array([1+1j, -1j, 2, -2+1j, 1], dtype=np.complex64)
    nmse = float(np.sum(np.abs(direct - targets)**2) / np.sum(np.abs(targets)**2))
    mse_power = float(np.mean(np.abs(direct - targets)**2) / np.mean(np.abs(targets)**2))
    return {'kind': 'packing/transpose/serialization algebra only', 'author_network_executed': False,
            'array_roundtrip_exact': bool(np.array_equal(loaded, g)),
            'pilot_roundtrip_exact': bool(np.array_equal(packed[:, 0, :, 0] + 1j * packed[:, 1, :, 0], pilots)),
            'transpose_is_not_Hermitian': bool(not np.allclose(pilots @ w.conj().T, direct)),
            'direct_vs_scalar_max_abs_error': mismatch, 'complex64_atol': 1e-5,
            'NMSE_sum_ratio': nmse, 'NMSE_mean_ratio': mse_power,
            'passed': mismatch < 1e-5 and bool(np.isclose(nmse, mse_power, rtol=1e-7, atol=0)),
            'full_training_or_performance_measurement': False}

class PlainText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
    def handle_data(self, text):
        self.items.append(text)

def text_of_html(text):
    parser = PlainText()
    parser.feed(text)
    return ' '.join(' '.join(parser.items).split())

def win_memory():
    from ctypes import wintypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [(n, ctypes.c_ulonglong) for n in ['total_phys', 'avail_phys', 'total_page', 'avail_page', 'total_virtual', 'avail_virtual', 'avail_extended']]
    class ProcessMemory(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in ['peak_working_set', 'working_set', 'quota_peak_paged', 'quota_paged', 'quota_peak_nonpaged', 'quota_nonpaged', 'pagefile', 'peak_pagefile']]
    status, proc = MemoryStatus(), ProcessMemory()
    status.length, proc.cb = ctypes.sizeof(status), ctypes.sizeof(proc)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)) or not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(proc), proc.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'actual_worker_pid': os.getpid(), 'total_RAM_bytes': status.total_phys, 'available_RAM_bytes_snapshot': status.avail_phys,
            'audit_worker_peak_working_set_bytes': proc.peak_working_set, 'audit_worker_peak_commit_bytes': proc.peak_pagefile}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    start = time.perf_counter()
    pin = json.loads(audit_read(OUT / 'sp_u6a_source_freeze.json'))
    for row in pin['sources']:
        assert sha(audit_read(SOURCES / row['path'])) == row['sha256']
    paper = text_of_html(source('2506.00452v5.html'))
    extension = text_of_html(source('2608.15945v1.html'))
    abstract = text_of_html(source('2506.00452_abs.html'))
    assert 'this version, v5' in abstract and '19 Jul 2026' in abstract
    title = 'Learning MMSE Filters for OFDM Channel Estimation: Attention Transformer Gains at Linear Inference'
    assert title in abstract
    core = source('A-MMSE__AMMSE.py')
    native = dense_decoder_bound(core, 72, 1008, 1008)
    project = dense_decoder_bound(core, 64, 4096, 4096)
    resource_pass = project['float32_decoder_bytes'] <= 8 * 2**30
    assert not resource_pass
    evidence = {}
    for a, b in [('IV-B 3 Training Method', 'IV-C Discussions'), ('V Rank-Adaptive A-MMSE', 'VI Simulation Results'), ('VI-A Simulation Setup', 'VI-B Normalized')]:
        i = paper.rfind(a)
        j = paper.find(b, i+len(a))
        evidence[a] = paper[i:j]
    save('paper_evidence', {'paper': evidence, 'extension_supervision': extension[extension.rfind('II-B Clean-Label-Free Setting'):extension.rfind('III Ridge-Regularized A-MMSE')]})
    previous = json.loads(audit_read(ROOT / 'outputs/sp_u1_author_code_access.json'))[0]
    old_docs = '\n'.join(audit_read(ROOT / name).decode('utf-8') for name in ['docs/sp_u1_closest_work_matrix.md', 'docs/sp_u1_modern_baseline_gate.md'])
    assert title in old_docs
    equality = {}
    for name in ['README.md', 'A-MMSE/AMMSE.py', 'A-MMSE/main.py', 'A-MMSE/data.py', 'A-MMSE/train_ra.py']:
        current_name = name.replace('/', '__')
        old_name = ROOT / ('outputs/sp_u2a_sources/' + current_name + '.txt')
        equality[name] = sha(audit_read(old_name)) == sha(audit_read(SOURCES / current_name))
    fields = [
        ('title', title, title, title, 'UNCHANGED'),
        ('training_mode', 'supervised true-H; attention learns fixed population filter', 'Eq16 noisy-pilot generator; IV-B3 fixed final filter after training', 'main default global-token trained with noisy-pilot/clean-H MSE', 'PAPER_CODE_TOKEN_RECIPE_UNSPECIFIED'),
        ('filter_sharing', 'global-token repo default; online path not verified', 'one final filter consistently applied to all test data', 'token branch dominates hybrid and sharing modes', 'UNCHANGED; alternatives excluded'),
        ('global_filter_definition', 'fixed exported training-population filter', 'fixed final filter from entire training dataset; token/aggregation recipe unstated', 'g_theta(tau); export without dataset averaging in default branch', 'RECIPE_GAP_REMAINS'),
        ('rank_adaptation', 'learned rank factors; whole-filter compression', 'fixed W times trained real U,V; factorized A,B', 'cascaded U_s,V_s then complex SVD A,B', 'compatible extension; no RA execution'),
        ('data_split', 'frame split author; project trajectory split required', '36000 train/4000 val from first40000; last4000 test', 'sklearn random frame val seed42; last4000 test', 'project split deviation disclosed'),
        ('pilot_layout', 'SISO72x14 DMRS', 'arbitrary patterns allowed; experiments comb2/t2,t11', 'hardcoded selector j+72*i, L36/72/108/144', 'SHAPE_ADAPTER'),
        ('loss', 'MSE/Huber alternatives noted', 'Eq28 MSE or Huber; both discussed', 'complex_mse; current default no Huber', 'MSE ONLY primary'),
        ('online_adaptation', 'plots are not executable online update evidence', 'fixed test filter; main paper no per-slot retraining', 'online results README only; no trainer update loop', 'EXCLUDED'),
        ('evaluation_path', 'saved fixed filter plus pilots', 'Eq15 and Eq3 full-grid estimate/error energy ratio', 'main exports g and predictions; test sanity and external saved-filter NMSE', 'algebra calibration only; author NN NOT_RUN')]
    save('ammse_version_ledger', {'old_commit': previous['commit'], 'current_commit': pin['commit'], 'same_commit': previous['commit'] == pin['commit'], 'current_vs_u2a_bytes_equal': equality,
         'old_understanding_source': 'SP-U1 literature-only docs, not experimental outcomes', 'old_PDF_byte_identity': 'NOT_VERIFIED: no old PDF in allowed source snapshot',
         'fields': [dict(field=f, old_sp_u1=o, current_paper=p, current_repo=r, amendment=a) for f,o,p,r,a in fields]})
    from src.pilots import build_fdm_pilot_layout
    layout = build_fdm_pilot_layout(2, 64, 16, 16, 4)
    observed = np.broadcast_to(layout.pilot_mask, (2,2,64,16)).reshape(-1)
    indices = np.flatnonzero(observed)
    frozen_bytes = audit_read(ROOT / 'outputs/sp_u5a_geometry_16.npz')
    import io
    with np.load(io.BytesIO(frozen_bytes)) as geometry:
        geometry_keys = list(geometry.files)
        # Compare actual selector if present; never load operator/experiment outcomes.
        comparisons = {k: bool(np.array_equal(geometry[k], getattr(layout, k))) for k in ['pilot_mask','pilot_union','data_mask','mute_mask','pilot_owner'] if k in geometry}
        selector_matches = {k: bool(np.array_equal(geometry[k], indices)) for k in ['selected_indices','selected'] if k in geometry}
    assert len(indices) == 64 and all(comparisons.values()) and all(selector_matches.values())
    save('geometry', {'flatten_order':'C[rx,tx,k,t]', 'full_index':'(((rx*2+tx)*64+k)*16+t)', 'pilot_indices':indices.tolist(), 'L':len(indices),
            'frozen_geometry_keys':geometry_keys, 'frozen_mask_comparisons':comparisons, 'selector_comparisons':selector_matches,
            'pilot_owner_per_symbol':[0,1], 'tx0_frequency':[0,16,32,48], 'tx1_frequency':[8,24,40,56], 'pilot_symbols':[0,4,8,12],
            'sample_rate_hz':1920000, 'FFT_times_spacing_hz':64*30000, 'candidate_classification':'CASE_2_INTERFACE_ONLY_NOT_EXECUTION_PASS'})
    save('algebra_identity', packing_identity())
    fairness = [
        ('offline clean environments','C30+B30+C120','C30','pooled C30+B30+C120'),
        ('clean frames','15000','5000','15000'),
        ('environment labels offline','used to construct BP/BM','source corpus origin; no network token','origin for split only; no network token'),
        ('noisy training observations','not used for frozen statistical construction','generated from same C30 clean frames','generated from same pooled clean frames'),
        ('target supervision','no B120 clean CSI','no B120 clean CSI','no B120 clean CSI'),
        ('training SNR knowledge','known sigma2 in filter construction','one model per SNR; known frozen sigma2','one model per SNR; known frozen sigma2'),
        ('online current pilots','64 complex','64 complex','64 complex'),
        ('online history','current+previous4','current only','current only'),
        ('target environment label','not used','not used','not used'),
        ('model storage form','base operator + compressed BP4/BM4','one complex4096x64 W per SNR','one complex4096x64 W per SNR'),
        ('inference form','(W0+alpha BP4+beta BM4)y; causal cross-pilot fitting','Wy','Wy')]
    save('fairness', {'rows':[dict(interface=i, LR4=l, Source=s, Prepared=p) for i,l,s,p in fairness],
           'future_trajectory_split':{'rule':'trajectory_id0..399 train,400..499 validation within each permitted environment', 'Source':[4000,1000], 'Prepared':[12000,3000]},
           'P_ref':9.941134822523972,'sigma2_10':0.9941134822523973,'sigma2_20':0.09941134822523973,
           'training_executed':False,'noise_generated':False,'labels_as_network_input':False,'B120_opened':False,
           'comparison_claim':'same per-frame pilot overhead; different temporal and offline-label interfaces'})
    source_rows = [
        ('input loader','SHAPE_ADAPTER','data.py:12-39','project joint-C-order adapter and trajectory split; no payload grid required'),
        ('pilot selector','SHAPE_ADAPTER','data.py:42-61; main.py:34-38','replace constant index set only; preserve frozen FDM owners'),
        ('reshape/packing','SHAPE_ADAPTER','data.py:18-26; AMMSE.py:15-22,177-181','explicit permutation from MATLABF to projectC; no Hermitian transpose'),
        ('attention encoder','CONFIG_ONLY','AMMSE.py:25-46,85-99,130-142','same two encoder blocks; dimension-driven joint output embedding; head interpretation open'),
        ('decoder','CONFIG_ONLY','AMMSE.py:49-81','same Dense topology; no row-wise or low-rank rescue; STOP-D'),
        ('global token path','UNCHANGED','AMMSE.py:122-150,298-314','freeze global token only; dropout suppressed by training=False'),
        ('loss','UNCHANGED','AMMSE.py:7-10; main.py:102-106','complex full-target MSE'),
        ('filter save','CONFIG_ONLY','main.py:135-139','same real1x2xLxD MAT representation; new isolated output name'),
        ('final filter application','UNCHANGED','main.py:47-53','complex W times current noisy pilots'),
        ('test sanity path','SHAPE_ADAPTER','main.py:153-166','trajectory validation; full-grid summed error/energy, no B120 this task')]
    save('source_audit', {'items':[dict(component=c,status=s,author_lines=l,required_diff=d) for c,s,l,d in source_rows],
          'actual_author_source_diff':'NONE; snapshots unchanged; proposed adapters NOT_IMPLEMENTED',
          'per_link_fallback':'UNFAIR_WEAKENING_RISK; NOT_IMPLEMENTED',
          'mode_freeze':'global-token; no hybrid/no directly trained W/no averaging fallback/no online update',
          'paper_global_token_recipe_explicit':False,
          'joint_embedding_proposal':'d_model1=64; d_model2=d_model_out=4096; two stage topology unchanged; not a validated paper-MIMO inductive bias',
          'heads_proposal':'not frozen: per-symbol pilot interpretation differs from original two-DMRS L/2 rule; resource bound independent of heads'})
    save('equation_inventory', {
        'author_native':{'h':'C^1008, vec_F(H), index=k+72*t','yp':'C^72, DMRS constant selector',
            'training_packing':'x=(B,2,L,1); concatenate all Re then all Im -> (B,2L,1)',
            'token':'tau=(1,2,L,1), trainable zeros initializer; no noisy sample enters token encoder',
            'generator':'g_theta(tau): (1,2,L,D); each batch uses identical filter',
            'complex_filter':'W=(g[0,0]+j*g[0,1])^T in C^(D x L); ordinary transpose',
            'loss':'mean_(samples,full coefficients) |W y_p-h_clean|^2',
            'deployment':'h_hat=W y_p; current pilots only; no clean H/network/environment token at inference',
            'encoder':'2L tokens -> Dense N -> encoder1 -> Dense D -> encoder2 -> flat ResidualFC -> g',
            'sharing_precedence':'global token > hybrid > sharing(2/1/0)',
            'token_training_dropout':'process_batch(global_token,training=False), even while optimizing',
            'paper_source_gap':'paper Eq16 writes observation-dependent generator; IV-B3 fixes final filter; no explicit global-token or aggregation recipe'},
        'joint_project_candidate':{'h':'C^4096, C-order[rx,tx,k,t]', 'yp':'C^64, rx-major/tx-major selected-index order',
            'S':'64x4096 binary one-hot selector; FDM mute prevents other-TX contamination',
            'observation':'yp=S h+epsilon for frozen unit pilot=1; general known unit phases require Dp^H derotation',
            'W':'C^(4096x64), unrestricted joint cross-link filter',
            'loss':'mean |W yp-h_clean|^2 across all4096 coefficients; no payload-only masking',
            'flatten_change':'pure coordinate permutation; W_C=P_h W_F P_y^T and y_C=P_y y_F',
            'real_expansion':'[Re(hhat);Im(hhat)]=[[Re(W),-Im(W)],[Im(W),Re(W)]][Re(yp);Im(yp)]',
            'network_implemented':False},
        'RA_audit_only':{'W_fixed':True,'real_factors':'U_s,V_s in R^(L x r_s)',
            'filter':'W_RA=W product_s(U_s V_s^T), rank <= r_final',
            'SVD_export':'W_RA=P diag(s) Q^H; A=P_r diag(s_r), B=(Q^H)_r^T, so AB^T uses ordinary transpose',
            'factor_shapes_project':'A4096xr, B64xr; two products A(B^T yp)',
            'training':'same noisy-pilot/clean-full-CSI MSE, same split; only U/V optimized',
            'rank_policy':'hardware capability, not instantaneous SNR/Doppler',
            'complexity':'r(4096+64) complex MACs vs4096*64; compresses entire W, unlike two rank4 corrections plus base',
            'implemented':False}})
    versions = {}
    for name in ['numpy','scipy','pytest','torch','sionna-no-rt','scikit-learn','tensorflow','tf_keras']:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'NOT_INSTALLED'
    gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    resource = {'native':native,'project':project,'cap_GiB':8,'STOP_D':True,'decision_basis':'source-derived unavoidable parameter storage; no OOM attempt',
          'batch1':'NOT_RUN: weight lower bound exceeds cap','batch50':'NOT_RUN: same batch-independent bound',
          'author_toy_network':'NOT_RUN: experimental branch STOP-D', 'one_optimizer_step':'NOT_RUN',
          'GPU_training_peak_bytes':None,'activation_peak_bytes':None,'training_wall_seconds':None,
          'exported_project_W_complex64_payload_bytes':4096*64*8,'GPU_snapshot':gpu,
          'process':win_memory(),'environment':{'python':sys.version,'platform':platform.platform(),'packages':versions},
          'license':{'github_license':pin['license'],'license_files':[p for p in pin['all_repository_paths'] if 'license' in p.lower() or 'copying' in p.lower()],
               'status':'LICENSE_OPEN_RISK','STOP_C':False,'reason':'absent permission is open; inability of clean-room port not established; no author code executed'},
          'audit_wall_seconds':time.perf_counter()-start}
    save('resource_gate',resource)
    clean = json.loads(audit_read(ROOT / 'outputs/sp_u4a_clean_gate.json'))
    assert clean['trajectories'] == 1500 and clean['frames'] == 15000
    permitted_hashes = {name:sha(audit_read(ROOT/name)) for name in sorted(ALLOWED) if (ROOT/name).exists()}
    save('input_provenance', {'read_allowlist':sorted(ALLOWED),'actual_audit_reads':sorted(READS),'permitted_input_sha256':permitted_hashes,
          'clean_corpus_sha256_from_provenance':clean['corpus_sha256'],'clean_channels_loaded':False,
          'T20B_U2_U4_U5_performance_outcomes_read':False,'no_new_B120_data':True,'no_history_performance_selection':True})
    save('decision', {'task':'SP-U6A','engineering_verdict':'FAIL / STOP-D','scientific_verdict':'interface-compatible candidate; faithful experimental reproduction NOT_ESTABLISHED',
         'source_pin':'PASS','mode':'unique repository-global-token target with disclosed paper recipe gap',
         'MIMO':'CASE2 interface candidate; shape smoke NOT_RUN','pilot_geometry':'PASS selector geometry algebra only',
         'fairness_design':'PASS with disclosed differences','algebra_calibration':'PASS','toy_author_network':'NOT_RUN',
         'project_forward_backward':'NOT_RUN','resource_gate':'FAIL_STATIC_LOWER_BOUND','license':'LICENSE_OPEN_RISK','firewall':'PASS',
         'next_phase_authorized':False,'stop_after_delivery':True})
    print(json.dumps({'verdict':'FAIL / STOP-D','commit':pin['commit'],'decoder_project_GiB':project['float32_decoder_GiB'],
        'five_square_lower_bound_GiB':project['unconditional_five_square_kernel_bytes']/2**30,
        'native_decoder_GiB':native['float32_decoder_GiB'],'algebra_only':True,'training_executed':False,
        'GPU_snapshot':gpu,'actual_worker_pid':os.getpid()},indent=2))

if __name__ == '__main__':
    main()
