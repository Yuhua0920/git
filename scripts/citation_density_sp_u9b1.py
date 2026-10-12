"""Keep grouped method assertions adjacent to their primary-source citations."""
from pathlib import Path
p=Path(__file__).resolve().parent/'sp_u9b1_content.py'
t=p.read_text(encoding='utf-8')
pairs=[
 ('Thus affine combination, unrestricted coordinates and windowed least squares are established components.',r'Thus affine combination, unrestricted coordinates and windowed least squares are established components~\cite{azpicueta2010ls}.'),
 ('These works already establish offline statistical learning, observation-dependent mixing, conditional LMMSE estimators and compact statistical representations.',r'These works already establish offline statistical learning, observation-dependent mixing, conditional LMMSE estimators and compact statistical representations~\cite{koller2022gmm,fesl2022structured,fesl2023mfa}.')]
for before,after in pairs:
 if t.count(before)!=1:raise ValueError('Unexpected prose state')
 t=t.replace(before,after)
p.write_text(t,encoding='utf-8')
print('Bound two synthesis assertions to direct citations')
