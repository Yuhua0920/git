"""Remove redundant editorial caveats and repair a new prose-line overflow."""
from pathlib import Path
path=Path(__file__).resolve().parent/'sp_u9b1_content.py'
text=path.read_text(encoding='utf-8')
pairs=[
 ('Their titles and verified abstract establish relevant intellectual debt; they do not justify assigning them an unverified fitting objective or online information contract. ',''),
 ('GMM estimators learn channel distributions offline and combine component-wise conditional LMMSE estimates using posterior responsibilities evaluated from the received observations','GMM estimators learn statistical channel mixtures offline and combine conditional LMMSE estimates using observation-dependent posterior responsibilities'),
 ('No numerical comparison with the adaptive sparse-filter implementations is claimed.',''),
 ("Channelformer is an existing attention-based neural channel-estimation work in the manuscript's literature set",'Channelformer is an attention-based neural channel estimator'),
 ('We retain it as an example of a learned estimator, without importing its detailed online-training interface into the present comparison. ',''),
 ('That limitation concerns the examined reproduction interface and available experimental evidence, not the attainable accuracy of all implementations of those estimators.','')]
for before,after in pairs:
 if text.count(before)!=1:raise ValueError('Unexpected prose state: '+before)
 text=text.replace(before,after)
path.write_text(text,encoding='utf-8')
print('Applied six targeted prose edits')
