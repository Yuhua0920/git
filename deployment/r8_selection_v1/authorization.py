"""Independent R8 human attestation; never upgrades or imports R6/R7 approval."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json

ACTION='FROZEN_STRUCTURE_SELECTION_ONCE'
GRANTS={'selection_authorized':True,'candidate_retraining_authorized':False,
        'validation_rerun_authorized':False,'final_training_authorized':False,
        'target_access_authorized':False,'B2_B3_unseal_authorized':False}
FIELDS={'format','status','scope','allowed_actions','grants','bindings',
        'approval_id','approved_by','approved_UTC','mentor_instruction_reference',
        'mentor_instruction_SHA256','no_automatic_retry','maximum_executions'}
def check(v,expected,instruction,synthetic=False):
    def need(ok,msg):
        if not ok:raise RuntimeError(msg)
    need(set(v)==FIELDS,'R8_AUTHORIZATION_SCHEMA_INVALID')
    need(v['format']=='R8_SELECTION_ONLY_HUMAN_ATTESTATION_V1','R8_AUTHORIZATION_FORMAT')
    need(v['status']==('SYNTHETIC_TEST_APPROVED'if synthetic else'MENTOR_APPROVED'), 'R8_SELECTION_NOT_AUTHORIZED')
    need(v['scope']==ACTION and v['allowed_actions']==[ACTION],'R8_ACTION_SCOPE_REJECTED')
    need(v['grants']==GRANTS and all(type(x)is bool for x in v['grants'].values()),'R8_GRANTS_EXCEED_SELECTION_ONLY')
    need(v['bindings']==expected,'R8_AUTHORIZATION_BINDING_DRIFT')
    need(type(v['maximum_executions'])is int and v['maximum_executions']==1 and v['no_automatic_retry']is True,'R8_ONCE_ONLY_REQUIRED')
    need(all(isinstance(v[k],str)and v[k].strip()for k in ('approval_id','approved_by','approved_UTC','mentor_instruction_reference')),'R8_HUMAN_FIELDS_MISSING')
    need(v['mentor_instruction_SHA256']==hashlib.sha256(instruction).hexdigest(),'R8_HUMAN_INSTRUCTION_HASH_MISMATCH')
    try:stamp=datetime.fromisoformat(v['approved_UTC'].replace('Z','+00:00'))
    except (ValueError,TypeError):raise RuntimeError('R8_HUMAN_TIMESTAMP_INVALID')
    need(stamp.tzinfo is not None and stamp<=datetime.now(timezone.utc),'R8_HUMAN_TIMESTAMP_INVALID')
    # This validates supplied human bytes and bindings, not a cryptographic signer.
    return {'PASS':True,'approval_id':v['approval_id'],'authority':'INDEPENDENT_HUMAN_ATTESTATION_NOT_CRYPTOGRAPHIC_SIGNATURE',
            'authorized_action':ACTION,'final_training_authorized':False,'target_access_authorized':False}
