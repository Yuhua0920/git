"""Build advisor-facing material from the saved manuscript and frozen evidence."""
import re
import subprocess
from pathlib import Path
from paper_support import ROOT, json_read, json_write, csv_read, sha256
from build_paper_tables import CONDITIONS, CONDITION_NAMES, CORE_METHODS, NAMES, number
from audit_paper_style import printed_tex
from audit_manuscript_draft import sentence_records
from paper_history import historical_bytes, prior_paper_commit


def expand(text, macros):
    for name,record in macros.items():text=text.replace('\\'+name+'{}',record['display_value'])
    return text


def plain(text):
    text=printed_tex(text)
    text=re.sub(r'\\begin\{(align|equation)\}.*?\\end\{\1\}','',text,flags=re.S)
    text=re.sub(r'\\(?:section|subsection|caption|textbf)\{([^{}]*)\}',r'\1',text)
    text=re.sub(r'\\(?:begin|end)\{[^}]*\}(?:\[[^]]*\])?|\\centering','',text)
    return re.sub(r'\s+',' ',text).strip()


def headings(text):return re.findall(r'\\subsection\{([^}]+)\}',text)


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def build(root=ROOT):
    root=Path(root);table=json_read(root/'paper/tables/t20f_provenance.json');macros=table['number_macros']
    setup=(root/'paper/sections/05_experimental_setup.tex').read_text(encoding='utf8')
    results=(root/'paper/sections/06_results.tex').read_text(encoding='utf8')
    a=json_read(root/'outputs/t20a_results.json');protocol=a['formal_protocol']
    formal=json_read(root/'outputs/t20b_formal_results.json')
    conditions=[]
    for name in CONDITIONS:
        _,profile,speed,delay=name.split('-')
        meaning={'ID':'同源分布，独立 realization','Profile':'已用于离线准备的 profile 条件，独立且不重合 realization',
            'Mobility':'已用于离线准备的 mobility 条件，独立且不重合 realization','Composed':'留出的 profile/mobility 组合',
            'Boundary':'留出的 nominal delay-spread 条件'}[CONDITION_NAMES[name]]
        conditions.append([CONDITION_NAMES[name],'CDL-'+profile,speed,delay,meaning])
    baseline_rows=[
        ['Perfect CSI','真实 H；evaluation-only reference'],
        ['LS-LIN','Pilot-LS 的频率/时间线性插值；边界常值延伸'],
        ['Source','冻结源统计量、源均值和完整 LMMSE operator'],
        ['Hard','同一四折五帧 pilot loss，在 Source/P/M 三个顶点中选择；exact tie 顺序 Source/P/M'],
        ['Convex','同一 pilot loss，alpha>=0、beta>=0、alpha+beta<=1；比较顶点、边及可行内部解'],
        ['Dense Affine','同一 unrestricted real alpha/beta，完整 corrections，source means'],
        ['LR4 Affine','与 Dense 完全相同 alpha/beta；每个 correction 用原 rank-4 factors；完整 Source 与 cross-pilot operators 保持'],
        ['Matched','独立准备的目标统计量/均值；evaluation-only，非可部署方法']]
    packet=['# T20F advisor review packet','', '## A. Experimental Setup subsection structure','']
    packet+=['- '+h for h in headings(setup)]
    packet+=['','## B. Results subsection structure','']+['- '+h for h in headings(results)]
    packet+=['','## C. Evaluation conditions','',markdown_table(['Condition','Profile','Speed (km/h)','Nominal DS (ns)','Scientific role'],conditions),'',
        f"每个条件 {formal['trajectories_per_condition']} 条独立 trajectory，每条 {formal['frames_per_trajectory']} 连续 frames，score {min(protocol['scored_frames'])}–{max(protocol['scored_frames'])}，SNR {formal['snr']} dB。所有 evaluation realizations 与 offline preparation 不重合。",'',
        '## D. Baseline definitions','',markdown_table(['Estimator','Definition / information'],baseline_rows),'',
        '## E. Metrics and statistical protocol','',
        'NMSE = 总 data-domain error energy / 总 truth energy，再转 dB；不平均 trajectory NMSE_dB。',
        'BER = 总 data bit errors / 总 data bits，Gray QPSK、冻结 ZF，仅作 end-to-end detector sensitivity。',
        f"独立单位为 trajectory；同 trajectory 的相关 frames 一并 resample。Paired bootstrap {protocol['bootstrap_resamples']} 次；descriptive 95% intervals。",
        f"Source comparisons: Profile/Mobility/Composed × 两 SNR；Convex 和 compression comparisons: Composed × 两 SNR。共 {protocol['primary_cell_comparisons']} 项，one-sided Bonferroni bounds，原 quantiles 0.005/0.995，原 compression margin {protocol['compression_margin_db']} dB。",
        '噪声沿用冻结 source-ensemble reference，不对 target 或逐帧 H 归一化。Matched 的目标均值/统计量仅作评价参考。','',
        '## F. Core finding per Results subsection','']
    sections=re.split(r'\\subsection\{([^}]+)\}',results)[1:]
    for title,body in zip(sections[::2],sections[1::2]):
        entries,_=sentence_records(expand(body,macros),'results')
        first=next(e['sentence'] for e in entries if e['category']=='CLAIM')
        packet+=['### '+title,'',first,'']
    used=[name for name in macros if '\\'+name+'{}' in setup+results]
    packet+=['## G. All key numbers used in the prose','',
        '数字全部通过冻结来源生成的宏引用；下表为本稿实际使用的宏。System/condition 数字另见本稿自动生成的 Table I 与 C 部分。','',
        markdown_table(['Macro','Printed value','Frozen source / selector','Operation'],[
            [name,macros[name]['display_value'],macros[name]['source_artifact']+' '+str(macros[name]['selector']),macros[name]['operation']] for name in used]),'',
        '### System parameter table','', '```latex', (root/'paper/tables/01_system.tex').read_text(encoding='utf8').strip(), '```','',
        '## H. Three figure captions','']
    captions=re.findall(r'\\caption\{(.*?)\}\\label\{fig:(nmse|endpoints|ber)\}',results,re.S)
    for caption,label in captions:packet += ['### '+label,'',expand(caption,macros),'']
    packet+=['## I. Compact tables','',
        'Source/Convex/Dense/LR4/Matched × 全部五条件 × 两 SNR。Perfect/LS-LIN/Hard 仍在全量图和附录；ID 损失和 Boundary 压缩损失均保留。','']
    for metric,file in (('nmse_db','nmse'),('ber','ber')):
        rows=csv_read(root/f'outputs/t20b_formal_{file}.csv')
        lookup={(r['environment'],r['snr_db'],r['method']):r for r in rows}
        packet += ['### '+('NMSE (dB)' if metric=='nmse_db' else 'BER'),'',
            markdown_table(['Condition','SNR']+[NAMES[m] for m in CORE_METHODS],[
                [CONDITION_NAMES[n],str(s)]+[number(lookup[n,str(s),m][metric],metric) for m in CORE_METHODS]
                for n in CONDITIONS for s in formal['snr']]),'']
    packet+=['## J. Potentially strong wording','',
        '- “substantial part”: 由限定的 single-shift gains 支持，为描述性用语，没有新增阈值。',
        '- “remained close”: 仅指已准备的两个单项条件；完整 Matched gap 在正文表中。',
        '- “consistent gain”: 仅对应 Composed 两 SNR 的 paired gains 与正 adjusted lower bounds。',
        '- “within the predefined margin”: 仅用于 Composed；不推广至 Boundary。',
        '- representation/fidelity boundary 是测试条件下的限制，不将 Matched gap 全部归因于某个独立物理机制。','',
        '## K. Boundary wording','']
    boundary=sections[-1]
    entries,_=sentence_records(expand(boundary,macros),'boundary')
    packet+=[' '.join(e['sentence'] for e in entries),'',
        '## L. Advisor decisions','',
        '- 是否保留 “remained close” 或改为只指向表中数值；当前不增加新的判据。',
        '- 紧凑表保留全部五条件；是否接受精确值在表、区间在图/附录的分工。',
        '- 最终 venue 模板确定后再核验 Setup/Results 页数与通栏表图布局；本轮不切换模板。']
    (root/'docs/t20f_advisor_review_packet.md').write_text('\n'.join(packet)+'\n',encoding='utf8',newline='\n')
    word_counts={}
    for path,text in zip(('05_experimental_setup.tex','06_results.tex'),(setup,results)):
        old=historical_bytes(prior_paper_commit(),'paper/sections/'+path).decode()
        current=expand(text,macros)
        word_counts[path]=dict(scaffold_words=len(plain(old).split()),draft_words_including_captions=len(plain(current).split()),
                              subsection_words={title:len(plain(expand(body,macros)).split()) for title,body in zip(re.split(r'\\subsection\{([^}]+)\}',text)[1::2],re.split(r'\\subsection\{([^}]+)\}',text)[2::2])})
    notes=['# T20F writing notes','',
        'Scope: Experimental Setup and Results only. Method/System Model and other prose sections remain byte-frozen. All source scientific outputs remain immutable.','',
        '## Evidence allocation','',markdown_table(['Result function','Placement / rationale'],[
            ['core discovery','Profile/Mobility efficacy and Composed affine-over-Convex comparisons in main text'],
            ['necessary support','Paired bounds; shared Dense/LR4 coordinates; trajectory sampling/aggregation in Setup'],
            ['qualification','ID NMSE loss and all Boundary gaps/compression penalty visible in main text'],
            ['detector consequence','Descriptive BER main figure/table without a success threshold'],
            ['complete record','Eight methods × five conditions × two SNR, full values/CIs in appendix'],
            ['provenance detail','Hashes, execution and historical-version checks only in repository audit files']]),'',
        '## Compression and repetition','',
        'Main tables contain 50 estimates each instead of 80 rows with two CI limits. Full values and intervals are preserved. Prose selects gains, paired bounds, ID losses and delay limits; captions explain reading conventions rather than duplicate gains. No historical development sequence or mechanism analysis is added.','',
        '## Word-count record','',str(word_counts),'',
        'Page targets remain approximately 1.0–1.5 (Setup) and 2.5–3.5 (Results), subject to final venue layout. No pagination claim is made without compilation.','',
        '## Historical provenance','',
        'Old T20C/T20D/T20E table/figure products are checked against their committed snapshots. When a stored working-copy hash differs from the Git blob because of checkout line endings, native git cat-file --filters restores the historical representation and must exactly match the original recorded SHA256; arbitrary normalization is not accepted. Current derivatives have separate T20F provenance and frozen-field/numeric-geometry regressions. Historical JSON hashes are not rewritten. Read-only math/static checks no longer write historical reports during pytest.','',
        '## Figure contract','',
        'Quantitative grids. NMSE panels cover source-matched control, prepared individual shifts, held-out composition and delay boundary. BER reports the paired detection consequences. Difference panels isolate source benefit, affine-over-Convex benefit and compression fidelity. Eight methods remain, with Perfect NMSE explicitly N/A because linear error is zero. All exports use Python and original data geometry.','',
        'Markers/line styles provide redundant identity in grayscale/CVD views. Near-equal curves overlap legitimately, especially under ID/single shifts; exact small differences are retained in the appendix. Actual 174-mm minimum glyph sizes are audited, not inferred from nominal font settings.','',
        '## Bibliography','',
        'One verified 3GPP report entry is added for the implemented V19.2.0 CDL/delay model. Its official report number and permanent ETSI PDF are used under the approved persistent-identifier rule; no DOI/arXiv was verified in checked sources. The prior 14 entries remain unchanged. Sionna version is taken from frozen project provenance, not from a mutable documentation landing page.','',
        '## Known engineering issues','',
        'Native compilation fails before parsing with “Unable to find standard directories for platform”; static validation does not imply a PDF compile. Existing Python launcher and Git line-ending notices remain visible in logs. Early new-formatter syntax and ambiguous bootstrap-namespace failures were repaired before integration; their logs are preserved. A later paper-test run had 90 passes and two historical editorial provenance failures caused by CRLF working bytes versus LF Git blobs; native Git checkout-filter recovery preserves the exact historical recorded hash. The first final packaging attempt found trailing whitespace in Matplotlib SVG exports after 2075 full tests passed; export cleanup now requires exact drawing-token and meaningful-text equality before writing. No scientific algorithm, metric or historical scientific regression was relaxed.','',
        '## Decisions pending advisor review','',
        'Qualitative “remained close” wording, compact-table packaging and final venue pagination. These do not block the evidence-linked first draft. No later section is drafted.']
    (root/'docs/t20f_writing_notes.md').write_text('\n'.join(notes)+'\n',encoding='utf8',newline='\n')
    out=dict(word_counts=word_counts,used_number_macros=used,figure_caption_count=len(captions),
        advisor_packet_sections=list('ABCDEFGHIJKL'),page_count_verified=False,
        manuscript_quality_first_draft=True,discussion_or_other_later_section_written=False,
        generated_sha256={p:sha256(root/p) for p in ('docs/t20f_advisor_review_packet.md','docs/t20f_writing_notes.md')})
    json_write(root/'outputs/t20f_writing_audit.json',out);return out


if __name__=='__main__':
    r=build();print({'word_counts':r['word_counts'],'used_number_macros':len(r['used_number_macros']),'figure_captions':r['figure_caption_count']})
