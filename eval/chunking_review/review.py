"""Offline boundary experiment; does not modify the production index or chunker."""
import json
import re
from pathlib import Path

from clinical_trial_rag_copilot.retrieval.chunking import create_trial_chunks

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'eval/pipeline_runs/20261116T070337Z/datasets/oncology_trials.json'
IDS = ['NCT03940703', 'NCT05894889', 'NCT06222489', 'NCT06926413',
       'NCT07710612', 'NCT07912723', 'NCT07360743', 'NCT01505400',
       'NCT06667076', 'NCT05379803']


def split_preview(text):
    # Offsets retain every source character. No sentence splitting or rewriting.
    markers = list(re.finditer(r'(?m)^(?:\* |\d+\. )', text))
    if re.search(r'(?m)^\* \d+\\\.', text):
        return [text], 'Keep whole: mixed bullet/escaped numbering and inline criteria.'
    if not markers:
        return [text], 'Keep whole: no unambiguous top-level list boundaries.'
    if markers[0].start() != 0:
        return [text], 'Keep whole: introductory scope must remain attached to the list.'
    styles = {m.group().startswith('*') for m in markers}
    if len(styles) != 1:
        return [text], 'Keep whole: mixed top-level list styles.'
    if False in styles:
        nums = [int(m.group().split('.')[0]) for m in markers]
        if nums != list(range(1, len(nums) + 1)):
            return [text], 'Keep whole: numbering is not consecutive.'
    starts = [m.start() for m in markers] + [len(text)]
    chunks = [text[a:b] for a, b in zip(starts, starts[1:])]
    if any(c.rstrip().endswith(':') for c in chunks[:-1]):
        return [text], 'Keep whole: a list item introduces following items without reliable nesting.'
    if len(chunks) == 1:
        return chunks, 'Keep whole: only one item; preserve even very short source text.'
    return chunks, 'Split at top-level items; retain continuation text and indented sublists.'


def main():
    trials = {t['nct_id']: t for t in json.loads(SOURCE.read_text())}
    records = []
    report = ['# Exclusion chunk boundary review: 10 local trial records',
              'Source: `eval/pipeline_runs/20261116T070337Z/datasets/oncology_trials.json`. '
              'These are stored registry excerpts, not a live registry refresh. '
              'Purposive formatting sample, not random or representative. '
              'This tests boundary preservation, not retrieval accuracy. '
              'Each clear top-level item is shown separately to expose boundaries; '
              'a production version could group short items. No token-size policy is tested. '
              'Trial headers/metadata would accompany each child chunk; omitted below to avoid repetition.']
    for index, nct in enumerate(IDS, 1):
        trial = trials[nct]
        original = trial['exclusion_criteria']
        chunks, reason = split_preview(original)
        current = [c for c in create_trial_chunks(trial) if c.chunk_type == 'TRIAL_EXCLUSION']
        assert ''.join(chunks) == original, nct
        records.append(dict(nct_id=nct, title=trial['brief_title'], original=original,
                            current_chunk_count=len(current), proposed_chunks=chunks,
                            reason=reason, exact_text_preserved=True))
        report += [f'## {index}. {nct}', trial['brief_title'],
                   f'[Registry record](https://clinicaltrials.gov/study/{nct})',
                   f'Current: {len(current)} exclusion chunk(s). Preview: {len(chunks)}. {reason}',
                   '### Original exclusion text', '```text\n' + original + '\n```',
                   '### Proposed boundary preview']
        if len(chunks) == 1:
            report.append('Unchanged: retain the full original text above as one chunk.')
        else:
            for number, chunk in enumerate(chunks, 1):
                report += [f'Chunk {number}:', '```text\n' + chunk.rstrip() + '\n```']
    out = Path(__file__).parent
    (out / 'examples.json').write_text(json.dumps(records, indent=2, ensure_ascii=False))
    (out / 'review.md').write_text('\n\n'.join(report) + '\n')
    for row in records:
        print(row['nct_id'], row['current_chunk_count'], '->', len(row['proposed_chunks']), row['reason'])
    print('PASS: exact source text preserved in all 10 examples (checked before display formatting).')


if __name__ == '__main__':
    main()
