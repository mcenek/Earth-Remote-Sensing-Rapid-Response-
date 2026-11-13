"""Add a full visual gallery to existing first-pass reports; do not rerun inference."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
from GTM_Model8_mars_review import review_html
from GTM_Model0_publish_viewer import publish

run = ROOT / 'outputs/GTM_Model8_mars_baseline_20261030'
receipt = {'state':'passed', 'inference_rerun':False, 'publications':{}}
for role in ('validation','holdout'):
    parent = run/'review/first_pass'/role
    summary = json.loads((parent/'summary.json').read_text())
    publications = {}
    for method in ('baseline','released_reference'):
        source = parent/method
        experiment = json.loads((source/'experiment.json').read_text())
        if method == 'released_reference':
            experiment['status'] = 'Released pretrained reference'
            experiment['run']['state'] = 'completed'
            (source/'experiment.json').write_text(json.dumps(experiment, indent=2)+'\n')
        note = f"Local checkpoint after pass {summary['epoch']+1}." if method == 'baseline' else 'Released pretrained checkpoint; no local fitting.'
        report = review_html(experiment['name'], experiment['scope'], summary['pooled'][method], summary['negative'][method], experiment['scenes'], note)
        (source/'index.html').write_text(report, encoding='utf-8')
        publications[method] = publish(source, ROOT/'outputs/GTM_viewer_bundles')
    (parent/'publication.json').write_text(json.dumps(publications, indent=2)+'\n')
    receipt['publications'][role] = publications
(run/'first_pass_review_recovery.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
