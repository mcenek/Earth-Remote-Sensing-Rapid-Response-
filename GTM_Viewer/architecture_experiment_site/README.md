# Architecture experiment viewer source archive

This directory preserves the separate architecture-experiment-chart-site source at commit `df77a01` (2026-10-30). It does not replace the main GTM_Viewer runtime or deploy a website. Earlier viewer implementations, experiment chart, comparison controls, research progress pages, export tools and validation scripts are retained here.

The original site's image/grid bundles, prediction inventories, generated catalogs, hosting configuration and Git history are deliberately omitted. The HTML and JavaScript still describe their original relative data contracts; a source-only checkout cannot show those omitted experiments. Do not interpret missing assets as empty predictions or rerun inference to fill them. The retained small GTM_Model0_data.js registry was already present in the main research repository; its historical scene metadata is not a new dataset release.

Run the repository's ordinary viewer from the repository root for its existing portable examples. Snapshot export tools assume the original separate-site directory layout and saved local exports. They are archived source, not an instruction to acquire or publish data. Historical review documents describe their original inspection date; those reviews are not new visual acceptance of this archive.

Recent Task0016 fine-tuning, datasets, checkpoint files, private correspondence, transfer receipts and machine configuration are excluded. No training, inference or acquisition was run for this publication.

Verify the archived import/path/no-data contract without omitted bundles: from this directory run `node tools/GTM_Model0_verify.cjs --source-only`. The verifier now supplies the browser URL primitives in its Node sandbox. Full bundle checks require the excluded assets.
