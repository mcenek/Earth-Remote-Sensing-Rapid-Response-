# Local viewer setup for Sean

The viewer shows saved satellite images, methane reference data, and model predictions. It does not run models.
Python 3.11 or later is necessary. A GPU is not necessary.

## Get the viewer

Use Git to get the current research branch:

```text
git clone --depth 1 --single-branch --branch research/gtm-model6-restart-2026-09-22 https://github.com/mcenek/Earth-Remote-Sensing-Rapid-Response-.git methane-viewer
cd methane-viewer
```

If you already have the repository, stop the viewer. Enter these commands from the repository folder:

```text
git switch research/gtm-model6-restart-2026-09-22
git pull --ff-only
```

Start the viewer again. Reload the browser page.

## Start the viewer

On a MacBook or Linux desktop, enter this command:

```text
python3 tools/GTM_Model0_local_viewer.py --open
```

On a Windows desktop, enter this command:

```text
py -3 tools/GTM_Model0_local_viewer.py --open
```

If Windows cannot find `py`, use `python` instead.
The Windows file `GTM_Model0_OpenViewer.cmd` is an alternative launcher.

1. Keep the terminal open while you use the viewer.
2. Open <http://127.0.0.1:8766/> if the browser does not open.
3. Press Ctrl+C in the terminal to stop the command-line server.

The viewer uses only Python's standard library. The project training packages are not necessary.
The saved results stay on your computer. Satellite background tiles need an internet connection.
The normal map and saved result images work without an internet connection.

## Open the saved comparison

1. Open <http://127.0.0.1:8766/>. The viewer opens the saved plume comparison.
2. Select **Start presentation** if an old bookmark opens a different experiment.
3. Compare the original image, reference outline, and prediction.
4. Use **Model** to select **Sentinel predictions vs EMIT** for the three EMIT reference examples.
5. Use **Scene** to select another saved image. The six plume-mapper cases include partial matches, a missed plume, and false detections.
6. Select **Disagreement** to inspect pixel errors. Areas outside an EMIT positive outline remain unknown.
7. Select **Map** to see the selected scene at its location. Use **Fit scene** after you zoom out.

The cyan outline shows the reference. Orange shows the prediction.
Gray stripes identify pixels without model output. These pixels are not confirmed methane-free areas.

The branch includes six Model6 R5 scenes, three separate EMIT support examples, and one original-team archive image.
It also includes the Volusia input review and the earlier meeting review.
The default **Presentation examples** collection shows nine saved prediction scenes. No model run or data download is necessary.
Select **Full research archive** to inspect other local runs, old folds, or reports. These are hidden from the default presentation.
Some archive entries contain only reports. They do not contain viewable predictions.

## Explain the results

The selected R5 scene has 38.8% native intersection over union (IoU).
The complete R5 development run has 2.15% IoU. The run failed its quality check.
This scene shows a useful local result. It does not prove that the model works across the United States.

The new three-CNN design is a proposal. This package does not contain results from that design.
The default collection contains selected development examples. It is not a new validation set.
The EMIT examples show predictions compared with known positive plume support. They do not measure false-positive rate outside that support.

## If the viewer does not start

1. Check the Python version with `python3 --version` on macOS or `py -3 --version` on Windows.
2. If necessary, install Python 3.11 or later from <https://www.python.org/downloads/>.
3. Check that the terminal is in the folder that contains `tools` and `GTM_Viewer`.
4. If port 8766 is in use, add `--port 8767` to the start command.
5. For port 8767, open <http://127.0.0.1:8767/>.

The saved-example link uses port 8766. Change that port in the link if necessary.

## Verification

The portable viewer files were checked on Windows on October 2, 2026.
All 68 HTTP checks passed. All nine presentation scenes served their image, reference, prediction and numeric grid.
The browser opened the prediction without a local training cache. Map, Satellite, model selection and layer controls were checked.
No private training folder was present. No model packages were installed for the viewer.
The macOS command has not been tested on a MacBook.

Viewer source: `research/gtm-model6-restart-2026-09-22`. Use `git pull --ff-only` to get the prediction-first update.
