# Live GTM viewer on a backend machine

The US map, comparison and experiment list use the same saved result catalog. The frontend remains a small local HTML/JavaScript app using the bundled Leaflet renderer; no Node build or separate map service is required.

## Moving to the research machine

1. Check out the research branch and copy any complete experiment exports you want to keep. The repository already includes a small portable sample; full local runs are ignored by Git and must be transferred separately.
2. Start the viewer on that machine with the command under **Viewer API and access** below. Python 3.11 or newer is sufficient.
3. Connect through an SSH forward or the machine's existing private Tailscale route. Open `http://127.0.0.1:8766/#national` when using the SSH forward. The frontend and image URLs use the same server, so workstation paths do not need to be edited.
4. Have the experiment's export step prepare the manifest and assets, then invoke the publisher after each complete scene or batch. Include every scene that should remain in that experiment. The viewer checks every five seconds by default while its tab is visible. It does not start training or automatically modify the training scripts.

The connection bar shows the backend name and last successful check. Optional run progress is shown only when the producer reports it. New scenes are announced without taking over the current selection. The current image, zoom, threshold and layer controls remain available if the backend disconnects; the viewer retries with backoff. **Refresh now** retries immediately and reloads the selected scene.

Map filters, camera, satellite choice and selected observation are encoded in the URL. **Open experiment viewer** opens the exact saved scene in a new tab. Pixel-only scenes remain image comparisons unless the export supplies a location; the viewer does not invent a geographic footprint.

## Failure and acceptance scenarios

The repeatable HTTP end-to-end check in `tools/GTM_Model0_viewer_e2e.py` covers these observable cases:

1. A fresh checkout with an empty writable bundle folder serves the built-in registry and portable saved scenes. The API returns a live revision and never invents model scores.
2. Publishing a first scene subset makes it visible. Publishing a larger revision of the same experiment ID changes the revision and scene count without modifying the prior version's assets.
3. A malformed replacement manifest or a manifest referencing a missing file produces a warning while the last complete experiment remains visible. A malformed new bundle is skipped without hiding other experiments.
4. A same-name asset update changes the revision using its file size and modification time. The HTTP response uses `Cache-Control: no-store` so a new image or grid is fetched.
5. Traversal and symlink escapes cannot be served as assets. The server listens on loopback by default; `--host` is explicit for a different bind address.
6. Published run state passes through only when its state and progress fields have the documented shape. The viewer does not derive epochs or runtime from files.

The E2E command writes a JSON record with the named outcome of each check and the commands under test. It runs a temporary child server and stops it afterward.

```sh
python tools/GTM_Model0_viewer_e2e.py --artifact output/viewer-http-e2e.json
```

Browser verification and screenshots are recorded in [the October 2 review](../reports/research/GTM_Model0_viewer_polish_review_20261002.md). That check replays existing saved outputs and does not represent a new model experiment.

## Producer publication contract

Prepare a directory containing `experiment.json` and all referenced scene assets. Every referenced path must be relative to that directory, must exist, and must remain inside it. Publish with:

```sh
python tools/GTM_Model0_publish_viewer.py --bundle-dir /path/to/prepared --bundles-dir outputs/GTM_viewer_bundles
```

The publisher validates the prepared manifest, copies it into a unique immutable `_versions/<version>` directory beneath the destination experiment folder, rewrites its asset references to that version, and replaces `<experiment-folder>/experiment.json` as the final operation using `os.replace`. Never edit a published version. To add scenes, prepare and publish another complete manifest with the same experiment ID and all scenes that should remain visible. Prior version folders may be retained for open browser tabs. A producer can repeat this after each completed scene or batch. The CLI rejects unsafe paths and files that change while being copied. The manifest's `id` must match its destination folder, which is derived from that ID.

Optional run state in the prepared manifest is:

```json
{"run":{"state":"running","progress":{"completed":2,"total":4,"unit":"scenes"},"startedAt":"2026-10-02T12:00:00Z","updatedAt":"2026-10-02T12:05:00Z","message":"Two scene exports are ready"}}
```

States are `queued`, `running`, `completed`, and `failed`. Progress values are nonnegative integers with `completed <= total`; timestamps are ISO 8601 strings. The server validates and passes these values through. It does not infer them from artifact timestamps.

## Viewer API and access

```sh
python tools/GTM_Model0_local_viewer.py --host 127.0.0.1 --port 8766 --bundles-dir outputs/GTM_viewer_bundles --poll-seconds 5 --name "Lab GPU"
```

`GET /api/experiments` keeps the existing registry fields and adds top-level `live: {revision, checkedAt, pollSeconds, serverName}` and each published experiment's `live: {revision, updatedAt}`. The revision changes when manifest bytes or any referenced file size or modification time changes; it also changes when the warning list changes. Poll the endpoint and replace the scene data when its revision changes. Send `If-None-Match` with the quoted revision from the previous `ETag`; unchanged catalogs return HTTP 304. Image and grid URLs remain same-origin. Responses use `Cache-Control: no-store`; a client may append the experiment revision as a query parameter to force a refresh in intermediate caches. `local.warnings` explains rejected or temporarily incomplete bundles. The existing `/api/health` app name remains `GTM local experiment viewer`.

The bind address defaults to `127.0.0.1`. For a remote backend, keep that default and use an SSH local port forward (`ssh -L 8766:127.0.0.1:8766 user@backend`) or an existing private Tailscale Serve route. The viewer has no login or cross-origin API, so do not expose the bind port directly to an untrusted network.
