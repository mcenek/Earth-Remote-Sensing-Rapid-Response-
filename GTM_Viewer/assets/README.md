# Local viewer assets

The main viewer does not load a mapping library or US boundary file from a CDN at runtime.

- `us-states-10m.json`: the `states-10m.json` geographic TopoJSON from [topojson/us-atlas](https://github.com/topojson/us-atlas), derived from US Census boundaries. Includes state and nation geometry for the local overview and satellite outside-US mask. The filename's `10m` is the cartographic scale, not ten-meter imagery. See `us-atlas.LICENSE`.
- `topojson-client.min.js`: [topojson-client 3.1.0](https://github.com/topojson/topojson-client/tree/v3.1.0), vendored with `topojson-client.LICENSE`.

These files provide geographic context, not source detections or evaluation labels. Scene footprints come from each saved export. Optional satellite tiles come from Esri World Imagery and carry attribution in the map; their acquisition date is not the selected research scene's acquisition date.
