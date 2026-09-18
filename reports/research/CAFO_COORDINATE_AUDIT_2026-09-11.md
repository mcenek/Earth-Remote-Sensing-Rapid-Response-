# CAFO coordinate audit (2026-09-11)

## Scope and result

This is a read-only audit of `EarthRemoteSensingRapidResponse/Polygon_CSV_Files/` and the local CAFO-related scripts. No imagery was downloaded and no existing data or scripts were changed.

The only CAFO table present is `iowa_cafos_2024_arcgis_api.csv` (14,155 data rows, 55 fields). It is an Iowa DNR/ArcGIS-style export by its filename, field schema, facility hyperlinks, and provenance fields. I found no local file, metadata, or commit message that identifies this table as a Huang dataset. Therefore the advisor instruction to use Huang CAFO GPS/UTM coordinates is not yet satisfied by this artifact; Huang’s source and exact release should be recorded before treating it as the authoritative replacement.

## Machine-readable fields

| Role | Fields and audit finding |
|---|---|
| Stable facility identity | `uniqueID` is populated for all 14,155 rows and is unique; `progid` is also populated and unique. `LOCID` and `stfacid` are populated but each has 70 duplicate excess rows (14,085 unique values). |
| Geographic coordinates | `Latitude`, `Longitude` are populated and parseable for all rows. Ranges are latitude 40.48087–43.93104 and longitude -96.59452–-90.20071. |
| Projected coordinates | `X_Coord`, `Y_Coord` are populated and parseable for all rows. Ranges are X 206,914–731,949 and Y 4,482,118–4,864,635, consistent with meter-based UTM-like coordinates, but the CSV declares no CRS, datum, EPSG code, or UTM zone. |
| Coordinate quality/provenance | `Accuracy` has 16 values (most common: 20 for 10,537 rows; 570 for 1,574; 0 for 1,282). `colMthTxt`, `refPntTxt`, `VerifyTxt`, `collectBy`, `ColDate`, and `locComment` describe collection context. `collectBy` is missing in 1,283 rows. |
| Source trace | Every row has a distinct `Hyperlink` to an Iowa DNR facility summary URL. `State` is IA for 14,131 rows, with 16 `XX`, 6 MN, 1 AR, and 1 MO. |

There are 149 duplicate latitude/longitude groups (159 excess rows when grouped at six decimal places). The largest exact coordinate group has six different facilities at `42.37969,-93.24053`; other groups contain distinct facility names/IDs sharing a point. These are therefore not safe grounds for dropping rows by coordinate. They may represent generalized/interpolated facility locations, multiple records at one site, or source data collisions. Exact full-row duplicate counts were not used as a deduplication rule; `uniqueID` remains the preferred row key.

The coordinate bounds are broader than Iowa alone (including the `XX`/MN/AR/MO records). A future Huang import should define the intended geographic scope explicitly and retain the source row ID, original lat/lon, original projected coordinates, CRS/datum, and any uncertainty/accuracy field. Facility location is a point-source/site locator; it is not a methane concentration label and must not be used as one.

## Provenance and author check

The CSV name contains `2024_arcgis_api`; its schema includes Iowa DNR program fields (`stfacid`, `progid`, `NPDESPermi`, `Hyperlink`) and row-level collection fields such as `collectBy` and `ColDate`. Git history has one tracked import/related commit (`52096ea1`, 2025-12-05, Gonon, “This is my attempt to determine which coordinates in a CAFO csv file are inside a polygon area.”). Neither the file nor that history attributes the data to Huang. `facName`, `locComment`, and operational fields are descriptive facility metadata, not labels of plume presence or methane magnitude.

## Existing code paths and reuse

* `EarthRemoteSensingRapidResponse/ERSRR_Model.py` hardcodes this CSV path and passes it to `Plume_Classifier.evaluate_model` during evaluation.
* `EarthRemoteSensingRapidResponse/Plume_Classifier.py` reads only `Latitude` and `Longitude`, then matches predicted points using a 1 km geodesic threshold. It does not use `X_Coord/Y_Coord`, validate CRS, filter state, collapse duplicate sites, or attach a facility/site grouping key. Its `compute_image_center`/pixel conversion assumes geographic coordinates while raster scenes may be projected, so this path needs coordinate-CRS review before reuse for a new benchmark.
* `Data Collection/attach_cafo.py` is the most reusable low-level pattern: it detects common lat/lon names, interprets them as WGS84, transforms points to each raster CRS, and tests raster bounds. Its default `cafo_csv` directory is absent here, and the command copies rasters into output directories, so it was not run. It should be adapted to an explicit, provenance-locked Huang table and a dry-run/manifest mode before use.
* `Polygon_CSV_Files/Polygon_CAFO.py` recursively finds the named CSV, drops duplicate full rows, writes `Master_arcgis.csv`, and tests a hand-coded polygon. `Polygon_CAFO_Randomized.py` recursively reads every CSV, writes `points_inside.csv`/`points_outside.csv`, and uses a random bounding rectangle. These are exploratory plotting/scripts, not deterministic acquisition or split code; they must not define the five rotations.
* `tools/ersrr.py` audits paired GeoTIFF datasets and writes dataset manifests, but has no CAFO-coordinate audit or Huang import path. The existing acquisition documentation points to manifest-based, CRS-preserving S2/EMIT collection under `Data Collection/s2_emit_pairs/`; that path is reusable for imagery provenance, while facility coordinates should be joined separately by a stable facility/site key.

## Implications for Martin’s requested experiment

The proposed five rotations should be facility/site-grouped: hold out 80% of facility-image groups for validation and fit on 20%, with the same frozen facility coordinate source and deterministic rotation manifests. Do not split individual image tiles if multiple rows or acquisitions share a facility coordinate. Record whether the 20/80 direction is intentional, since it is opposite the common train/validation proportion. Keep facility coordinates as metadata used for spatial matching/eligibility, separate from EMIT plume masks or methane concentration targets. A later TESSERA comparison should preserve the same site grouping and report source/CRS/temporal differences rather than silently merging labels.

## Action needed before implementation

Obtain the Huang coordinate file or citation and confirm: authoritative release/version, row identity, coordinate order, horizontal datum/CRS and UTM zone(s), uncertainty semantics, intended state/geographic filter, and whether repeated facilities/sites should be grouped by a supplied site ID. Until then, this local ArcGIS CSV is a traceable baseline/reference inventory, not evidence that Huang coordinates are available.
