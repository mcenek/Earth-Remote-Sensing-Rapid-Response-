# Iowa CAFO CSV provenance follow-up (2026-09-18)

## Result

The repository contains one CAFO inventory table:

`EarthRemoteSensingRapidResponse/Polygon_CSV_Files/iowa_cafos_2024_arcgis_api.csv`

The file is strongly evidenced as an Iowa Department of Natural Resources (Iowa DNR) / ArcGIS OneStop facility export. Its schema, row-level Iowa DNR facility-summary hyperlinks, and field names match the official Iowa DNR ArcGIS service. I found no evidence in the repository, the file, or its Git history that this particular export is Huang's coordinate file. Huang attribution remains **unverified**.

The file is suitable as a traceable inventory/reference frame and for a bounded facility-location recovery pilot. It is not, by itself, a date-specific methane label or proof that every row is a regulatory CAFO at an image acquisition date.

## Local file and acquisition trail

| Item | Verified finding |
|---|---|
| Rows / columns | 14,155 data rows; 55 CSV fields |
| SHA-256 (current checkout) | `5571BB76D2F012DE70A0022CDF5FE38F65A9D17D9908752E164872062B272D38` |
| Git introduction | Commit `52096ea106200f8a52c09ce20d3731c8c5d89da1`, 2025-12-05, author `Gonon` |
| Git commit description | “This is my attempt to determine which coordinates in a CAFO csv file are inside a polygon area.” |
| Acquisition code found | No downloader/API acquisition script was found. `Polygon_CAFO.py` recursively locates the already-local filename; `Polygon_CAFO_Randomized.py` reads local CSV files for exploratory polygons. |
| Embedded source links | Every row has an Iowa DNR `afoemmp/Facility/FacilitySummaryByID?facilityID=...` hyperlink. |

The Git commit adds the CSV and two exploratory polygon scripts together; it does not record a source URL, ArcGIS item ID, export timestamp, author named Huang, or coordinate-reference-system declaration.

## Official Iowa DNR web metadata

The official Iowa DNR ArcGIS REST service is [OneStop/IaFmFacilities (MapServer)](https://programs.iowadnr.gov/geospatial/rest/services/OneStop/IaFmFacilities/MapServer). Its service directory lists these relevant point layers:

* [Animal Feeding Facility, layer 2](https://programs.iowadnr.gov/geospatial/rest/services/OneStop/IaFmFacilities/MapServer/2)
* [Confinement, layer 3](https://programs.iowadnr.gov/geospatial/rest/services/OneStop/IaFmFacilities/MapServer/3)
* [Open Feedlots, layer 4](https://programs.iowadnr.gov/geospatial/rest/services/OneStop/IaFmFacilities/MapServer/4)

The service metadata identifies the layers as point feature layers and exposes fields matching the local CSV, including `progid`, `facName`, `opStatus`, `OperatType`, `AnimalUnit`, `Latitude`, `Longitude`, `X_Coord`, `Y_Coord`, `Accuracy`, `colMthTxt`, `refPntTxt`, `VerifyTxt`, `collectBy`, `ColDate`, and `Hyperlink`. The layer pages describe the displayed service spatial reference as EPSG/Web Mercator Auxiliary Sphere `102100 (3857)`.

That 3857 declaration is for the ArcGIS service/map geometry. It does **not** prove that the CSV's numeric `Latitude`/`Longitude` fields use a particular horizontal datum, nor that `X_Coord`/`Y_Coord` use Web Mercator. The CSV itself has no `CRS`, `EPSG`, datum, UTM-zone, or export-metadata field.

## Inventory scope and status

Verified local value distributions include:

| Field | Values observed |
|---|---|
| `opStatus` | 12,703 `Active`; 1,452 `Inactive` |
| `OperatType` | 10,231 `Confinement`; 2,487 `Open Feedlot`; 1,387 `Confinement/Open Feedlot`; 21 truck-wash variants; 10 `Confinement/Digester`; 9 `Confinement/Digester/Open Feedlot`; 5 `Livestock Truck Wash/Open Feedlot`; 3 `Confinement/Livestock Truck Wash`; 1 blank; 1 `Confinement/Livestock Truck Wash/Open Feedlot` |
| `State` | 14,131 `IA`; 16 `XX`; 6 `MN`; 1 `AR`; 1 `MO` |
| `Accuracy` | 20 for 10,537 rows; 570 for 1,574; 0 for 1,282; smaller counts at 25, 280, 1,140, 100, 150, 50, 15, 200, 10, 250, 3, and 40 |
| `colMthTxt` | 11,345 `INTERPOLATION-PHOTO`; 1,369 `ADDRESS MATCHING-STREET CENTERLINE`; 310 `INTERPOLATION-MAP`; 275 `PUBLIC LAND SURVEY - SIXTEENTH SECTION`; 118 quarter-section; remaining rows use other methods or are blank |

The service/layer names and fields establish that these are mapped animal-feeding/environmental-facility records. They do not establish the legal definition used for every row in this export, current permit status at a target overpass, historical operation on a filename year, or methane emission activity. `Active`/`Inactive` should therefore remain inventory status fields, not methane labels.

## Coordinate findings

All rows have parseable geographic coordinates in the expected Iowa region (approximately latitude 40.48–43.93 and longitude -96.59–-90.20). All rows also have populated numeric `X_Coord`/`Y_Coord` values. The projected ranges (approximately X 206,914–731,949 and Y 4,482,118–4,864,635) are compatible with meter-based UTM-like coordinates, but the source does not declare their CRS or datum. Do not silently label them NAD83/UTM zone 15N or any other EPSG code without source confirmation.

The coordinate quality fields matter for crop-level recovery: the dominant `Accuracy` value is 20, but 1,574 rows have 570 and 1,282 have 0; collection method and reference-point fields vary substantially. There are multiple distinct facility records at identical coordinates, so deduplicating by coordinate would discard inventory identities. Preserve `uniqueID`/`progid`, raw coordinates, projected coordinates, `Accuracy`, collection fields, and the original hyperlink.

The current `attach_cafo.py` helper explicitly assumes WGS84 for CSV longitude/latitude and transforms those points into each raster CRS. That is a reasonable operational assumption for a pilot only after it is recorded as an assumption; it is not evidence of the inventory's declared datum. `Plume_Classifier.py` uses geographic latitude/longitude and a 1 km matching threshold but does not validate the raster/coordinate CRS, use `X_Coord`/`Y_Coord`, or filter/group the inventory. Its output should remain a candidate diagnostic until those assumptions are audited.

## Huang attribution: verified versus unknown

**Verified:** the local filename, schema, row hyperlinks, field vocabulary, and official service metadata point to Iowa DNR's ArcGIS facility inventory. The tracked import was made by `Gonon` in the December 2025 polygon experiment.

**Unknown:** whether Huang supplied, curated, transformed, or recommended this export; whether Huang has a separate GPS/UTM file; the authoritative release/version; the source datum and UTM zone for `X_Coord`/`Y_Coord`; coordinate uncertainty semantics beyond the opaque `Accuracy` values; and whether “2024” is an observation/snapshot year rather than a filename convention.

No Huang paper, citation, data release, file attribution, or message is present in the inspected repository artifacts. The existing research notes correctly keep the Huang association pending confirmation.

## Consequence for the authorized CAFO recovery experiment

Continue with a bounded, provenance-labeled facility-recovery/observability pilot using this file as a **provisional Iowa DNR inventory**, preserving active and inactive rows for audit and using active rows only as the initial sampling frame if that choice is predeclared. Keep facility recovery separate from methane truth. A detection at a listed facility is evidence of spatial association or a review candidate; it is not a date-specific methane true positive. Unlisted detections remain unclassified pending source review.

Before a scientific benchmark or Huang-specific claim, obtain the Huang file/citation and confirm row identity, coordinate order, horizontal datum/CRS, UTM zone(s), accuracy semantics, intended geographic filter, facility/site grouping key, and activity-date meaning. Record the source hash and these assumptions in every acquisition manifest.
