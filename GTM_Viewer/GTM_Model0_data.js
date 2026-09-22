window.GTM_DATA = {
  "schemaVersion": 1,
  "experiments": [
    {
      "id": "GTM_Model1",
      "name": "Compact ResUNet",
      "status": "Real checkpoint replay \u00b7 diagnostic examples",
      "decisionThreshold": 0.01,
      "description": "log1p B2/B3/B4/B11/B12 then five-channel z-score",
      "notes": "Two legacy validation-folder scenes in Texas. Fresh Keras/Torch inference, not a rerun of a frozen benchmark. Reference acquisition dates and product-level metadata are unverified; geographic holdout independence has not been established. These examples cannot validate US-wide performance.",
      "reportUrl": "GTM_Model0_visual_review.md",
      "scenes": [
        {
          "id": "20230203T171519_20230203T172113_T14RMU",
          "name": "T14RMU \u00b7 2023-02-03",
          "location": "Texas, United States",
          "date": "2023-02-03",
          "bounds": [
            [
              30.38697175405154,
              -99.92649376605561
            ],
            [
              30.433037096835793,
              -99.87342070839739
            ]
          ],
          "rgb": "assets/GTM_Model1_20230203T171519_20230203T172113_T14RMU_RGB.png",
          "emit": "assets/GTM_Model1_20230203T171519_20230203T172113_T14RMU_EMIT.png",
          "truthImage": "assets/GTM_Model1_20230203T171519_20230203T172113_T14RMU_reference.png",
          "grid": "assets/GTM_Model1_20230203T171519_20230203T172113_T14RMU_grid.json",
          "predictionAvailable": true,
          "checkpointHash": "d73dd7728626a6a2e584f1ff7757087ed84f26b41fd05596db53c952a84a48e6",
          "source": "20230203T171519_20230203T172113_T14RMU \u00b7 SHA256 c7c1daf78345ae65d65c248a12b7e1dfe710e59f8931ab1919ac14cacb62685c",
          "resolution": "Approx. 20 m native; 128\u00d7128 Mercator display (~40 m)",
          "split": "Legacy validation folder; not independently certified held-out sites",
          "referenceDate": "Unknown; not encoded in source filename",
          "rgbNote": "Observed B4/B3/B2; 2\u201398% display stretch",
          "emitNote": "Paired EMIT band; cyan intensity 0\u20133000 ppm\u00b7m, clipped for display",
          "truthLabel": "EMIT-derived reference mask",
          "truthNote": "EMIT >300 ppm\u00b7m where observed; missing pixels are unknown",
          "notes": "Diagnostic replay using four native 128\u00d7128 windows; no overlap blending. Artifact threshold 0.01. Approximately 20 m source grid; physical resolution was not resampled to an exact metric grid. EMIT date and Sentinel product level unverified. Most pixels lack EMIT support; no-data is not a negative. Output is plume probability, not methane concentration."
        },
        {
          "id": "20230410T172859_20230410T174507_T13RFQ",
          "name": "T13RFQ \u00b7 2023-04-10",
          "location": "Texas, United States",
          "date": "2023-04-10",
          "bounds": [
            [
              31.106971753990976,
              -103.60669271997764
            ],
            [
              31.153037096835785,
              -103.55322107060005
            ]
          ],
          "rgb": "assets/GTM_Model1_20230410T172859_20230410T174507_T13RFQ_RGB.png",
          "emit": "assets/GTM_Model1_20230410T172859_20230410T174507_T13RFQ_EMIT.png",
          "truthImage": "assets/GTM_Model1_20230410T172859_20230410T174507_T13RFQ_reference.png",
          "grid": "assets/GTM_Model1_20230410T172859_20230410T174507_T13RFQ_grid.json",
          "predictionAvailable": true,
          "checkpointHash": "d73dd7728626a6a2e584f1ff7757087ed84f26b41fd05596db53c952a84a48e6",
          "source": "20230410T172859_20230410T174507_T13RFQ \u00b7 SHA256 fce90283246a555ca87eef236af438465c0785eafb0ff25198a4802b7b33fe3f",
          "resolution": "Approx. 20 m native; 128\u00d7128 Mercator display (~40 m)",
          "split": "Legacy validation folder; not independently certified held-out sites",
          "referenceDate": "Unknown; not encoded in source filename",
          "rgbNote": "Observed B4/B3/B2; 2\u201398% display stretch",
          "emitNote": "Paired EMIT band; cyan intensity 0\u20133000 ppm\u00b7m, clipped for display",
          "truthLabel": "EMIT-derived reference mask",
          "truthNote": "EMIT >300 ppm\u00b7m where observed; missing pixels are unknown",
          "notes": "Diagnostic replay using four native 128\u00d7128 windows; no overlap blending. Artifact threshold 0.01. Approximately 20 m source grid; physical resolution was not resampled to an exact metric grid. EMIT date and Sentinel product level unverified. Most pixels lack EMIT support; no-data is not a negative. Output is plume probability, not methane concentration."
        }
      ]
    },
    {
      "id": "GTM_Model5",
      "name": "Physics-feature ResUNet",
      "status": "Real checkpoint replay \u00b7 diagnostic examples",
      "decisionThreshold": 0.05,
      "description": "log1p B2/B3/B4/B11/B12 plus six exact physics features, then 11-channel z-score",
      "notes": "Two legacy validation-folder scenes in Texas. Fresh Keras/Torch inference, not a rerun of a frozen benchmark. Reference acquisition dates and product-level metadata are unverified; geographic holdout independence has not been established. These examples cannot validate US-wide performance.",
      "reportUrl": "GTM_Model0_visual_review.md",
      "scenes": [
        {
          "id": "20230203T171519_20230203T172113_T14RMU",
          "name": "T14RMU \u00b7 2023-02-03",
          "location": "Texas, United States",
          "date": "2023-02-03",
          "bounds": [
            [
              30.38697175405154,
              -99.92649376605561
            ],
            [
              30.433037096835793,
              -99.87342070839739
            ]
          ],
          "rgb": "assets/GTM_Model5_20230203T171519_20230203T172113_T14RMU_RGB.png",
          "emit": "assets/GTM_Model5_20230203T171519_20230203T172113_T14RMU_EMIT.png",
          "truthImage": "assets/GTM_Model5_20230203T171519_20230203T172113_T14RMU_reference.png",
          "grid": "assets/GTM_Model5_20230203T171519_20230203T172113_T14RMU_grid.json",
          "predictionAvailable": true,
          "checkpointHash": "bf3b7bf2c37525c225c11a9ac0e7adc1b69b850ccab9e8db957c063440ca623d",
          "source": "20230203T171519_20230203T172113_T14RMU \u00b7 SHA256 c7c1daf78345ae65d65c248a12b7e1dfe710e59f8931ab1919ac14cacb62685c",
          "resolution": "Approx. 20 m native; 128\u00d7128 Mercator display (~40 m)",
          "split": "Legacy validation folder; not independently certified held-out sites",
          "referenceDate": "Unknown; not encoded in source filename",
          "rgbNote": "Observed B4/B3/B2; 2\u201398% display stretch",
          "emitNote": "Paired EMIT band; cyan intensity 0\u20133000 ppm\u00b7m, clipped for display",
          "truthLabel": "EMIT-derived reference mask",
          "truthNote": "EMIT >300 ppm\u00b7m where observed; missing pixels are unknown",
          "notes": "Diagnostic replay using four native 128\u00d7128 windows; no overlap blending. Artifact threshold 0.05. Approximately 20 m source grid; physical resolution was not resampled to an exact metric grid. EMIT date and Sentinel product level unverified. Most pixels lack EMIT support; no-data is not a negative. Output is plume probability, not methane concentration."
        },
        {
          "id": "20230410T172859_20230410T174507_T13RFQ",
          "name": "T13RFQ \u00b7 2023-04-10",
          "location": "Texas, United States",
          "date": "2023-04-10",
          "bounds": [
            [
              31.106971753990976,
              -103.60669271997764
            ],
            [
              31.153037096835785,
              -103.55322107060005
            ]
          ],
          "rgb": "assets/GTM_Model5_20230410T172859_20230410T174507_T13RFQ_RGB.png",
          "emit": "assets/GTM_Model5_20230410T172859_20230410T174507_T13RFQ_EMIT.png",
          "truthImage": "assets/GTM_Model5_20230410T172859_20230410T174507_T13RFQ_reference.png",
          "grid": "assets/GTM_Model5_20230410T172859_20230410T174507_T13RFQ_grid.json",
          "predictionAvailable": true,
          "checkpointHash": "bf3b7bf2c37525c225c11a9ac0e7adc1b69b850ccab9e8db957c063440ca623d",
          "source": "20230410T172859_20230410T174507_T13RFQ \u00b7 SHA256 fce90283246a555ca87eef236af438465c0785eafb0ff25198a4802b7b33fe3f",
          "resolution": "Approx. 20 m native; 128\u00d7128 Mercator display (~40 m)",
          "split": "Legacy validation folder; not independently certified held-out sites",
          "referenceDate": "Unknown; not encoded in source filename",
          "rgbNote": "Observed B4/B3/B2; 2\u201398% display stretch",
          "emitNote": "Paired EMIT band; cyan intensity 0\u20133000 ppm\u00b7m, clipped for display",
          "truthLabel": "EMIT-derived reference mask",
          "truthNote": "EMIT >300 ppm\u00b7m where observed; missing pixels are unknown",
          "notes": "Diagnostic replay using four native 128\u00d7128 windows; no overlap blending. Artifact threshold 0.05. Approximately 20 m source grid; physical resolution was not resampled to an exact metric grid. EMIT date and Sentinel product level unverified. Most pixels lack EMIT support; no-data is not a negative. Output is plume probability, not methane concentration."
        }
      ]
    },
    {
      "id": "GTM_Model0",
      "name": "CAFO acquisition pilot",
      "status": "Observed inputs \u00b7 no inference",
      "description": "Three bounded Sentinel-2 crops from the provisional Iowa inventory.",
      "notes": "Three complete crops; one additional site rejected at 36.47% clear pixels. Acquisition coverage is not methane-detection recall.",
      "metrics": {
        "Completed": "3",
        "Quality rejected": "1",
        "Imagery payload": "30.34 MiB"
      },
      "scenes": [
        {
          "id": "56216",
          "name": "Inventory site 56216",
          "location": "Iowa, United States",
          "date": "2024-12-06",
          "bounds": [
            [
              41.46561323572109,
              -92.38232723511503
            ],
            [
              41.4772254865391,
              -92.36688790178155
            ]
          ],
          "rgb": "assets/GTM_Model0_CAFO56216.png",
          "rgbNote": "Sentinel-2 L2A display RGB; yellow inventory marker",
          "notes": "Input-only acquisition pilot. No model prediction or methane ground truth. WGS84 coordinate assumption; facility identity unreviewed. RGB stretched for display. Geographic image bounds approximate the UTM crop; not for pixel alignment scoring.",
          "resolution": "10 m RGB / 20 m SWIR",
          "source": "S2C_15TWF_20241206_1_L2A",
          "split": "Engineering pilot, not a classifier test"
        },
        {
          "id": "56217",
          "name": "Inventory site 56217",
          "location": "Iowa, United States",
          "date": "2024-12-19",
          "bounds": [
            [
              40.91441675927428,
              -95.68242207654102
            ],
            [
              40.926287314346034,
              -95.66677262424051
            ]
          ],
          "rgb": "assets/GTM_Model0_CAFO56217.png",
          "rgbNote": "Sentinel-2 L2A display RGB; yellow inventory marker",
          "notes": "Input-only acquisition pilot. No model prediction or methane ground truth. WGS84 coordinate assumption; facility identity unreviewed. RGB stretched for display. Geographic image bounds approximate the UTM crop; not for pixel alignment scoring.",
          "resolution": "10 m RGB / 20 m SWIR",
          "source": "S2A_15TTF_20241219_0_L2A",
          "split": "Engineering pilot, not a classifier test"
        },
        {
          "id": "56218",
          "name": "Inventory site 56218",
          "location": "Iowa, United States",
          "date": "2024-12-06",
          "bounds": [
            [
              42.57927509502598,
              -92.92176670398138
            ],
            [
              42.59081415000847,
              -92.90615151845391
            ]
          ],
          "rgb": "assets/GTM_Model0_CAFO56218.png",
          "rgbNote": "Sentinel-2 L2A display RGB; yellow inventory marker",
          "notes": "Input-only acquisition pilot. No model prediction or methane ground truth. WGS84 coordinate assumption; facility identity unreviewed. RGB stretched for display. Geographic image bounds approximate the UTM crop; not for pixel alignment scoring.",
          "resolution": "10 m RGB / 20 m SWIR",
          "source": "S2C_15TVH_20241206_1_L2A",
          "split": "Engineering pilot, not a classifier test"
        }
      ],
      "report": "CAFO pilot receipts, 2026-09-18"
    },
    {
      "id": "GTM_Model2",
      "name": "Frozen ERSRR successor",
      "status": "Historical benchmark \u00b7 summary only",
      "description": "Exact MARS-S2L paper benchmark, full view.",
      "metrics": {
        "Scene AP": "0.64693",
        "Recall": "0.81688",
        "FPR": "0.07069",
        "Pixel IoU": "0.33233"
      },
      "notes": "Does not pass every preregistered superiority gate. Global historical cohort, not a US validation. Matched-FPR operating point. Aligned imagery and prediction exports are not locally available.",
      "scenes": [],
      "report": "MARS_SUCCESSOR_PAPER_TEST.md",
      "reportUrl": "assets/MARS_SUCCESSOR_PAPER_TEST.md"
    },
    {
      "id": "GTM_Model3",
      "name": "Spatial successor",
      "status": "Post-test development \u00b7 summary only",
      "description": "Exact MARS-S2L paper benchmark after post-test development.",
      "metrics": {
        "Full scene AP": "0.67210",
        "Unseen-site AP": "0.46774",
        "Superiority gates": "FAIL"
      },
      "notes": "This is not an untouched confirmation cohort. Unseen-site AP improvement interval crosses zero. No aligned prediction rasters available; cannot infer map quality from scores alone.",
      "scenes": [],
      "report": "MARS_SPATIAL_SUCCESSOR_PAPER_POSTTEST.md",
      "reportUrl": "assets/MARS_SPATIAL_SUCCESSOR_PAPER_POSTTEST.md"
    },
    {
      "id": "GTM_Model4",
      "name": "ERSRR v5.1 \u00b7 MethaneS2CM",
      "status": "Frozen location test \u00b7 summary only",
      "description": "Three-seed model trained on MethaneS2CM L2A.",
      "metrics": {
        "Scene AP": "0.8180",
        "Recall": "0.3778",
        "FPR": "0.0607",
        "Pixel IoU": "0.1852"
      },
      "notes": "All predeclared comparison checks did not pass. Precision on the roughly balanced crop benchmark is not deployment precision. L1C-trained zero-shot comparators do not isolate architecture effects. No aligned prediction exports available locally.",
      "scenes": [],
      "report": "METHANES2CM_V5_1_LOCATION_TEST.md",
      "reportUrl": "assets/METHANES2CM_V5_1_LOCATION_TEST.md"
    }
  ]
};
