window.GTM_FRIDAY_DATA = {
  "title": "Methane research review",
  "meetingDate": "2026-10-02",
  "builtDate": "2026-09-30",
  "summary": "Six saved evidence cases: one EMIT/Sentinel input review and five MARS model scenes. The pilot fails its quality gate; use this review to choose the next measurable milestone.",
  "aggregate": {
    "scenes": 65,
    "positiveScenes": 3,
    "iou": 0.02148471615720524,
    "precision": 0.022487316604963663,
    "recall": 0.325181758096497,
    "status": "Failed quality gate",
    "scope": "Pooled native pixels within actual central support; reused development fold"
  },
  "cases": [
    {
      "id": "volusia_20250601",
      "label": "Volusia intake",
      "kind": "data",
      "date": "2025-06-01",
      "description": "Actual Sentinel-2 RGB and EMIT matched-filter retrieval from the 2025 Volusia intake. The cyan outline marks the saved provider mask displayed at 10 m; its annotation is native 60 m, not a 10 m prediction or verified fine-resolution truth.",
      "lesson": "A 12.74-minute overlap and five available target bands support an input review. Label-zero validity, retrieval units, radiometric scale/offset, a reference-date pair, and a reviewed negative still block training.",
      "panels": [
        {
          "title": "Sentinel-2 RGB",
          "image": "assets/case_volusia_rgb.png",
          "caption": "Saved display stretch of actual Sentinel-2 target RGB."
        },
        {
          "title": "EMIT retrieval + provider mask",
          "image": "assets/case_volusia_retrieval.png",
          "caption": "Saved matched-filter display; cyan is the saved 60 m mask boundary shown on the 10 m display grid."
        },
        {
          "title": "Model prediction",
          "image": null,
          "emptyReason": "No model prediction exists for this intake.",
          "caption": "Input review only. No Volusia model score or enhancement result."
        }
      ],
      "metrics": [
        {
          "label": "Time offset",
          "value": "12.74 min"
        },
        {
          "label": "Available target bands",
          "value": "5 / 5"
        },
        {
          "label": "Training gate",
          "value": "Closed"
        }
      ],
      "fullViewerUrl": "../GTM_Model6_emit_2025_intake.html"
    },
    {
      "id": "MARS_2c6ef011-0729-42e2-b64f-10ff8207778b",
      "label": "Localized result",
      "kind": "prediction",
      "date": "2023-11-01",
      "description": "MARS reviewed plume scene. Saved mean of four independent scale logits, displayed at the fixed 0.50 threshold. The model predicts only the central 88 × 88 pixels of the 200 × 200 source grid (19.36%); gray stripes mean no output, not a true negative.",
      "lesson": "Compare the red model activation against the cyan reviewed reference within the scored center. This is a reused development fold and does not establish independent performance or source origin.",
      "panels": [
        {
          "title": "Sentinel-2 RGB",
          "image": "assets/case_0000_rgb.png",
          "caption": "Same full scene field; striped gray pixels lack valid model output."
        },
        {
          "title": "Reviewed reference",
          "image": "assets/case_0000_reference.png",
          "caption": "Cyan outline is the saved MARS reviewed plume mask within valid prediction support."
        },
        {
          "title": "Saved model output",
          "image": "assets/case_0000_prediction.png",
          "caption": "Orange-red is saved mean probability ≥ 0.50; cyan is the reference boundary. No output in gray stripes."
        }
      ],
      "metrics": [
        {
          "label": "Native IoU",
          "value": "38.8%"
        },
        {
          "label": "False positives",
          "value": "192"
        },
        {
          "label": "True positives",
          "value": "348"
        },
        {
          "label": "Support",
          "value": "19.36%"
        }
      ],
      "fullViewerUrl": "../index.html?experiment=GTM_Model6_mapper_r5_portable&scene=MARS_2c6ef011-0729-42e2-b64f-10ff8207778b#compare"
    },
    {
      "id": "MARS_2d64f495-fd6b-491c-8614-69f7d303fa1d",
      "label": "Missed plume",
      "kind": "prediction",
      "date": "2023-06-04",
      "description": "MARS reviewed plume scene. Saved mean of four independent scale logits, displayed at the fixed 0.50 threshold. The model predicts only the central 88 × 88 pixels of the 200 × 200 source grid (19.36%); gray stripes mean no output, not a true negative.",
      "lesson": "Compare the red model activation against the cyan reviewed reference within the scored center. This is a reused development fold and does not establish independent performance or source origin.",
      "panels": [
        {
          "title": "Sentinel-2 RGB",
          "image": "assets/case_0001_rgb.png",
          "caption": "Same full scene field; striped gray pixels lack valid model output."
        },
        {
          "title": "Reviewed reference",
          "image": "assets/case_0001_reference.png",
          "caption": "Cyan outline is the saved MARS reviewed plume mask within valid prediction support."
        },
        {
          "title": "Saved model output",
          "image": "assets/case_0001_prediction.png",
          "caption": "Orange-red is saved mean probability ≥ 0.50; cyan is the reference boundary. No output in gray stripes."
        }
      ],
      "metrics": [
        {
          "label": "Native IoU",
          "value": "0.0%"
        },
        {
          "label": "False positives",
          "value": "257"
        },
        {
          "label": "True positives",
          "value": "0"
        },
        {
          "label": "Support",
          "value": "19.36%"
        }
      ],
      "fullViewerUrl": "../index.html?experiment=GTM_Model6_mapper_r5_portable&scene=MARS_2d64f495-fd6b-491c-8614-69f7d303fa1d#compare"
    },
    {
      "id": "MARS_35027778-4dc9-494b-b876-fb8391c69a61",
      "label": "Partial result",
      "kind": "prediction",
      "date": "2023-11-21",
      "description": "MARS reviewed plume scene. Saved mean of four independent scale logits, displayed at the fixed 0.50 threshold. The model predicts only the central 88 × 88 pixels of the 200 × 200 source grid (19.36%); gray stripes mean no output, not a true negative.",
      "lesson": "Compare the red model activation against the cyan reviewed reference within the scored center. This is a reused development fold and does not establish independent performance or source origin.",
      "panels": [
        {
          "title": "Sentinel-2 RGB",
          "image": "assets/case_0002_rgb.png",
          "caption": "Same full scene field; striped gray pixels lack valid model output."
        },
        {
          "title": "Reviewed reference",
          "image": "assets/case_0002_reference.png",
          "caption": "Cyan outline is the saved MARS reviewed plume mask within valid prediction support."
        },
        {
          "title": "Saved model output",
          "image": "assets/case_0002_prediction.png",
          "caption": "Orange-red is saved mean probability ≥ 0.50; cyan is the reference boundary. No output in gray stripes."
        }
      ],
      "metrics": [
        {
          "label": "Native IoU",
          "value": "25.4%"
        },
        {
          "label": "False positives",
          "value": "220"
        },
        {
          "label": "True positives",
          "value": "144"
        },
        {
          "label": "Support",
          "value": "19.36%"
        }
      ],
      "fullViewerUrl": "../index.html?experiment=GTM_Model6_mapper_r5_portable&scene=MARS_35027778-4dc9-494b-b876-fb8391c69a61#compare"
    },
    {
      "id": "MARS_2a0831e4-fde7-40ea-8c39-eb0ea8675c89",
      "label": "False alarm 869",
      "kind": "prediction",
      "date": "2023-01-25",
      "description": "MARS reviewed no-plume scene. Saved mean of four independent scale logits, displayed at the fixed 0.50 threshold. The model predicts only the central 88 × 88 pixels of the 200 × 200 source grid (19.36%); gray stripes mean no output, not a true negative.",
      "lesson": "False activation on a reviewed negative is a core failure mode. Gray outside the supported center is unavailable and is excluded from scoring.",
      "panels": [
        {
          "title": "Sentinel-2 RGB",
          "image": "assets/case_0034_rgb.png",
          "caption": "Same full scene field; striped gray pixels lack valid model output."
        },
        {
          "title": "Reviewed reference",
          "image": "assets/case_0034_reference.png",
          "caption": "Cyan outline is the saved MARS reviewed plume mask within valid prediction support."
        },
        {
          "title": "Saved model output",
          "image": "assets/case_0034_prediction.png",
          "caption": "Orange-red is saved mean probability ≥ 0.50; cyan is the reference boundary. No output in gray stripes."
        }
      ],
      "metrics": [
        {
          "label": "Native IoU",
          "value": "0.0%"
        },
        {
          "label": "False positives",
          "value": "869"
        },
        {
          "label": "True positives",
          "value": "0"
        },
        {
          "label": "Support",
          "value": "19.36%"
        }
      ],
      "fullViewerUrl": "../index.html?experiment=GTM_Model6_mapper_r5_portable&scene=MARS_2a0831e4-fde7-40ea-8c39-eb0ea8675c89#compare"
    },
    {
      "id": "MARS_b2453f15-531e-4377-8576-f46fbd2234f7",
      "label": "False alarm 717",
      "kind": "prediction",
      "date": "2023-10-27",
      "description": "MARS reviewed no-plume scene. Saved mean of four independent scale logits, displayed at the fixed 0.50 threshold. The model predicts only the central 88 × 88 pixels of the 200 × 200 source grid (19.36%); gray stripes mean no output, not a true negative.",
      "lesson": "False activation on a reviewed negative is a core failure mode. Gray outside the supported center is unavailable and is excluded from scoring.",
      "panels": [
        {
          "title": "Sentinel-2 RGB",
          "image": "assets/case_0012_rgb.png",
          "caption": "Same full scene field; striped gray pixels lack valid model output."
        },
        {
          "title": "Reviewed reference",
          "image": "assets/case_0012_reference.png",
          "caption": "Cyan outline is the saved MARS reviewed plume mask within valid prediction support."
        },
        {
          "title": "Saved model output",
          "image": "assets/case_0012_prediction.png",
          "caption": "Orange-red is saved mean probability ≥ 0.50; cyan is the reference boundary. No output in gray stripes."
        }
      ],
      "metrics": [
        {
          "label": "Native IoU",
          "value": "0.0%"
        },
        {
          "label": "False positives",
          "value": "717"
        },
        {
          "label": "True positives",
          "value": "0"
        },
        {
          "label": "Support",
          "value": "19.36%"
        }
      ],
      "fullViewerUrl": "../index.html?experiment=GTM_Model6_mapper_r5_portable&scene=MARS_b2453f15-531e-4377-8576-f46fbd2234f7#compare"
    }
  ],
  "architectureImage": "assets/GTM_Model6_architecture.svg",
  "architectureCaption": "Current saved diagnostic: independent S16/S32/S64/S128 logits averaged at inference. No learned fusion, source-origin head, or quantitative EMIT enhancement was trained.",
  "decisions": [
    {
      "question": "What acceptance criteria and sequence serve the fixed US objective?",
      "why": "Agree measurable gates for plume regions, quantitative methane mapping, and emission origin. The current saved model addresses only plume segmentation and has failed its quality gate."
    },
    {
      "question": "What provider documentation confirms retrieval units and label-zero validity?",
      "why": "The Volusia annotation cannot support quantitative targets or verified negatives until those semantics are established."
    },
    {
      "question": "What source and wind labels will define an origin task?",
      "why": "Plume shape alone does not verify an emission origin; source and wind evidence must be specified before training that head."
    }
  ],
  "nextSteps": [
    {
      "title": "Set milestone criteria and sequence",
      "detail": "Keep the US methane objective fixed; agree held-out success criteria and order for plume regions, quantitative map, and emission origin before new model work.",
      "status": "Decision needed"
    },
    {
      "title": "Close Volusia data semantics",
      "detail": "Five target bands are local. Resolve exact-release label-zero validity, retrieval units, and target radiometric scale/offset.",
      "status": "Blocked"
    },
    {
      "title": "Build an independent paired cohort",
      "detail": "Acquire a matching reference-date crop and reviewed negative, then evaluate outside the reused development fold.",
      "status": "Blocked"
    }
  ],
  "pdf": "GTM_Model6_Friday_brief_20261002.pdf"
};
