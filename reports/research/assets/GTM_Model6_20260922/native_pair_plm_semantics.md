# Clarification of historical PLM counts

The v3 report's `native_plume_positive_pixels` field counts positive continuous CH4PLM enhancement values. It is not a verified plume-pixel count or a binary reference label. Native raster values can include background and negative retrieval noise. The current tool no longer uses positive-valued PLM intersection as a readiness gate. The independent findings that none of the six original crops satisfies both timing and peak coverage remain unchanged.

See the [repaired-pair QA](../GTM_Model6_georgia_crop_repair_20260922/qa_v2/GTM_Model6_georgia_repaired_pair_qa.md) for actual continuous-value checks and [NASA's V2 guide](https://lpdaac.usgs.gov/documents/2250/EMIT_L2B_GHG_User_Guide_V2.pdf) for the separate vector outline product.
