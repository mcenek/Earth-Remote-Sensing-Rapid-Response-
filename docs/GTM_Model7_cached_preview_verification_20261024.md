# Cached input preview verification

Before changing the display, prevent these failures:

1. Different per-date scaling conceals illumination differences. Use one scalar range pooled across both dates and all RGB bands, with a minimum 1,000 DN range. Preserve identical mapping for each band and date. This is display DN, not calibrated reflectance or a learned output.
2. An annotation becomes an inferred plume. Label every figure as current input, observed earlier reference, or released source annotation. Do not add an experiment catalog entry or prediction file.
3. A display fix alters training. Rebuild the 561 source arrays independently, retain both input manifest hashes, and write a separate v2 preview folder. Preserve v1.
4. Narrow-stretch artifacts are mistaken for real image defects or all references are called perfect. Preserve raw source evidence and image-only diagnostics. Record actual haze/brightness differences and avoid claiming validated subpixel registration.
5. A polished but broken report is delivered. Check every served PNG and the JSON receipt against local hashes. In the browser, check both filters, visible counts, image loading, source details and a screenshot. Save a repeatable E2E artifact.

The root agent inspects contact sheets for all 100 positive sites and samples of no-plume inputs. The cloud mask and metadata remain the producer's observations, not independent methane measurements.

## Display amendment before v3

Inspection of v2 found that a single range across raw L1C RGB bands creates a strong orange cast over desert sites. For v3, use a separate pooled two-date range for each RGB band, each at least 1,000 DN wide. Both dates still use exactly the same mapping in each band. Preserve v2. This display balancing is not atmospheric correction and is not used by training. Recheck extreme-brightness pairs and negative examples.
