# Local viewer verification — 2026-09-18

- Windows PowerShell 5 launcher starts a hidden loopback-only server and reuses it on a second invocation. Corrected an unavailable HTTP exception type found during that test.
- Catalog loads six experiments, including eight authentic v5.1 scenes with all four scene outcomes. Rebuilt QA manifest no longer shadows the canonical bundle. Duplicate persistent experiment IDs produce warnings.
- All 47 initial catalog asset references returned HTTP200; the subsequently added v5.1 human-readable evidence report also returned HTTP200. Path traversal checks returned404.
- Browser inspection confirmed linked image views, georeferenced legacy view, pixel/geographic experiment switching, outcome filters, probability/mask switching, and folder reload preserving the current experiment.
- v5.1 sample86360: probability .618–.639, 100% positive at .40, pixel TP251/FP773, scene TP. Sample83149: .013–.019, 0% positive, pixel TN1024, scene TN. Sample82448: .611–.632, 100% positive, pixel FP1024, scene FP.
- JavaScript syntax and Python parse checks passed. Model outputs were not changed, retrained, or downloaded.
- v5.1 uses native image coordinates because the cached point locations do not provide verified footprints. Legacy exports retain geographic coordinates. Online basemap defaults off.

## Direct overlay revision

- Verified via HTTPS Tailscale URL at desktop and390x844 mobile viewport. Default combined overlay shows sample86360's entire orange prediction against the smaller cyan ground-truth boundary; reference pane remains separate and unchanged.
- Ground-truth toggle removes both boundary paths from prediction pane (2→0) while the reference pane retains2 paths. Pixel agreement values remain unchanged.
- At threshold.99, sample86360 gives pixel TP0/FP0/FN251/TN773 while the ground-truth area remains24.5% and the independent scene decision staysTP. Reset returns the frozen.40 threshold.
- Overlay and side-by-side layouts, mobile scene picker, and layer controls inspected. Fixed mobile collapsed-navigation specificity and hidden-element handling discovered during visual checks. JavaScript syntax passes.
