# MethaneUnion acquisition efficiency audit — 2026-09-04

## Decision

The frozen 480 m candidate acquisition is paused at its data-feasibility gate. Do not
restart the blind full-archive scan without an explicit decision accepting the remaining
transfer or a corrected archive/member map from the dataset maintainer.

This work was not scientifically empty: it identified a truncated repository inventory,
repaired the frozen source identity, proved that the first 102 immutable shards do not
contain any of the 6,528 requested exact member paths, and exposed a mismatch between the
published manifests and packaged archive paths. Continuing the same method is now poor
expected value.

## Verified state

- Source: `yuyao42/MethaneUnion` at immutable revision
  `e654cd777f85d701a4aff2894f513dcdf6981531`.
- Frozen inventory: 144 archives totaling 764,197,113,632 bytes.
- Fully downloaded, checksum-verified, and exact-path scanned: 102 archives totaling
  540,791,785,536 bytes.
- Scan interval: 164.66 minutes from the first through last receipt.
- Selected candidate members: 0; selected assets/TIFFs: 0.
- Remaining: 42 archives totaling 223,405,328,096 bytes.
- Preserved partial shard 103 download: 939,066,813 bytes. A zero-byte lock file is stale;
  neither file is evidence of a running process.
- Final acquisition report and row adjudication: not produced because the frozen data gate
  was not reached.

## Bounded remaining-shard survey

A Range-request diagnostic read the first 262,144 compressed bytes of each remaining shard.
The corrected PAX-aware pass used 11,010,048 bytes; together with the superseded PAX-only
pass, the diagnostic transferred at most 22,020,096 bytes.

The first real members show an ordered packaging layout:

- shards 103–110: `data/original/provided_split/test/emit/...`;
- shards 111–124: `data/original/geo/train/s2/<id>/s2.tif`;
- shards 125–139: `data/original/geo/train/s2/<id>/s2_pre.tif`;
- shards 140–144: `data/original/geo/train/s2/<id>/s2_pre_pre.tif`.

This is not a complete absence proof for later members in a shard. It is strong triage
evidence, however, because the frozen candidate contract requests
`data/480m_GSD/s2/<id>/...`, while the remaining prefixes expose original-scale paths.

## Release-contract mismatch

The public 480 m manifest provides the exact paths used by the frozen acquisition, but none
were found in 102 complete shard inventories. The original-scale training manifest contains
1,633 rows at the same 56 novel physical coordinates, balanced across 804 negatives and 829
positives, but it names `data/original_scale/train/s2/<id>/...`. The observed archives use
`data/original/geo/train/s2/<id>/...`. Treating these as aliases would silently change the
frozen data contract and still would not establish where all required frames and plume masks
are packaged.

## Recommended next evidence request

Ask the MethaneUnion maintainers for one of the following, pinned to the same release:

1. a complete TAR member index mapping manifest paths to archive names;
2. a corrected manifest matching the packaged `data/original/geo/...` layout; or
3. confirmation that the derived 480 m assets were omitted, plus a corrected download.

If none is available, close this candidate cohort as infeasible and use a different source
of genuinely new geographic training evidence. Do not reinterpret the zero-hit scan as a
model result, and do not open protected MARS-S2L evidence to compensate for the failed data
gate.
