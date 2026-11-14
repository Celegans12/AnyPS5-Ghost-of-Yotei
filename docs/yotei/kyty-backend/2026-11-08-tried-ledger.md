# Ledger (2026-11-08): every speed lever and experiment we tried, with its result

This is the single list behind `PLAN-v4-draft.md`. It was compiled from:
- the archive;
- `PLAN.md` (v2) and `PLAN-v3-draft.md`;
- every lane report in `reports/` from 10-27 on (the Codex PR, library and upkeep files are left out);
- the `research/` result files.

Numbers are presents/10 s (10 = 1 fps) or % presents from an in-run A/B (neighbour pairs, mean ± SE), unless a row
says otherwise. Before 10-29 runs are at 4K, and from 10-29 on at 1080p. [M] measured, [E] estimate. Each row names
its source report; paths are under `C:\ps5\work\`.

## 0. Speed over time

| date | build / change | speed | resolution | source |
|---|---|---|---|---|
| 10-09 | old pre-#639 base, opening cinematic | 13.5 → 3.8 s/frame (~0.26 fps) | 3328×1872 | perf-step1 |
| 10-11..16 | main tree, loading (no world yet); title ~5.5 fps | loading 13.8 → 0.5 s/frame | 4K | PLAN-10-16 |
| 10-17 | first open world (long12) | 2.35 presents/10 s | 4K | world/long12-profile |
| 10-18 | drawcpu, fps-draw, merge, gpuind, device-local BDA | 3.0 → 4.4-4.9 | 4K | fps/draw-result, gpu-lane-result |
| 10-19 | async lane, batched staging | 5.0 → 5.1 | 4K | STATE-history |
| 10-20..24 | merge19 fixes, merge21 fix, committer items, lease skip | ~5.5-7.0 | 4K | STATE-history |
| 10-25..27 | owned family, merge24 (A-B-A 7.87 / 7.75 / 7.92) | **7.9** | 4K | STATE-log 10-24..27 |
| 10-29 | 1080p relinker patch | 8.46 | 1080p | S5-1080 |
| 10-30 | resident staged ranges (hritikutekar) | 8.74 | 1080p | H-adopt2 |
| 11-02 | d0-long-1080 (yotei/best 6e85d0c493) | 9.25 | 1080p | STATE |
| 11-03 | yotei/best 4565a51dc4 / 72af1e5ea23 | ~10-10.6 | 1080p | STATE |
| 11-08 | unchanged (v3 levers all default off) | ~9.5-10.4 (A arms) | 1080p | fm2, v3-exec |
| — | **KytyPS5-Yotei on this machine** | **48.1** | 1080p | kyty-prof |

Caveats:
- 10-25 ~04:45: the user switched Windows to "Best performance", so absolute numbers before and after are not
  comparable.
- Separate runs drift up to 22-25 %. Every claim above 2 % in this ledger is from an in-run A/B or an A-B-A.

---

## 1. Bring-up and PLAN v1 (to 10-27): levers on the committer path

### 1.1 Old fast base (10-09, opening cinematic, s/frame)

| lever | result | exact | status, reason |
|---|---|---|---|
| remove the 6144 MiB host-import cap | 120.5 → 5.6 s/frame | yes | kept ("never 6144"): refused imports became 34k heap mirrors and lock convoys |
| Graphics Mode 2 preset | +9 % slower | — | rejected; the save's mode overrides it |
| riders (log gating, sort, O(log N) registry) | −25 % s/frame | yes | kept |
| texture-cache thrash fix (#1865, #1971, budget) | −8 %; frames 31-70 −36 % | yes | kept |
| image-table reuse across frames | slower (4.31 vs 3.78) | yes | reverted: stale premise |
| GPU-side indirect args (old base) | +3 % slower | yes | reverted: queue 0 waits elsewhere for the same GPU work |
| #2096 / #1905 / #1961 | slower / neutral / large loss | yes | reverted |
| A2+A3 address-space overlay | calm frames −25 % | **no** (split frames) | reverted; the A3-only follow-up was never done |
| S1 unlocked prewait | −6.3 % | yes | kept (presenter lock hold 211 → 22 ms) |
| batch 2 (page-precise compare, scratch blocks, VirtualQuery memo) | never measured | — | the project moved to main |

### 1.2 Main tree, loading and compile stalls (10-11..10-17)
These were not world fps, but they made the world reachable:
- offline giant-pipeline precompile (20-37 min compiles gone);
- parallel prepare (~14k shaders in ~6 min);
- #6168 + #6060 write tracking (guest compares 2.9 TB → 13.6 GB; loading 1.6× faster);
- spanned wave #5405 (−39..−45 % words);
- B2' one V# bounds check (−53 % words);
- **bindless images** (table resolve 40-58 s → 31-60 ms/frame; loading 13.8 → 1.0 s/frame);
- mapfast (#6321, ~2× presents after the big table);
- the introfix pipeline-cache-save stall;
- live bindless budget (loading 1.2 → 0.5 s/frame);
- runtime-format buffer loads (cinematic skinning 206-303 → 37-64 ms);
- word3 specialization (~70 programs compiled twice).

Dropped or parked:
- #6416 (doubled NVIDIA compile time);
- C3 batched barriers and C4 WAIT_REG_MEM on the GPU timeline (noise; C4 had a stale-read bug).

### 1.3 In-world fps (10-17..10-27, 4K)

| lever | result | exact | status, reason |
|---|---|---|---|
| drawcpu r1 block-level revalidation | 2.35 → 3.00 (+28 %, separate runs) | yes | kept |
| drv5 PR set (#6617, #6566, #7044, #6763) | 2.97 vs 2.95 | yes | no gain |
| drawkey (draw-cache key fixes) | lookup 13.2 → 11.6 µs; presents unchanged | yes | the draw cache has 0 hits in-world (keys carry per-draw user data) |
| fps-draw: walk-refused fix + #6187/#6192 + #7183 | 2.97 → 3.89-3.94 (+31 %); texhash 30-36 GB → MiB per 10 s | yes | kept |
| fps-serial (#6632, #6631, #7134, #6501, PIPELINE_INDIRECT, HTILE GPU expansion) | trace p50 −13 %; presents "not proven" | yes | kept |
| APS5_STAGING_DIFF | GPU −104 ms/present; presents unchanged (CPU-bound) | yes | on |
| epoch 5 (resolve memo, fast snapshot, block-wise capture) | bench miss 49 → 20 µs; world fps unchanged, every thread less busy | yes | kept; the frame is latency-bound |
| drop APS5_PIPELINE_REQUESTS from measured runs | p50 −12 % | — | rule: never in measured runs |
| S1 preparer thread, two-phase build | committer 316 → 269 µs; HTILE waits → 0; frame unchanged (the wait moved) | yes | kept |
| gpuind (GPU-side draw params for GPU-written records) | p50 −16 %; all-GPU variant slower | yes | kept |
| **device-local BDA buffers (GPU 1b)** | cloth 414 → 189 ms; GPU busy −200..−300 ms; presents 4.3-4.5 → 4.6-4.9 | yes | kept |
| dispatch overlap v1/v2 | frame within noise (each staged copy-in barrier is a wait-for-idle on NVIDIA) | yes | on after guard fixes |
| f32 v_fma_mix (0x141d55d00) | −34..−38 % on that pass | yes | kept |
| **async compute lane** (0x51 on its own compute-only queue) | **+8-13 %** | yes | kept (`APS5_ASYNC_LANE=51`) |
| batched staging | +9 % (5.12 vs 4.68) | yes | kept |
| merge19 → committer fixes (#8911-#8913) | merge −19 %, fixes +18 % | yes | kept |
| merge21 → image-slab restriction | merge −44 %, fix +11 % | yes | kept |
| #9176 range check | n.s., GPU +43 ms | yes | on for correctness |
| #6188 draw-ahead | n.s.; capture −51 µs but room wait +38 µs | yes | off |
| address-table templates | unsafe oracle −47 % committer; exact stage 1 +5 % (< +8 % gate) | stage 1 yes | full design no-go |
| read-only lease skip (#10382) | +2.8..+8.5 % | yes | kept |
| hritikutekar committer batches 1 and 2 | +2.5..+4 %, +2.8 % | yes | kept |
| pipelined dispatch hits+misses | −4 % (misses moved onto the committer); hits only ~+3 % | yes | hits only |
| #9400 writer presync | −18 % | yes | off |
| owned proofs v1-v3 (fault-based) | +3 / +3.8 / **+8.5 %** | yes | **removed 10-26**: guest-store faults break the SysV red zone (upstream #5596/#9181) |
| owned proofs via write watch / epoch answers | neutral; whole owned family +6.4 % | yes | kept |
| flush lane wait | +3.3 % | yes | kept |
| resident copy-in skip | v1 −70 % (texture-cache thrash); v2 −2 % while saving 55 ms GPU | yes | parked "for when GPU-bound" |
| compute template budget | **harmful** (2.0 vs 6.4; pinned textures, VRAM 8 → 14.4 GB) | yes | dropped |
| stage A' / D1 / D2 / P1 / P3 / label pre-wait / dispatch admission / shape keys | neutral to +1.5 % | yes | parked for good |
| mimalloc | committer heap share 2.0-2.6 % < 3 % gate | — | parked (queue-0 share unmeasured; hritikutekar's jemalloc +14.5 % was on Linux) |
| LLD linker | unsupported relocation | — | dropped |

### 1.4 Profiling facts from this era
- **10-17:** ~2,100 draws/frame at ~0.7 ms each on queue 0; GPU 19 % busy; DrawPipeline 46 % in ContentHash::Of.
- **10-18:**
  - ~3,200 render passes per ~9,700 draws/10 s (every heap draw broke the pass);
  - committer: template hit 298 µs, BDA build 1,073 µs;
  - cloth family 508 of 937 ms of programs; 0x80001c0d00 ~7 ms per dispatch on one SM (native estimate 0.1-0.4 ms);
  - staging ~0.8 GB each way per present, only ~1 % of copied-back bytes change.
- **Nsight:** any transfer or barrier between dispatches is a wait-for-idle on NVIDIA (8 dispatches 0.80 vs
  6.33 ms).
- **10-25:**
  - ~95 µs per draw on queue 0 plus ~210 µs on the committer, against 2-5 µs natively;
  - GPU 663 ms timed per present (programs 372 + classes 291), wall ~540 ms.
- **Ownership probe:** 162 of 182 touches of GPU-written memory come from our own host code.

### 1.5 Lessons this era wrote down
- Compare only the same build in the same session (the same code read 3.78 vs 4.92 morning vs afternoon), and check
  frame dumps for every perf change.
- **Moving a wait does not remove it:** GPU indirect args, S1, epoch 5.
- **Moving work onto the committer hurts or is neutral:** dispatch miss items, P3, draw-ahead.
- **Anything resident in VRAM competes with the texture cache:** resident v1, the template budget, image slabs.
- **Upstream merges regressed speed** (−19 %, −44 %), so every merge needs an A-B-A.
- **Diagnostics cost speed:** PIPELINE_REQUESTS −12 %, MEM_STATS, PROFILE_GPU (serializes), tracing +35-40 %.

---

## 2. PLAN v2 (10-27 to 11-03): the "new path" beside the old one

**What it was.** A second driver path behind `APS5_NEW_PATH=1`:
- S1: guest memory resident on the GPU;
- S2: every command resolved on the GPU (twins);
- S3: no global lock;
- S4: near-native translation;
- S5: 1080p.

Exactness gate: replay rows byte-identical to the old path, T1 MATCH. Budget: 30 fps, with 5 fps at day 28 and 10 at
day 35 (PLAN.md §1-§4).

**Speed over the period:** 7.9 (4K, 10-27) → 8.46 (1080p, 10-29) → 9.25 (d0-long-1080, 11-02) → ~10.6 (yb-combo,
11-03). All of it came from adoptions and old-path switches. **None came from the new path.**

| lane / lever | what | result | exact | status, reason |
|---|---|---|---|---|
| S0 harness (D/R/T/H 1-13) | replay rows, census, captures | byte-identical only up to frame-1 submit 10; no checkpoint with draws is byte-reproducible | — | infrastructure; draw exactness only ever T1-checked |
| S1 residency (R-1..R-20, R-copyin, R-written) | an earlier residency core [redacted] | world run hung (deadlock), then copy-in 55.6 GB for 76.6 MiB resident; **no command ever read a mirror in-world** | rows partly | never exited; F4 refuses resident pages |
| S2 twins (D-emit, D-map, D-cost, D-c1, D-draws, D-draws2) | GPU-resolved dispatches and draws | share 0-8 %; each A/B within ±0.2 presents (n.s.) | yes | the share limits the gain; every twin kept the old build, lease and capture |
| D-combo | all S2 pieces together | **new path −8 %**, GPU +92 ms/frame, committer +33 µs/commit | yes | slower |
| NP-overhead (11-03) | cost with nothing routed | not slower; draws routed +4.0 %; dispatch twins +2.7 % GPU | yes | "cost-free, not faster"; routing stopped |
| D-ahead | twin pipelines built ahead | 337 s in-frame hitch → 0 | yes | kept |
| S3 pair: DEFER_LABEL_READBACK + RESOURCE_CACHE_LARGE | label read-back deferral; bigger resource cache | **+6.5 ± 1.4 %** (s3-ab2), +4.8 ± 1.8 % (s3-ab3); each alone n.s. | T1 | **taken** into yotei/best 11-03 |
| S3 D1 dispatch cache 8k → 64k | | +0.2 % | yes | dropped: per-dispatch ring pointers make the keys unique |
| S3b address-free dispatch keys | | −2.5 % (4.2 SE) | yes | committer −10 µs/commit, but syncwait +57 ms/10 s |
| S3 multi-stream recorder (D5) | no global mutex | never built | — | design at rev 4 at the pause |
| Epoch 1 (SV 814a) | recompiler content | A-B-A −1.5 % n.s. | yes | no effect |
| T-4/5/6 one lane per invocation (cloth) | | bench −11..−27 % | **no** (v_readlane 63 wraps on 32-lane hosts) | void |
| S4-cloth: one lane, multi-wave | | slower (9.07 vs 7.38 ms) | no | parked |
| S4-cloth2: hardware f32 flush; NaN tests; GPU-checked overlap | | flush −4..−8 % bench (inexact); NaN 0 ms; overlap 4.1× on 8 copies | flush no | reverted / dropped |
| S4-cloth3 → s4e (11-04): exact cloth overlap with claim map | | 0 conflicts in 160 members; **game −8.7 %**, GPU +78 ms | yes | claim/redo cost exceeds the overlap |
| S5 1080p relinker patch | DRS steps and output at 1080p | **+9.9 %**; GPU 4K:1080p = 1.12 | own ref `t1-ref-1080` | **kept**, public default 10-30 |
| H-adopt: hritikutekar's 9 perf commits + TheBlenDed | | +3.5 % (3.9 SE) | yes | taken |
| H-adopt2: hritikutekar 25574c7776, resident staged ranges | | **+8.4 %** (4.9 SE); copy-in 9.3 → 2.5 GiB/10 s; GPU −170 ms → frame −114 ms | yes | **taken** |
| Adopt (11-03): fork PR #6 VRAM gate, preparer commits, PR #4 | | PR #6 at default **−17 %**; preparer +2.9 %; PR #4 0.0 %; A-B-A adopt2 +4.3 % | yes | set taken, PR #6 gate default off |
| Swiftlyx switches (ysw-ab2/ab3) | pinning, indirect label args, read-only BDA, DISPATCH_MISSES, DEFER_LABEL_STORE | ab2 +3.8 % did not reproduce in ab3 (all flat) | yes | nothing taken |
| G-copy (10-30) | barrier merge; device forwarding | +2.1 % n.s.; +1.2 % n.s. | yes | ~150 of ~14,000 barriers; 2 MiB forwarded |
| G-occupancy | the 10-11 KB shared memory | NVIDIA's register spill; driver's choice is best | — | no fix on our side |
| Q0-par | parallel queue-0 decode | bound ≤ 11 ms/frame | — | stopped at design |
| D-lanes | Linux lane-open-wait fix | A-B-A flat | yes | stability only |
| v1 day-1 batch (chain49) | affinity mask, spin 0, priority high, PIPELINED_DRAWS 256, mimalloc | void (14 % session drift), spin −14 %, priority crashed, depth +1.3 %, mimalloc parked | — | dropped |
| merge25 / merge27 (upstream main 10-29 / 11-03) | | merge25 cold T1 DIFFERS; **merge27 −52 %** (4.73 vs 9.7-9.9) | merge27 T1 MATCH | parked; suspect main's recompiler/materialization; never bisected |

Profiling facts from this era (1080p):
- **GPU idle 45 %** (568 ms of a ~1,249 ms frame). Cloth runs on 1-2 of 48 SMs for ~166 ms with nothing else running
  (G-trace).
- Programs 341 ms/present (cloth 180.5), draws 141, copies 219 (G-busy). ~12,000 barriers/present. Real occupancy
  ~611 ms. A GPU cut reaches the frame at ~0.67:1.
- Idle attribution (G-gaps): committer recording 357, queue-0 dispatch path 101, the 0x20 frame-start hold 38, GPU
  mutex ~39 ms/frame.
- **95 % of CPU reads of GPU-written data come from our own driver** (R-4).
- On NVIDIA/Windows, every device-local allocation also adds the same bytes to process private commit (R-4).

---

## 3. PLAN v3 (11-04 to 11-08): Kyty-informed rewrite in pieces

### 3.1 P0 ceilings and the 11-04 G-lanes (replay/bench ablations inside our design)

| item | result | source |
|---|---|---|
| real GPU occupancy | ~560 ms (136 ms of async overlap was double-counted) | P0-gpu |
| H-PCIe | not bandwidth: PCIe reads 729 MB/frame are the copy classes. BDA table → device copies of the imports (inexact): programs −14..−23 %; ~40-80 ms in-game [E] | P0-gpu |
| staging ceiling M2+M3 | ~92-97 ms/present of class time; the busy span falls less; net ~30-60 ms [E] | P0-mem E1 |
| publish (M5) | only 2-6 % of published bytes have a bound reader; no clean ceiling | P0-mem E4 |
| M4 lease ownership | writable lease = whole heaps (~10 GB) → ~90k faults/frame; **no-go** | P0-mem E5 |
| ReBAR | present; CPU reads of BAR 0.09 GiB/s (downloads must be GPU copies); CPU writes 8.4-8.9 GiB/s | P0-mem E6 |
| translation, all Kyty-style lowerings | cloth 1.18-1.24×, mostly from native float (inexact); exact items 0-5 % | P0-shader |
| flat BDA page table | ~0; a net loss inside cloth | P0-shader E3 |
| CPU what-if (C1-C5) | no single component reaches the GPU; all together ~2×; C5's CPU part ~3 µs/draw | P0-cpu |
| replay committer with proofs on | 1,100-1,650 µs/draw (VirtualQuery path) vs the game's 160-170; proofs off ≈ the game | P0-cpu |
| **P0 verdict** | exact gains add up to ~2-3×; the 3× bar not met | P0-gonogo |
| G-barrier | ~2 µs per barrier, ~7 ms/frame for all dispatch barriers | G-barrier |
| G-bda | BDA lookup 2-9 % of cloth; range memo −2..−7 % on cloth, 0.4-1.1 % of frame | G-bda |
| G-ballot | one-barrier ballot ~1 % of frame; =1 costs occupancy; 4K T1 DIFFERS unresolved | G-ballot |
| G-copy | own-shadow skip ~28 ms/present predicted; not exact | G-copy |
| C-commit | template hit 104 µs, BDA build 485 µs (31 % of committer busy), miss 1.7 ms; borrow +2.4 % | C-commit |
| A1 indirect draws join open pass | passes −16 %, ~0 ms/frame | A1-indirect |
| P0b-draws | pass breaks ~75 µs each, 86 % of passes hold one draw, breaks come from bindless publishes | P0b-draws |
| P0b-reach | real BDA reach 21-28 MiB read + 1.4-1.8 MiB written per frame vs 9.3 GB lease; static analysis covers < 1 % | P0b-reach |
| A2 reach-predicted device copies | programs +5-8 % **slower** | A2-reach |
| B-fast (bundled inexact fast mode) | **−35 %**: PROOFS forced texture hashing (~45 GB/10 s); CLONE hung the GPU, STAGING/PUBLISH=2 gave garbage | B-fast |

### 3.2 P1 lanes (exact unless marked; all in-run A/Bs at 1080p)

| lane | lever (switch) | result | status, reason |
|---|---|---|---|
| exec | C1-R recording worker (`APS5_V3E_RECORD_WORKER`) | milestone A-B-A +1.0 % (A arms 4.5 % apart) | off: single CPU cuts are absorbed |
| exec | worker pin / no-spin | −3 % / −3 % | off |
| exec | C3 byte log (memcmp at hit) | saves ≤ 2 µs/draw (lookup is 97 % content proof) | adoption not built |
| exec | C4 image flags | committer −11 µs/draw in game; presents pinned ~10 | in the baseline set; the committer isn't the only limit |
| exec | C4 storage/target flags | only 105-160 of ~9-10k refreshes clean; B arm 3× slower from a tracker bug | off |
| exec | C4 buffer flags | net −4.8 µs/draw; caused an AV (race, fixed) | off: a flag check costs as much as the cheap path |
| exec | C4 input flags, texture room | vertex −4.7 µs/draw; committer −10 µs/draw; presents flat | in the baseline set |
| exec | DISPATCH_MISSES as pipeline items | ±0 %; drains −34 %, committer +30 ms | off: work moved to the committer |
| exec | pipeline depth 256 / 128 | −3.3 % / −2.7 % | off: room wait → drain wait → guest-link idle |
| exec | draw-proof memo | −2.5 ± 1.9 % | off: what it replaced was already cheap |
| exec | lock split (i), per-leaf locks | −3.0 % / −2.5 % | superseded |
| exec | lock split (i), stage holds | +0.24 ± 1.46 % | base for (ii) |
| exec | lock split (ii) prepare proof | −22.8 % (lock contention) → −18.6 % (tracker waits) → **−1.84 ± 0.99 %** after the fix; 41 % served | off; **user stopped the lock split 11-08** |
| exec | lookup census (§39) | lookup 38.8 µs/draw; ceiling +4-10 % now, +9-20 % after 6-9 weeks | — |
| mem | writer device copies (WREACH) | GPU +12 %, programs not faster, not exact | gate failed |
| mem | homes (resident ranges skip copy-ins) | bytes −15..−43 %, class ms unchanged; not exact | stopped: copies bound by per-use commands, not bytes |
| mem | resident no-copies ceiling | r18 −69 ms GPU busy (bar 100) | resident buffers dropped |
| mem | dirty tracker (page protection) | refusals 86 % → 7 %; ~3.6 ms/frame arm cost; 0 violations | delivered to exec |
| mem | address-publish skip ceiling | GPU −24 ms, presents +1.2 % | exact build held: frame CPU-bound |
| mem | import-open fix | title 22 → 160 presents/10 s with the tracker on | merged |
| mem | storage tokens | −2.2 % | off |
| mem | probe memo | never engaged | parked |
| waits | land-ahead | −2.5 ± 3.3 % n.s. | off |
| waits | fence known; CPU writes unlocked | −0.9 %; −3.1 % | parked |
| trans | efbg: V# shape specialisation, one load, branchless loads/stores, SRT guards once | **cloth 0x80001c0d00 −17.5 % instructions** (exact; T1 1080 + 4K); other cloth −6..−14 % | merged in `local/v3` 9154aded9ee, switches off; fps never measured (cold caches; frame CPU-bound) |
| trans | native float; all inexact items | −19 %; 1.49× | not exact |
| fm2 (11-08) | trust-the-cache, SNAP, TARGET memo (`APS5_FM2*`, inexact) | clean variant +0.2 %; TRUST across 16 presents **+2.0 % with cyan garbage frames** | stopped by the stop rule (< +20 %) |

### 3.3 Where the ~1 s frame goes (the best model, from v3-exec)
- **The critical path is serial:**
  1. The queue-0 worker processes frame N (~0.8-0.97 s).
  2. Flip N.
  3. ~140-145 ms of guest job work (10 job threads).
  4. The 0x52 submit.
  5. ~25 ms to its EOP.
  6. Frame N+1 is submitted.
- **One GPU mutex is held ~730 of ~985 ms.** The committer holds 484-488, the queue-0 worker ~100, the async queues
  ~115 and the presenter ~13.
- **Committer, 154 µs per draw:**
  - resources ~53 (made texture snapshots; 97 % of them are capacity misses);
  - record ~35-42 (descriptor sets ~23);
  - lookup ~31-39 (content proof 32);
  - readTarget 12, vertex 9-14, the rest ~25.
- **GPU:** busy 620-870 ms with 350-565 ms of idle gaps. The GPU is not the limiter today.
  - cloth 0x80001c0d00 alone ~150 ms/frame;
  - draws ~142 ms (742 passes);
  - copy classes ~177 ms.
- **Cloth shader:** the bound-buffer V# path (run-time stride/swizzle, from upstream 573ba3b2935) is 42.5 % of the
  dynamic SPIR-V ops (v3-trans §2).
- **Why the lock split capped at +4-10 %:** 34 % of draws can't be vouched for. RefusedEpoch is 18.5 % (the refusal
  cache is sticky per mapping epoch), and BDA leases are 15.8 % (whole-heap leases with no registry serial). The
  served draws skipped only the already-cheap part, and validation costs ~5 µs back.

---

## 4. KytyPS5-Yotei measured on this machine (11-05/06)

| item | Kyty v0.1.4 here | ours | source |
|---|---|---|---|
| presents/10 s, untraced, river village 1080p | **48.1** | 9.3-10.6 | kyty-prof w2 |
| draws / dispatches per frame | 3,450-3,530 / 2,720-2,760 | ~3,100-3,470 / ~2,800 | kyty-prof |
| CPU per draw on the ordered thread | 26 µs | 154-170 µs | kyty-prof w3 |
| GPU active per frame | ~60 ms | ~580 ms | kyty-prof w2 |
| cloth / other programs / draws / copies | 52 / ~28 / ≤23 / ~0 ms | 158-279 / 180-200 / 142-199 / 178-214 ms | kyty-prof |
| cloth 0x80001c0d00 per dispatch | 555-596 µs; 0.6-0.8 M warp instructions | 2,915 µs; ~3.1 M | kyty-cloth §7 |
| PCIe reads per frame | 17 MB | 729 MB | kyty-prof |
| inexact shortcuts' share of their lead | ≤ 10-15 % [E] | — | kyty-prof §5 |

Notes:
- **1.008 needs `KYTY_PERF_DIRECT_RANGES=0`.** Every release asserts at present 2 without it.
- **Kyty is not at its best here:**
  - CPU reads of ReBAR cost ~36 ms/frame;
  - frame-start waits cost 65 ms/frame (8-9 on their PC);
  - its 26 µs/draw is ×7-9 its documented 3-4 µs, and only ×1.5-2 of that is CPU speed.
- **Their history:** their base (upstream KytyPS5) was already at 4.63 fps at 4K before any Yōtei work. Their own
  three weeks gave ×3 (kytyplan-inventory §1).

---

## 4b. What others measured (discussion #675, read in full 11-08)

These are other people's numbers on their own hardware. They are not our A/Bs. Post ids refer to the 11-08 dump of
the thread (`C<n>` = comment, `.R<n>` = reply).

| who, date | setup | result | source |
|---|---|---|---|
| hritikutekar, 11-07 | Linux, RX 9070 XT (RADV + ACO patch), Ryzen 7 5700X, PGO build, GPU clock forced high, branch `yotei/linux-best-u2` 3de6af7a4f | **world 3.35 fps (33.5/10 s), title menu 20.4 fps**; 28 min stable | C100 |
| hritikutekar, 11-07 | same, A/Bs | table formats 1989128f55 world **+8.8 %**; hardware NaN on AMD 866e886803 +5.3 %; two-level BDA 7fe054cea1 +3.8 %; plain float ops on AMD 1165b6e883 +2.0 % (not bit-exact); NO_WRITTEN_SHADOW world +3.1 %, menu +6.6 %; CPU EPP performance +2.6 % | C100 |
| Pipodi, 11-07 | Windows, RTX 4090, i9-13900K | **NO_WRITTEN_SHADOW on NVIDIA: a net loss** (cinematic f1650→2000 261 → 301 s). Staging-in 35 → 9 ms, but draws 97 → 131 ms per present | C100.R0 |
| Pipodi, 11-07 | same | the cinematic's ~180 ms/frame host wait is a real GPU-serial dependency: an earlier release would save ~1.2 ms/frame. The lever is GPU time per batch | C100.R1 |
| Pipodi, 11-07 | same | all-cold time to the kanji prompt 3,020 → 2,907 s (native PS5 ~248 s) | C100.R0 |
| TheBlenDed, 11-07 | Windows, i5-11600K, RTX 3080 10 GB, **game v2.002** | village flyover 0.2-0.3 fps. Queue-0 room wait (committer 99 % busy), and sampled-snapshot handling is the lead | C101 |

| hritikutekar, 10-27 | same AMD setup | world 1.95 fps with PGO (1.8 default). **The title keeps one frame in flight** (it polls an end-of-frame fence): ~430 ms queue-0 recording, ~72 ms GPU tail, ~19 ms title. GCC PGO +5.5 %, pinning +2.1-3.3 % (Linux) | C71 |
| Swiftlyx, 11-01 | Windows, 9800X3D, RTX 5070 Ti, **game v2.002**, yotei/best 6e85d0c49 | world 18.5 → 20.0/10 s. Windows pinning +4.5 %; hritikutekar's committer trio +3.8 %; 8106f78f5 **−28 %**; MinGW native + PGO **+2.5 %** (LTO does not link under MinGW); DEFER_LABEL_STORE +5.5-6.3 %; OBS/RTSS layers off +6 % | C90 |
| shahulmubeen-a, 11-04 | Windows, 7950X, RTX 4070 Ti SUPER | merging yotei/best: world 17.3-18.8 → 22.6-24.0/10 s. Per switch: owned proofs −9.6 % off; GPU-ordered write-backs −11.8 % off; **RESOURCE_CACHE_LARGE −19.9 %** (pins textures past VRAM; it is in our README env); 94-98 % of copied-in bytes unchanged. Their own survey (including Kyty) concludes "persistent device-resident mirror with dirty tracking" | C97 |
| TheBlenDed, 11-02 | Windows, RTX 3080 10 GB | NO_WRITTEN_SHADOW + NO_ATOMIC_STAGING: hand scene 0.4 → 1.5 fps on the 10 GB card (VRAM-bound); queue-0 worker 92-96 % busy | C91, C91.R1 |
| Dextroid17, 10-30..11-07 | Linux, RTX 3080 Ti 12 GB, v1.008 | 6e85d0c493 +15 % over cc5d69fca6; NO_RESIDENT_RANGES −20.6 %; main waits ~3 s/frame in memory_wait. ForgetSerial on main erases 50 of 60 bound resident entries per 10 s (a lead for merge27's −52 %). #15780: V#s from user-data tables fall back to read_bda_span (24 % of program time on the fork) | C87.R1, C98, C99, C99.R2 |
| danielcadev, 10-27 | RX 9060 XT, RADV | world 0.99 fps; texture budget 6 vs 4 GiB +69.6 %; buffer-pool slots +27.8 % | C70 |
| Pipodi, 11-02 | RTX 4090 | **a single Vulkan buffer above 2 GiB returned wrong values** | C92 |
| pre-10-27 posts | various | Nobody mentions Kyty or a backend replacement. FardinHaque70 e911fbf40 intro +27 % (Linux NVIDIA); hritikutekar jemalloc +14.5 %; texture-cache frame age 65a4290fb +60-70 % when over budget | C40.R0, C44.R0, C58 |

Upstream and community state, from the thread:
- **Upstream direction.** oneandonlydean accepted our upstream stack order on 10-27 ("We'll line up review and CI on our
  side in that order", C72.R0). PLAN.md still says "no answer". This is Codex's lane.
- **Nobody else is porting Kyty or building a new GPU backend.**
- **Offers to help:**
  - Swiftlyx: testing;
  - Dextroid17: testing and bisects;
  - Pipodi: all-cold runs on a 4090;
  - TheBlenDed: 10 GB-card tests;
  - shahulmubeen-a: their write-ups.
- **Open questions to us (unanswered):**
  - C71: hritikutekar, 8106 queue-0 wait;
  - C79: TheBlenDed, captures for 0x8000173b00 and 0x800014e300;
  - C80/C84: Pipodi, a cold regform run on the 3080 Laptop, and a review of fork PR #3;
  - C88: Pipodi, the bindless async-writer PR;
  - C90.R5: Swiftlyx, when yotei/best moves;
  - C97: shahulmubeen-a, fork PRs #7 and #8, and RESOURCE_CACHE_LARGE −19.9 %;
  - C99: Dextroid17, the bisect offer.

What this means for v4:
- Of the four #675 ideas in HANDOFF-2026-11-08 §4:
  - NO_WRITTEN_SHADOW is already measured negative on NVIDIA/Windows by Pipodi.
  - Hardware NaN and plain float are AMD-only.
  - PGO and table formats stay as phase 4 items for the new backend.
- Every AnyPS5 tree on NVIDIA/Windows reports the same committer and snapshot bottleneck we measured.

---

## 5. Recurring lessons (each has several rows above)
1. **The frame is co-limited.** CPU cuts turned into GPU or sync waits, and GPU cuts into idle:
   - DISPATCH_MISSES;
   - depth 256;
   - S3b;
   - C4 flags;
   - the address-publish skip;
   - fm2;
   - the S4 overlap.
2. **Building beside the old path keeps its costs:** S2 twins, D-combo, homes.
3. **Ablation ceilings inside our design under-predict a different design:** P0-shader 1.24× vs Kyty ×5 per cloth
   dispatch; P0-mem ±0 for device copies vs ~180-210 ms of copy work Kyty does not have.
4. **"Byte-identical to the old path" blocked the structural changes:** cloth overlap, barrier model, publish and copy
   skips, trust-the-cache, whole-heap leases.
5. **Session drift and noise voided several A-B-As** (chain49, ysw-ab2). Small-effect measurement cost days per lever.
6. **Upstream merges cost speed:** merge19 −19 %, merge21 −44 %, merge27 −52 %. The first two were fixed; merge27's
   cause was never found.
7. **Page-protection tracking of guest stores was removed once already** (10-26, the SysV red zone). Any design that
   needs it must solve that first. KytyPS5-Yotei ships a static red-zone patcher and a `--redzone` option for it.
