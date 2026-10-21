# Full-PDF audit: SceneEval and SceneSmith

Audit date: 2026-10-20. Scope: two primary papers already invoked in the previous Table 1 recommendation. Read the locally downloaded full PDFs, main evaluation tables, metric definitions, relevant supplementary protocols, and limitations; visually inspected SceneEval p.8 and SceneSmith pp.8, 38, 40. This is a targeted source verification, not an exhaustive review of indoor scene synthesis. No new empirical measurements were made.

## Sources and evidentiary weight

| Source | Version and primary URL | Local PDF | Evidence tier and weight |
|---|---|---|---|
| SceneEval | arXiv:2503.14756v3, 2026-03-07, WACV 2026 Oral camera-ready; https://arxiv.org/pdf/2503.14756v3 | `docs/research/20261020-worldedge-table1/papers/sceneeval.pdf`, 24 pp. | Peer-reviewed primary benchmark paper. High weight for what its metrics mean; moderate weight for construct validity/generalizability. 500 annotated descriptions, six evaluated methods. Small external human validation, culturally bounded author annotations, VLM dependence. |
| SceneSmith | ICML 2026, PMLR 306, official proceedings PDF; https://raw.githubusercontent.com/mlresearch/v306/main/assets/pfaff26a/pfaff26a.pdf ; landing page https://proceedings.mlr.press/v306/pfaff26a.html | `docs/research/20261020-worldedge-table1/papers/scenesmith.pdf`, 69 pp. | Peer-reviewed primary system paper. High weight for its reported protocols; moderate weight for system comparison; limited weight for broadly generalizable downstream-agent conclusions (100 scenes, four task templates, one constructed controller and a degraded variant). SceneEval-derived metric families are reused evidence, not independent validation of those constructs. |

## SceneEval: actual tables and definitions

**Its Table 1 is dataset statistics, not a main performance table** (p.3). Table 2 is annotation schema (p.4). Main generation comparison is **Table 4, p.8**. It groups **Fidelity**, **Plausibility**, **Resource**, with a final CLIP similarity column.

| Native column | Direction | Definition/unit and aggregation | Exact anchor | Meaning for WorldEdge |
|---|---:|---|---|---|
| CNT | up | Fraction of annotated object-count requirements satisfied in a scene, after VLM object/category mapping. Requirements include equality/inequality quantities, not raw generated object count. Table 4 averages scene scores. | §3.2–3.3, pp.4–5; schema App.A.1 p.12 | Preserve semantic inventory fidelity separately from richness/#Obj. |
| ATR | up | Satisfaction of annotated object attributes (color, material, size, style etc.); VLM receives front image and image with 170 cm human scale reference. Table averages scene scores. | §3.3 p.5 | Previous Task-Spec Pass failed to explicitly retain asset appearance/attribute fidelity. |
| OOR | up | Satisfaction of annotated object–object relationships; open-vocabulary relation mapped to 13 geometric relation types; all mapped relations required. Default positive threshold 0.5. | §3.3 p.5, App.B.2.1 p.14, B.3.1 p.15 | Call this OORel in our paper to avoid collision with FastFill's object-overlap-ratio abbreviation OOR. |
| OAR | up | Satisfaction of annotated object–architecture relationships (10 relation types, default threshold 0.5), not architecture correctness alone. | §3.3 p.5, B.3.2 pp.15–16 | Separate from OORel: wall/corner/room placement fidelity. |
| COL_ob% | down | Percentage of objects in a scene that participate in mesh collision with another object. | §3.4 p.5; Table 4 p.8 | This is object-level collision prevalence, not pairwise overlap area and not binary all-clear world yield. |
| COL_sc% | down | Scene-level collision column reported alongside object-level COL. The PDF does not separately spell out its formula/edge cases in §3.4; it should be implemented against official code before claiming exact replication. | Table 4 p.8 | Do not silently equate it to our GSR. |
| SUP | up | Fraction of objects geometrically supported. VLM assigns ground/object/wall/ceiling. Rays toward support direction seek contacts within 1 cm. Wall/ceiling: any contact suffices. Ground/object: gravity projection of object centroid must lie in convex hull of contacts. | §3.4 p.5; App.B.2.2 p.14 | Static geometric test, **not** a physics rollout stability measurement. |
| NAV | up | Largest connected free floor area / total free floor area from 2D occupancy and connected components. | §3.4 p.5 | Not task-target reachability, navigation SR, path efficiency, or articulated interaction. |
| ACC | up | For each object, free pixels / total functional-side area; VLM predicts functional sides; take best side if multiple. | §3.4 pp.5–6 | An unblocked functional region is not proof a robot can reach, open, or manipulate it. |
| OOB | down | Object is outside floor plan when fewer than 99% of sampled surface-point downward rays hit floor. Report fraction of objects out of bounds. | §3.4 p.6 | Necessary counterbalance to high NAV/low collision achieved by placing objects outside. |
| Mem_GB, Time_sec | descriptive/lower generally desirable | Peak GPU memory and generation time; table averages scenes. | Table 4 p.8; §4.1 p.6 | Report efficiency with hardware and output complexity; do not compare unmatched timing setups. |
| CLIP_sim | up | Long-CLIP-L rendering–text similarity comparator. | Table 4 p.8; §4.2 p.6 | Coarse comparator; does not replace detailed fidelity. |

Corpus/protocol: 500 descriptions = 100 manually authored + 400 LLM-generated and manually corrected (§3.1 pp.3–4); 150 easy, 200 medium, 150 hard. Ten room categories. All original methods retain their native asset sources. Non-Holodeck methods receive a 6 m × 6 m floor plan with walls; DiffuScene and InstructScene cannot condition on the floor mask and are flagged for affected metrics. GPT-4o-2024-08-06 supplies VLM components (§4.1 p.6). Metrics are averages over scenes, **not all-components-pass rates**. Missing-applicable-annotation handling, exact NAV resolution/inflation and COL_sc denominator details are not fully specified in the PDF, so read released code before exact implementation.

Validation: authors manually evaluated 500 generated scenes from 100 manual descriptions; CNT/ATR/OOR/OAR agreement 89.8/83.5/94.6/94.1%, Cohen kappas .75/.56/.72/.77. External human study: 10 participants, 125 scenes; agreement 89.5/89.9/91.7/88.1%, kappas .72/.57/.58/.56 (§4.2 p.6). These are evaluator agreement, not scene success. Public Qwen2.5-VL-7B substitution has lower agreement (App.C p.16), showing evaluator-model dependence. Human study interface is binary constraint annotation, not SceneSmith-style pairwise realism ranking (App.E p.17).

Key negative control in Table 4: LayoutGPT has NAV 99.99 and object COL 11.46, yet SUP 30.13 and OOB 72.25. Thus NAV plus collision alone can reward an unusable scene. **We need OOB and fidelity/support/stability beside access.** Table 5 on p.8 reproduces FID/KID/SCA from earlier papers; these are not a newly rerun common SceneEval-500 evaluation.

No robot action-policy completion table, articulated execution rate, persistence score, or all-checks readiness yield is evaluated in SceneEval.

## SceneSmith: actual tables and definitions

**Table 1 (p.8) is HUMAN evaluation**, not robotic task success: columns `Comparison, #Obj`, and separate Realism and Faithfulness blocks with `Win%, 95% CI, Cohen's h, Sig.`. **Table 2 (same p.8) is automated WORLD evaluation**: `Method, #Obj, CNT, ATR, OOR, OAR, ACC, NAV, COL, STB, OOB`. Robot policy comparison is **Table 12 in App.L.4, p.38**.

| Native metric | Definition/scope | Anchor | Required caveat |
|---|---|---|---|
| #Obj | Mean number of generated objects, with 95% CI | Tables 1/2 p.8 | Not count-constraint fidelity; not per-square-meter density. The paper uses it to contextualize complexity/clutter. More objects alone is not always better. |
| CNT, ATR, OOR, OAR, ACC, NAV, OOB | SceneEval metrics as above; score per scene, report mean ± scene-based t CI | §4.1 p.7, App.M.2 pp.39–40, App.O.1 p.42 | Reuses SceneEval constructs; do not claim independent confirmation or aggregate into a pass metric while retaining the same name. |
| COL | Fraction of objects participating in a collision with penetration depth **>1 mm**, using Drake signed distances on collision geometry | App.M.2.1 p.40 | Different from original SceneEval mesh collision, FastFill axis-aligned overlap, and any-collision scene failure. |
| STB | 5 s gravity simulation, 1 ms timestep; stable if displacement **<1 cm** and rotation **<0.1 rad**. Wall/ceiling fixtures welded; VLM support types used for all methods. | App.M.2.1 p.40 | Dynamics test, not SUP. Paper defines fixture welding and carpet exclusion but does not explicitly fully spell denominator treatment of welded fixtures; verify released implementation before reproducing. |
| Realism Win% | Pairwise preference of SceneSmith against comparator; forced choice, no tie | Table 1 p.8, App.N pp.40–42 | Human perceptual measure, not physical validity; primary metric omitted from prior proposal. |
| Faithfulness Win% | Pairwise preference for adherence to text; allows ties, excludes ties in reported win denominator | Same | Different denominator than Realism; not CNT/ATR and not task SR. |
| MPD | Mean penetration depth among colliding objects per scene; then scene average, mm | Table 17 p.44 | Conditional severity; pair it with COL because low incidence can hide deep failures. |
| MD, XD, MR | Mean displacement across non-welded objects per scene; max displacement per scene; mean rotation | Table 16 p.44 | Stability severity diagnostics; not required in compact main table. |
| Cost / Time | Per-scene API cost and elapsed generation time | Tables 18/19 p.45 | 25 concurrent scenes on 8 L40S GPUs; time is explicitly not optimized standalone scene latency. |

Protocol and thresholds matter: baselines originally lack physical metadata, so author adapts them with V-HACD (128 max hulls) and uniform density 1000 kg/m^3 to infer mass/inertia/CoM; carpets excluded (p.40). Native SceneSmith includes nonpenetration optimization and gravity settling during construction (App.H pp.34–35), and **removes fallen objects** (furniture tilt, floor penetration, support departure). Hence final-output physics quality is not raw uncorrected layout quality. For our experiment, document allowed cleanup, count object losses in semantic fidelity, and do not allow silent deletion of required targets. Requiring equal cleanup everywhere is one possible controlled experiment; native-system vs native-system is also legitimate if labeled.

Cohort: 210 prompts = 179 room + 31 house; room prompts consist of 94 SceneEval-100 + 50 type-diversity + 25 density + 10 theme. Remaining 6 SceneEval prompts are among 31 houses. Only Holodeck among the selected baselines supports house-level generation, so separate block (App.M.1.3 p.39). These are not the SceneEval-500 corpus and old printed scores cannot be copied into a common WorldEdge table.

Statistics: automatic tables mean ±95% t-distribution confidence interval, n=number of scenes (App.O.1 p.42). Human study 210 recruited, 205 complete participants, 15 comparisons each, 3,051 valid responses, USA Prolific; interactive 3D views plus rendered alternatives; method identity blinded; scene side/question order randomized; participant sees each prompt at most once (App.N pp.40–42). Human CIs Wilson; two-sided binomial tests vs .5 with BH FDR for 28 tests; Cohen h (p.40). Paper's non-overlap-of-CIs language is heuristic and should not substitute for a properly paired analysis in our proposed experiment. Human repeated-rater structure should also be accounted for in our inference.

Small internal numerical discrepancy: Table 2 p.8 vs Table 17 p.44 COL values differ for LayoutVLM curated (25.9 vs26.1), Objaverse (28.9 vs31.4), SceneWeaver (12.5 vs12.6). This does not change metric design but is another reason not to transplant reported numbers without rerunning a common protocol.

## SceneSmith robot evidence: what is actually measured

App.L pp.36–38 defines task-conditioned prompt generation: require necessary objects and non-goal initial state. Model-based KUKA iiwa14 on mobile base+lift; privileged segmented point clouds, antipodal grasps, IK, RRT-Connect, trajectory parameterization. Degraded variant removes gripper speed/acceleration limits and extra clearance. Evaluator combines state queries (poses, distances, support) with multiview renders; it is an agent judgment, not a fixed deterministic predicate (§3.5 pp.6–7; App.L.2 p.37).

| Task | Standard | Degraded | Denominator |
|---|---:|---:|---|
| Fruit from bowl to plate | 3/25 =12% | 3/25 =12% | 25 generated SceneSmith scenes |
| Coke can from shelf to table | 6/25 =24% | 4/25 =16% | 25 |
| Cup from floor to sink | 4/25 =16% | 3/25 =12% | 25 |
| Bottle from kitchen to living-room coffee table | 3/25 =12% | 2/25 =8% | 25 |
| Overall | 16/100 =16% | 12/100 =12% | 100 |

Exact source: Table 12, App.L.4 p.38. 300 evaluator labels = initial, standard final, degraded final for100 scenes; agreement with human299/300=99.7%. **99.7% is NOT robot SR.** Robot table compares controllers within SceneSmith-generated scenes; it is not evidence SceneSmith improves SR over Holodeck/HSM. The paper supplies no CI or matched-discordance analysis for the4pp policy gap, so describe it as a descriptive demonstration, not statistically established improvement. No published persistence/revisit or articulated-task aggregate SR here. Articulated cabinet teleoperation and zero-shot generalist rollout are qualitative Figure5 demonstrations (p.9), distinct from the quantitative4task study. Scene-generation attempt counts, discarded task scenes, and cross-generator all-request yield are not established by this table.

## Corrections to the earlier WorldEdge proposal

1. **Ready/Spec/Phys/Access/Interact/Persist were our design, not native published columns.** Keep any new measures explicitly labeled proposed and define them separately. In particular Ready is an engineering acceptance/yield construct, not a consensus scientific score supported by these papers.
2. **Human realism and fine-grained semantic/appearance fidelity were underweighted.** SceneSmith places human Realism/Faithfulness in its Table1; SceneEval and SceneSmith preserve CNT/ATR/OOR/OAR rather than one semantic pass. For an end-to-end scene-generation paper, these are major missing dimensions.
3. **Geometry, dynamics, and behavior must stay distinct.** SUP static support ≠ STB simulated equilibrium ≠ COL signed-distance penetration ≠ successful interaction. NAV/ACC proxies ≠ actual agent task SR. Fixed metadata presence ≠ execution capability.
4. **World complexity is necessary context.** Include #Obj (and task-required counts), because SceneSmith's denser worlds have lower NAV than several sparse baselines. Do not label fewer objects automatically better or count alone as a quality score.
5. **Persistence and articulated execution can be novel evaluation contributions**, but these two papers do not supply those protocols. A fixed executor across generators is a valid proposed experiment; this is not what SceneSmith's robot Table12 has already done.

## Minimal table recommendation grounded in these sources

Use native metrics, with grouped headers, instead of six unvalidated summary passes:

`World panel: Method | #Obj | CNT | ATR | OORel | OARel | COL | STB | ACC | OOB`

If width allows, retain NAV next to ACC for direct SceneSmith compatibility; otherwise NAV is first to move to the appendix because it is nearly saturated (95.9–99.7% among room methods in SceneSmith) and is not target-specific. Do not remove OOB: SceneEval demonstrates an explicit failure when it is ignored. Keep all four fidelity metrics in the detailed benchmark table even if main text must be shorter. Human Realism and Faithfulness warrant a separate paired user-study table or separate block, since their rows/denominators differ from the automatic metrics. Cost and time can be a compact adjacent table.

`Agent panel: fixed policy on matched task specifications; task SR + partial goal completion + efficiency, with task-family strata supplied by task-benchmark papers audited by other agents.`

Report coverage/failure accounting alongside this (attempted/materialized/executed), and make overall end-to-end SR an explicitly defined WorldEdge system metric. Persistence and interaction validation should be labeled new WorldEdge protocol/extensions. No reason to elevate the unvalidated all-checks Ready rate above the native measures by default.

Confidence: **high** for exact native columns and distinction among scores; **moderate** that this grouping is the best compact WorldEdge presentation (depends on pipeline's intended contribution); **insufficient** evidence that any particular world score predicts downstream policy success, or that these two papers validate a general persistent-world benchmark.
