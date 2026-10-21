# Independently verified research report

Coverage/integrity status: **complete**. Research scope: **diagnostic only**.

Object order and category-index assignment inherit reference footprint dimensions and pose tie-breaking from V1 canonicalization. Removing explicit size fields does not remove this target-derived information. These results do not establish performance for arbitrary object lists with unknown sizes.

The frozen SFT/DPO stage comparison remains a diagnostic comparison under that original input contract. Its inferential labels describe only the preregistered checker proxy on these cohorts; they do not establish general size-unknown deployment validity, causal benefits of cleaning alone, or physical mesh quality.

A target-independent input order requires a separately versioned data contract, rebuilt training/evaluation data and a new training protocol. Existing generated outputs and metrics are preserved; the scoring function and model recipe are not changed retrospectively.

This report is published only after independent CPU rescoring of all saved outputs with the original frozen scorer and interpreter, and verification of the DPO model training/merge lineage. Failure is reported as incomplete without inferential conclusions.

# FastFill V1.1 research synthesis

Status: complete. Verified full evaluations: 12/12.

No test-based selection, tuning, or automatic training. Final synthesis requires all 12 evaluations; provisional dev comparisons require all six dev evaluations.

Primary outcome: repaired zero-collision acceptance over all requests. Failures count as false. Paired bootstrap resamples whole (source, group) clusters, with 2,000 draws, seed 42, and 95% percentile intervals.

| Dev split | Comparison | Rooms / clusters | Difference | 95% interval | Evidence |
| --- | --- | ---: | ---: | --- | --- |
| dev_rooms | dpo1 − sft | 8167 / 3163 | 0.0149 | [0.0084, 0.0220] | supported |
| dev_rooms | final − dpo1 | 8167 / 3163 | 0.0176 | [0.0105, 0.0245] | supported |
| dev_constrained_rooms | dpo1 − sft | 8035 / 3150 | -0.0111 | [-0.0195, -0.0026] | regression |
| dev_constrained_rooms | final − dpo1 | 8035 / 3150 | 0.0122 | [0.0036, 0.0208] | supported |

Coverage/size guard warnings are descriptive; they do not redefine the primary evidence status or set a quality threshold.

- dev_rooms, dpo1: Candidate complete rate decreased; inspect failure types and coverage before further research.
- dev_rooms, dpo1: Candidate all-request size IoU decreased; inspect size behavior before further research.
- dev_rooms, final: Candidate complete rate decreased; inspect failure types and coverage before further research.
- dev_rooms, final: Candidate all-request size IoU decreased; inspect size behavior before further research.
- dev_constrained_rooms, dpo1: Candidate complete rate decreased; inspect failure types and coverage before further research.
- dev_constrained_rooms, dpo1: Candidate all-request size IoU decreased; inspect size behavior before further research.
- dev_constrained_rooms, final: Candidate complete rate decreased; inspect failure types and coverage before further research.
- dev_constrained_rooms, final: Candidate all-request size IoU decreased; inspect size behavior before further research.

## Descriptive metrics

Full population/source, raw/repaired, complete-rate, size-IoU, and constraint metrics are retained in research-summary.json. Test metrics are descriptive only. Common-complete-pair size IoU includes its selected denominator; failed requests score zero in all-request size IoU.

## Limitations

- These intervals describe four prespecified comparisons without multiplicity adjustment.
- The size metric uses reference dimensions and is not a category-level realism assessment.
- Repaired acceptance includes postprocessing; raw metrics are reported separately.
- Stage comparisons do not isolate the effect of cleaning; a matched old-data control is absent
- This report does not select a winner, tune on test data, or launch subsequent training.

## Pending or invalid evaluations
