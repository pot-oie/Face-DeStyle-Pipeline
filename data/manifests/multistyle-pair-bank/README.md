# Multistyle pair-bank source lists

These CSV files are lightweight, operator-visible curation records for the active reconstruction
pair-bank experiment. Paths resolve against the private `Face-DeStyle-Data` root. They are not a
public image release or a formal-v1 manifest.

`origami_hard_v2_selection.csv` records the independent review of the 30 generator-accepted hard
Origami candidates. Its paths are portable relative to the private
`extensions/origami_hard_pairs_v2` root. Exactly 28 rows are accepted; `origami-hard-v2-021` and
`origami-hard-v2-023` remain explicit composition-drift rejections.

Current roles:

| Style | Candidate | Holdout | Rejected |
|---|---:|---:|---:|
| 3D cartoon | 27 | 6 | 34 |
| Clay | 19 | 5 | 12 |
| Origami | 24 | 6 | 0 |

`candidate` sources may receive Base FLUX and routed follow-up candidates. `holdout` sources stay
out of pair selection and later LoRA training, and are reserved for qualitative Base-vs-adapted
comparison. `rejected` entries remain in the inventory for an honest account of raw files but do
not run.

Clay's rejected sources are historical museum fragments and vessels. They remain useful evidence
about the ambiguity of reconstructing a natural person from an artifact, but the old progressive
pilot showed that natural outputs often invented unsupported human anatomy. The new fictional clay
portraits therefore form the active pair-bank source pool.
