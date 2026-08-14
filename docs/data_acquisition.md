# Artistic-image acquisition and curation guide

This guide defines how agents should find and curate source artworks for the face-domain study.
The objective is not to maximize image count. It is to build a small, diverse, legally traceable,
and experimentally useful set.

## 1. Source priority

Use sources in this order:

1. **Existing team-generated material with reconstructable provenance.** Record the model/tool,
   prompt or prompt family, seed if known, creation date, operator, and applicable terms. If these
   facts cannot be reconstructed, label the data `legacy_private`; use it for debugging or private
   qualitative analysis, not public redistribution or strong provenance claims.
2. **CC0/public-domain museum images with first-party metadata.** Prefer official APIs/object pages
   that expose a public-domain flag, rights statement, stable object ID, and image URL.
3. **New synthetic artwork generated under documented model terms.** Use synthetic identities and
   neutral style descriptions. Record model revision, prompt, seed, and output terms.
4. **Other datasets only after license review.** A paper's use of a dataset does not grant this
   project permission to download, redistribute, or create derivatives.

Do not use Pinterest, Google/Bing image thumbnails, social-media reposts, portfolio scraping,
fan-art repositories, or license-unknown WikiArt mirrors as formal sources.

## 2. Recommended first-party museum sources

### National Gallery of Art (United States)

- Open-access terms: <https://www.nga.gov/terms-and-notices>
- Use only object-page images marked by NGA as public domain/open access.
- Record the NGA object page and its rights statement. Do not infer that every NGA image is CC0.

### The Metropolitan Museum of Art

- Official collection API: <https://metmuseum.github.io/>
- Search with `hasImages=true`, then retrieve object records.
- Accept only records where `isPublicDomain` is true and `primaryImage` is present.
- Keep `objectID`, `objectURL`, `primaryImage`, title, artist, date, classification, and department.

### Art Institute of Chicago

- Official API: <https://api.artic.edu/docs/>
- Open-access overview: <https://www.artic.edu/open-access>
- Filter `is_public_domain=true` and require a non-empty `image_id`.
- Use the documented IIIF endpoint. Follow the museum's rate guidance: one image at a time and a
  delay of about one second when local download is necessary.
- Keep the artwork ID, API/object URL, IIIF image ID/URL, title, artist, date, public-domain flag,
  copyright notice, classification, style, and subject fields.

### Rijksmuseum

- Data service documentation: <https://data.rijksmuseum.nl/docs/>
- Information/data policy: <https://data.rijksmuseum.nl/policy/information-and-data-policy>
- Query only records with an available image, then verify the object/image rights statement rather
  than assuming the whole collection has identical terms.

Museum material is most useful for public-domain painting, watercolor, ink, print, and historical
portrait extensions. It will not by itself provide balanced modern `3d_cartoon`, `cyberpunk`, or
generic animation categories.

## 3. Search strategy

### Start from content, then verify style

Use search terms such as `portrait`, `self portrait`, `head`, `bust`, `figure`, `woman`, `man`,
`child`, or their controlled-vocabulary equivalents. Retrieve metadata first, not image-search
results. Then inspect candidates and assign a style only when visual evidence matches the written
category definition.

### Style definitions for the primary study

- `comic`: drawn/inked contours, graphic shading or halftone language; not merely a flat photo.
- `3d_cartoon`: visibly synthetic 3D rendering, stylized geometry/material, non-photographic skin.
- `ink`: ink wash, brush, pen, or line-dominant rendering where strokes define appearance.
- `watercolor`: pigment wash, paper texture, soft bleeding edges, translucent color layering.
- `cyberpunk` extension: clearly stylized neon/technological visual language; exclude ordinary
  nightlife photographs with color grading.
- `animation` extension: cel/painted animation appearance. Use a generic label in reports; avoid
  implying affiliation with a studio or living artist.

If two reviewers cannot distinguish a category from a natural photo or from another category, mark
the sample ambiguous and exclude it from the primary set.

## 4. New synthetic source generation

Synthetic sources are appropriate for styles poorly represented in public-domain museum data.
Follow these rules:

- generate fictional identities, not named real people;
- avoid living-artist names, protected characters, brand logos, and requests to imitate a specific
  studio in public-facing datasets;
- vary age presentation, skin tone, face shape, pose, lighting, clothing, and background without
  creating a demographic claim the sample count cannot support;
- keep the content description stable across style batches when testing style effects;
- record model ID/revision, full positive and negative prompts, seed, sampler/scheduler, steps,
  guidance, resolution, safety settings, and generation date;
- store the original unedited output and checksum;
- review the model license and service terms separately from this repository's code license.

For the first formal set, generate no more than needed to fill missing cells after existing and
museum candidates are curated. Do not spend GPU time producing thousands of unreviewed images.

## 5. Required provenance sidecar

The current `ImageRecord` intentionally stays small. Until the schema is expanded, maintain a
separate `provenance.jsonl` keyed by `source_id`. Recommended fields:

```json
{
  "source_id": "stable-id",
  "source_group_id": "near-duplicate-family-id",
  "local_path": "/persistent/data/raw/example.jpg",
  "sha256": "...",
  "phash": "...",
  "source_kind": "museum_cc0 | synthetic | legacy_private | licensed_dataset",
  "provider": "The Met",
  "provider_object_id": "12345",
  "landing_url": "https://...",
  "image_url": "https://...",
  "title": "...",
  "creator": "...",
  "creation_date": "...",
  "rights_statement": "Public Domain",
  "license": "CC0-1.0",
  "acquired_at": "2026-08-14T00:00:00Z",
  "style_category": "watercolor",
  "content_category": "single_face_portrait",
  "split": "pilot | calibration | test | extension",
  "qc_status": "accepted | rejected | pending",
  "qc_reasons": [],
  "notes": ""
}
```

For synthetic images add a nested generation record with model, revision, prompt, negative prompt,
seed, scheduler, steps, guidance, resolution, and service/local terms. Never fabricate missing
metadata; use explicit `unknown` fields and downgrade the sample to `legacy_private` where needed.

## 6. Download procedure

1. Query first-party metadata and save the candidate record.
2. Check the rights/public-domain flag and landing page.
3. Download sequentially under the provider's documented rate limits.
4. Compute SHA-256 and perceptual hash immediately.
5. Validate image decoding, dimensions, color mode, and obvious corruption.
6. Preserve the raw file; derive resized/cropped inputs into a separate processed directory.
7. Run duplicate grouping before split assignment.
8. Conduct manual QC and record reject reasons.

Do not crop before retaining the raw image and provenance. Do not remove watermarks to make an
otherwise unsuitable image usable.

## 7. Manual quality-control checklist

An image enters the primary set only if all required checks pass:

- rights/provenance recorded;
- valid decode and acceptable resolution;
- one clear primary face for the initial study;
- face is sufficiently large and not heavily occluded;
- declared style is visually present;
- no watermark, caption, UI, collage split, or severe compression;
- no unsafe or exploitative content;
- not a duplicate/near duplicate assigned to a different split;
- enough structure exists for a meaningful destylization comparison.

Record reject reasons from a controlled list, for example:

```text
license_unknown, missing_landing_page, decode_failure, too_small, no_face,
multiple_primary_faces, face_too_small, heavy_occlusion, ambiguous_style,
near_photographic, extreme_abstraction, watermark_or_text, collage,
duplicate, unsafe_content, privacy_risk
```

## 8. Balancing and stopping rule

Do not keep collecting while core code remains unverified. Stop the first acquisition pass when
each primary style has 30 accepted and source-independent images. Freeze pilot/calibration/test
splits before running formal methods. If one style cannot reach 30 without weaker licensing or QC,
replace or postpone that style and document the reason.

Balance obvious nuisance factors where feasible:

- close-up vs. half-body;
- frontal vs. non-frontal pose;
- simple vs. structured background;
- light vs. dark scene;
- different apparent ages and skin tones, without overclaiming fairness from a small sample.

## 9. Existing project archives

The historical archives in the user's project directory contain useful paired examples and
re-screened outputs. They demonstrate prior engineering activity, but filenames and pixels alone do
not establish model, prompt, consent, or license provenance.

Before formal use, create an inventory with:

- archive and member name;
- whether the file is a composite pair and how halves are interpreted;
- checksum and near-duplicate family;
- known generator/tool and date;
- style label and manual QC;
- redistribution status.

If provenance remains incomplete, use these images only for local debugging, qualitative history,
or interview preparation. Do not mix them into a public-domain test set and then describe the whole
set as CC0.

## 10. What agents must deliver from acquisition work

An acquisition task is not complete with a folder of JPGs. It must return:

- candidate and accepted counts by provider/style;
- `provenance.jsonl` and validation report;
- checksums and near-duplicate groups;
- frozen split manifest;
- rejection counts by reason;
- provider/license summary;
- a small contact sheet for manual review;
- no raw data committed to Git.
