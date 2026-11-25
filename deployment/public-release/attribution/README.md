# Attribution reconciliation and redistribution review

The release candidate covers all 4,230 historical PMC identifiers and preserves each article's training count, totaling 10,684 examples. PMC JATS permission paragraphs, copyright statements, article bylines and version identifiers are paired with PMC journal citations. Crossref deposits and targeted publisher records supplied additional evidence. The historical register is preserved byte-for-byte. [Corrected register](SOURCE_ATTRIBUTIONS.rc.json) · [Counts and open cases](RECONCILIATION.json) · [Frozen extraction receipt](../../../engineering-maintenance/frozen-v1-training-extraction-v1-20281029/EXTRACTION_RECEIPT.json)

| Article-text terms supported by the reviewed evidence | Articles |
| --- | ---: |
| CC BY 4.0 | 4,170 |
| CC BY 3.0 | 29 |
| CC BY 3.0 IGO | 2 |
| CC0 1.0 | 9 |
| US public domain according to the publisher | 1 |
| Exact article terms or version unresolved | 19 |

Of 55 historical CC0 labels, 46 refer to research data accompanying CC BY 4.0 articles. Per-item credits can limit the waiver, including for graphical content. Nine records apply CC0 to the work itself. A blank data-waiver field means the review identified no separate waiver in the article-level permission statement. The two IGO records retain their own license links and publisher notices. [Example publisher terms](https://link.springer.com/article/10.1186/s13071-026-07567-y) · [CC BY 3.0 IGO](https://creativecommons.org/licenses/by/3.0/igo/)

The candidate records 78 changes to historical article terms. It updates 109 bylines, restores two missing author lists, records 380 title differences including spacing around markup, and restores two DOIs. Changed bylines retain the historical credits alongside current article authors and available consortium-member credits. The unsigned Wiley editorial now carries the publisher’s title-first citation and retains its copyright notice. [Publisher citation](https://acsess.onlinelibrary.wiley.com/doi/full/10.1002/tpg2.20386)

The focused review resolved seven article licenses covering 22 examples, including two Aging and Disease PDF grants that explicitly link to CC BY 4.0. Nineteen cases affecting 42 examples still have a version, scope or use restriction to resolve. The case inventory separates article-text rights, manuscript permissions and underlying research-data restrictions. The frozen-file review verified the material in all 68 associated examples. [Record findings and redistribution decisions](SCOPED_EXAMPLE_REVIEW.md) · [Focused review of all 27 sources](FOCUSED_RIGHTS_REVIEW.md) · [Remaining cases](RECONCILIATION.json)

The receipt pins the training bytes and historical counts. It does not identify a PMC revision for each source passage. The review records the version retrieved on 2026-11-25. For 265 articles, PMC supplied a revision later than version 1. Rights affecting specific passages, illustrations, data, or versions require checking the material actually used. [PMC copyright guidance](https://pmc.ncbi.nlm.nih.gov/about/copyright/)

## Weight redistribution

The proposed first weight release is the merged BF16 model. Qwen's pinned Apache-2.0 license permits derivative redistribution subject to its conditions. Its license, card, attribution and notices of modification accompany this candidate. AxonDAO must confirm authority over the adapter and merged contributions, determine whether the training rights exceptions impose obligations on weight distribution, select derivative terms, and approve the final bytes and Hub destination. Once decided, record the derivative terms in the model card and accompanying license materials and regenerate the manifest. The GitHub source grant covers AxonDAO-owned code and documentation. [Pinned upstream license](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/LICENSE) · [Inherited rights record](../../../models/merged/AxonAI-v1-27B/RELEASE_RIGHTS_STATUS.inherited.json)

## Training dataset redistribution

The corrected register contains bibliographic metadata and notices. Releasing the 10,684 examples requires a separate decision on the actual expression reused in each example and the applicable article version. Resolve the open terms, commercial-use restriction, supplied credits, transformations and third-party exceptions. Secure permissions or omit affected material where needed. Record the approved dataset contents, credits and terms in a separate manifest. There is no training dataset in the BF16 upload allowlist. [CC BY 4.0 conditions](https://creativecommons.org/licenses/by/4.0/) · [Release inventory and procedure](../hub/README.md)
