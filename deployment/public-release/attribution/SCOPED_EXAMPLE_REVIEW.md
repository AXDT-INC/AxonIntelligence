# Frozen example review for the BF16 redistribution decision

The transferred JSONL matches the extraction receipt at 23,478,493 bytes and SHA-256 `641b43e32f3d69fa6fbf74754772e976777a26383c27de2692b90670c5dc5d07`. All 10,684 unique IDs occur in freeze-manifest order. The 68 selected rows match the historical count for each of the 27 PMCIDs. Every selected DOI agrees with its PMC article. This review covers those 68 examples only. [Extraction receipt](../../../engineering-maintenance/frozen-v1-training-extraction-v1-20281029/EXTRACTION_RECEIPT.json) · [Article licenses and publisher evidence](FOCUSED_RIGHTS_REVIEW.md)

## What the records contain

Each example has the same general scientific-assistant system instruction, one generated user prompt and one supervised answer. Each names PMC, a source unit, one or more chunk IDs and a generator. All 68 carry the same historical `CC-BY` label, including the four from the noncommercial article. The label does not establish distribution rights. The records contain text, reported findings and, in four cases, short Python implementations. We found no individual clinical or research records, figure images, complete research tables, sequence files, supplements or source database exports in these examples. Some prompts and answers name figures, derive from published aggregate values or describe other studies.

The comparison used all 27 corresponding version-1 PMC article XML files. Public article retrieval had no access to the training archive. The comparison ran offline in the existing CPU-only release image with read-only inputs. We inspected prompt and answer separately against article prose, including abstracts, methods, results and discussion. Source-unit and chunk identifiers do not pin historical XML bytes or establish which publisher version was originally ingested. The matched passages establish wording in the identified PMC version.

The table assigns one primary review finding per row. **Wording** (36) flags copied prose, quotation, close restatement or a source-specific selection and order for rights review. **Explanation** (25) identifies a newly worded interpretation or implementation, often with scientific terms, figures or short phrases from the paper. **Facts** (7) identifies predominantly reported measurements, names, procedures or findings. The latter two describe how the record reads. Generation from a source precludes inferring independent creation from wording alone. A wording flag identifies potentially protected expression and makes no legal determination. [U.S. Copyright Office on facts and expression](https://www.copyright.gov/help/faq/faq-protect.html)

## Decisions for redistribution

| Sources | Examples | Training-example decision |
| --- | ---: | --- |
| Discoveries, noncommercial article | 4 | Three records reproduce source wording, including an explicit quotation and a 33-token discussion match. One primarily reports treatment facts. Determine the licensed or other basis for the intended uses. Obtain broader permission or hold the affected expression out of a distributable dataset if the terms cannot cover it. |
| Wiley, two accepted manuscripts | 9 | Geriatrics contributes published HRS–Medicare group outcomes and discussion prose. Proteomics contributes study design, aggregate protein associations, reasoning about covariates and a comparison to findings discussed from prior studies. Six records across the two manuscripts warrant expression review. General Wiley terms and mining metadata leave the reuse basis open. |
| Cell Stem Cell, accepted manuscript | 4 | The abstract, discussion and statistical-method language match PMC manuscript text. One entire answer matches the abstract after normalization. Retain manuscript credit and determine the applicable CC BY terms. The version-of-record deposit alone does not establish the version of the manuscript grant. |
| JAMA and 14 JBRA articles, CC BY version unspecified | 25 | The publisher or article grants reuse with credit. Preserve the documented grant and attribution and identify adaptations where applicable. Decide whether that evidence suffices for the proposed release. Supply a version only if established. A JBRA answer discusses Figure 3. Another compares a study cited by the source. Neither includes the figure or the cited paper as a separate object. |
| Seven confirmed CC BY 4.0 articles | 22 | Apply the verified article credit and license to shared covered expression, retain supplied notices and mark adaptations. The records contain published outcomes, interpretations and small implementations. We found no separately supplied data files in them. |
| Unsigned Wiley editorial, CC BY 4.0 | 4 | Use Wiley's title-first editorial credit and preserve its copyright notice. Retain study-level credit for the Halladakeri, Young and Fernandez highlights. Two answers have 20-token matches to the editorial itself. None includes a genome assembly or an annotation collection. |

The **NHLS restriction** covers access to the underlying South African study data. The four proteomics examples contain no participant rows, identifiers, assay matrix or restricted data file. The review found no material in them requiring redistribution of that dataset. The manuscript's own wording remains a separate permission question. Small published subgroup counts elsewhere in the review are identified in the record notes. [Publisher data-availability statement](https://onlinelibrary.wiley.com/doi/10.1002/prca.202300015)

For a **training-example release**, both the prompt and answer would be shared. AxonDAO and counsel must decide the basis for reusing covered wording, the suitability of proposed downstream terms, and the required credits or exclusions. The historical per-row labels should not be presented as verified grants. The row classifications do not clear the rest of the training corpus. [CC BY 4.0 conditions](https://creativecommons.org/licenses/by/4.0/legalcode.en) · [Discoveries issue terms](https://www.discoveriesjournal.com/all-issues/2026-volume-14-issue-2-april-june)

For a **learned BF16 weight release**, direct wording in supervised answers is evidence about what the model was trained on. This review did not measure retention or elicitation of any passage. The record findings alone establish neither a bar to weight release nor permission to release. AxonDAO and counsel must assess relevant acquisition and training terms, applicable law, any expression embodied in or recoverable from the model, and whether further evidence is needed before approving derivative terms and release. The source-code license and upstream Qwen grant do not settle third-party training rights. [Copyright Office analysis of AI training](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf) · [Prior rights review](README.md)

## Record evidence

Line numbers are one-based in the hash-verified JSONL. The record ID and its raw-line SHA-256 (including its line ending, shown in the ID tooltip) bind each finding to the frozen bytes. Counts in the prompt and answer columns are the longest contiguous matching runs found within individual JATS text elements after Unicode NFKC normalization, case folding and splitting into alphanumeric tokens. Punctuation, abbreviations and decimal notation affect these counts. They aid inspection and are neither a legal threshold nor a measure of scientific correctness. Each PMCID links to the specific public XML used for comparison.

### [PMC13428630.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13428630.1/PMC13428630.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 2369 | <span title="SHA-256 df7fccdc1055dc3e20270c1ed1792fa106ddef026c10a9198ed41e3506efb6e8">axonai-pmc-batch002-prod-0002491</span> | 3 | 8 | Wording | Diet comparison and a quoted characterization of relative benefit, followed by the answer's critique of missing quantities. |
| 6694 | <span title="SHA-256 3e11e1e37c71c07a70194e5125508b300e310a60c8e61b0212d958583aab4d3a">axonai-pmc-batch002-prod-0007045</span> | 4 | 12 | Wording | Explicit quotation of the review's cardiovascular trial conclusion within an evidence assessment. |
| 8395 | <span title="SHA-256 89b4107ffd1f3d6a35bd43da2babbff4a8531009cda2930fe3d4ec9fb9f90be2">axonai20k-5b5425158112bd0bd1ea6853</span> | 5 | 9 | Facts | Named treatments, their classes and place in treatment, following the abstract's terminology and sequence. |
| 9261 | <span title="SHA-256 d54be1ceae5e9a8efec034ff10aa4dfee940ad1eeebf7085d76a01707d2960c5">axonai20k-72b2d26d1cc22c391116cde8</span> | 6 | 33 | Wording | Four limitations follow the review's argument and reproduce sustained discussion prose. |

### [PMC13428643.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13428643.1/PMC13428643.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 2103 | <span title="SHA-256 6b2324e9e143abba0cec8f75740dea59e795ae84bbb23a5eb2941b382fbe2e0b">axonai-pmc-batch002-prod-0002212</span> | 3 | 17 | Wording | Copies a discussion list explaining variation in access and care, then comments on its limits. |
| 4764 | <span title="SHA-256 5d28828a213af533e1abb6d45852f75cac28cc3d4c7c7ba83dec87cf3205187c">axonai-pmc-batch002-prod-0005025</span> | 7 | 14 | Wording | Repeats the manuscript's placebo-period result in prose alongside its published estimate and an interpretation. |
| 8763 | <span title="SHA-256 944a85357f9cc01c1ee1ac289c752510384a60495ab870be466a9ff72845d751">axonai20k-a7aeb4b550156e0cefe592aa</span> | 5 | 23 | Wording | Age-stratified admission results with a closely reproduced discussion explanation. |
| 9262 | <span title="SHA-256 3c54709dc188e97a1c430170899d84e24f2ce9fdc3446078648365ba77bc25f5">axonai20k-48da65e2575024968b9ba769</span> | 5 | 11 | Facts | Published subgroup odds ratios and intervals. The longest answer match consists of statistical notation. |
| 10339 | <span title="SHA-256 331899ccd0b74298c5b0489da96f551d8d4f5fa1b5f922fa68c25a39bffcc0ea">axonai20k-accb7c84c278ca49b6809732</span> | 6 | 46 | Wording | Repeats a comparative results sentence including both endpoints and their estimates. Numerals and conventional notation contribute to the length. |

### [PMC13432066.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13432066.1/PMC13432066.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 1210 | <span title="SHA-256 d51b7c011bf3595eaa47d06d2e9e6050aa5ae654c6639d4f7cb8b71869bb00b1">axonai-pmc-batch002-prod-0001275</span> | 3 | 10 | Wording | Explains the choice of correlation statistic and follows the manuscript's comparative rationale closely. |
| 5437 | <span title="SHA-256 a9fcd8f4770a0c5ea8bcb86cd7f21f7ca70f1deffb7b4d743fd7d3c802d54863">axonai-pmc-batch002-prod-0005732</span> | 3 | 11 | Explanation | Links the article's protein association to published sex-stratified mortality and hypertension findings discussed from other studies. The longest match includes reported rate ratios. |
| 7955 | <span title="SHA-256 cca1453314f957008fe74f8d6085fa256f6ddfa851457234e77841da8429772c">axonai20k-5d0b39ac2b600839b01c9715</span> | 7 | 28 | Wording | Restates aggregate adjusted associations, including the published phrasing and statistical notation for a protein effect. |
| 10015 | <span title="SHA-256 3f335874da636c1c79c236a6894bf92a2a11b276a556213fffa6d565062f84d6">axonai20k-052d509aeeba90bdafd9a086</span> | 5 | 10 | Facts | Cohort size, population, timing, assay coverage and clinical covariates using descriptive terminology from the abstract. |

### [PMC13432921.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13432921.1/PMC13432921.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 359 | <span title="SHA-256 fe2017e8bc7ecd47b50c9c031aa3f9e18f67b66443ed662544129b1f73b029c6">axonai-pmc-batch002-prod-0000376</span> | 4 | 11 | Wording | Synthesizes two cell-fate findings with a matching clause from the manuscript discussion. |
| 1279 | <span title="SHA-256 4f6aff1150e8f17d92f3c241f1240bd352cd0a28f76441aa3e5b5b7dd2eec1f1">axonai-pmc-batch002-prod-0001350</span> | 5 | 10 | Wording | Interprets supporting evidence and limitations while retaining manuscript wording about the mechanism. |
| 8404 | <span title="SHA-256 842c787ca545a9729ec1ef35a027b1db5bb9da652db37f2f7cec0c505bf4099c">axonai20k-2df8c0dbb74fe70c8aa2effc</span> | 4 | 20 | Wording | The complete normalized answer matches the abstract's concluding sentence. |
| 10370 | <span title="SHA-256 11f4e1313651ad582bd98e8a91f2d62c8e55f1008df63a3aaed7eaf808d89dca">axonai20k-3dc69900fe44d01335e303fa</span> | 4 | 29 | Wording | Several methods clauses track the manuscript. Named statistical tests and the authors' arrangement of the description both contribute. |

### [PMC13527639.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13527639.1/PMC13527639.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 3871 | <span title="SHA-256 08ee968130d4e1ce6b0efd9bcd3013d34b30430381991d38fb1a95917cf85e7a">axonai-pmc-batch002-prod-0004077</span> | 3 | 9 | Explanation | Explains the composite endpoint using published effect estimates, risk factors and a conventional endpoint list. |
| 6070 | <span title="SHA-256 1eb32f14273b19a61a60f0c80fc75f3e34eab674f41ad7c692517a3de5908fed">axonai-pmc-batch002-prod-0006392</span> | 5 | 9 | Explanation | Interprets follow-up completeness and group comparability. Includes a short quoted baseline descriptor. |

### [PMC13544255.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544255.1/PMC13544255.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 1486 | <span title="SHA-256 4ac5e87cf1472786e673ca94647978e00e8fff694a38aef341df2c65b4a3018b">axonai-pmc-batch002-prod-0001566</span> | 8 | 12 | Wording | Prompt quotes a claim about treatment outcomes. Answer combines study test results with an independent critique of causality. |
| 6353 | <span title="SHA-256 4b59a5d8a3ced8deef5aafb023e632bf1c4b7567d48db1f62bb92a70194c8b2b">axonai-pmc-batch002-prod-0006691</span> | 9 | 10 | Wording | Published subgroup score changes and a close restatement of the paper's interpretation of emotional distress. |

### [PMC13544257.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544257.1/PMC13544257.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 2447 | <span title="SHA-256 f73a410713a3b51c522eee828a905fff99fd1935c811e637d0659c22f2b1a8b6">axonai-pmc-batch002-prod-0002569</span> | 5 | 7 | Explanation | Explains the implications of repeated questionnaire assessments. Names the scale without reproducing its items. |

### [PMC13544258.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544258.1/PMC13544258.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 5292 | <span title="SHA-256 f77eeb38df78ee72387edd7d4d4132ce99cdbdc23b8354091279a1112d0244a8">axonai-pmc-batch002-prod-0005583</span> | 9 | 6 | Explanation | Explains two different units of analysis using published embryo and patient endpoints and significance values. |
| 6204 | <span title="SHA-256 21dc2ba6f4a7047e3e8450a91a227a53f42fe37187bc0e9d32446db5d7778b58">axonai-pmc-batch002-prod-0006536</span> | 10 | 11 | Explanation | Interprets reported proportions and regression results. The longest answer match consists principally of statistical notation. |

### [PMC13544259.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544259.1/PMC13544259.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 3611 | <span title="SHA-256 d876b59ee1b7aa6bb9941ce0a0beaf9b9e63b52c0c5d9ac4f48eb517c1a45956">axonai-pmc-batch002-prod-0003801</span> | 5 | 13 | Wording | Combines reported biomarker findings with a matching mechanistic clause from the discussion. |
| 4835 | <span title="SHA-256 7856b36f7b4888f40c998eed3dc34c23680e91a990bd2561c15db85b990c9191">axonai-pmc-batch002-prod-0005101</span> | 5 | 6 | Explanation | Interprets reported fold changes and named genes, with a brief quoted description of the authors' conclusion. |

### [PMC13544260.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544260.1/PMC13544260.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 3484 | <span title="SHA-256 d033bff3e343e87489f60abec640f4c917a43821439f39a7c71dde40ee446558">axonai-pmc-batch002-prod-0003671</span> | 6 | 8 | Wording | Summarizes results reported in a review and closely follows its enumeration of proposed effects. |

### [PMC13544262.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544262.1/PMC13544262.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 6037 | <span title="SHA-256 d974371619d773479f2bd8ab678bf8ce9c30b83c94bf9d7e2cbd5c89702d649c">axonai-pmc-batch002-prod-0006357</span> | 11 | 6 | Wording | The prompt repeats a clinical-results clause. The answer develops a physiological explanation from published group comparisons. |
| 6702 | <span title="SHA-256 9a70ce1ad742b328647eaa892e01741a1c7a1a27e939d5a0ab62cd94de87bf79">axonai-pmc-batch002-prod-0007053</span> | 5 | 4 | Explanation | Discusses qualitative accuracy and limitations, explicitly referring to Figure 3 panels A–D. No panel or individual measurement appears. |

### [PMC13544263.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544263.1/PMC13544263.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 2338 | <span title="SHA-256 7813b1f5c30da18479cfbf5993ab5c2d434378586ba60fcf60f2a6a69b572709">axonai-pmc-batch002-prod-0002460</span> | 8 | 8 | Explanation | Interprets uncertainty from a published small subgroup, interval and case count. Uses a short discussion phrase about sample size. |
| 3943 | <span title="SHA-256 bc017af7131d8f52ef586a021de395f9c3ee56f153c16573d6bcf0eebdd48a3f">axonai-pmc-batch002-prod-0004153</span> | 5 | 9 | Explanation | Discusses small subgroup counts and reported odds ratios. The individual observations cannot be read from the example. |

### [PMC13544265.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544265.1/PMC13544265.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 1217 | <span title="SHA-256 3a84da682bd18a336d42b27969899d4d41153c58d87a3b0480b9dd0305d1a6fd">axonai-pmc-batch002-prod-0001282</span> | 6 | 6 | Facts | Lists proposed input categories and clinical goals of the review. Contains no genetic, wearable or clinical records. |

### [PMC13544266.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544266.1/PMC13544266.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 6176 | <span title="SHA-256 6a43de7d236b12d1527b2dd1826e06cb23e4098152ad1dc4be0db640493e0db0">axonai-pmc-batch002-prod-0006506</span> | 8 | 4 | Explanation | Explains covariate adjustment using the reported study design and variable names. |
| 6493 | <span title="SHA-256 082a6af61d2051feccd1245c7647a2b8128ff9d94f4968c667a5bc83f68473b9">axonai-pmc-batch002-prod-0006837</span> | 4 | 10 | Wording | Reuses a summary phrase about the prognostic marker while describing the reported association. |

### [PMC13544267.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544267.1/PMC13544267.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 2041 | <span title="SHA-256 91306225d6c026914780d1455af3ffc528dbd8f4075ccfcf17363d4d77c5c089">axonai-pmc-batch002-prod-0002149</span> | 4 | 12 | Wording | Compares aggregate results with Yetkinel et al. as described in the discussion and repeats part of that description. |
| 4423 | <span title="SHA-256 56fee39b40b7d29201ab9f0e270d130b9c1a04b2b35226159d134c9fde9d7588">axonai-pmc-batch002-prod-0004658</span> | 4 | 11 | Wording | Follows the authors' conditional clinical implication and phrasing about larger studies. |

### [PMC13544269.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544269.1/PMC13544269.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 5664 | <span title="SHA-256 30b6e47377bdbbba1361ae9e87a52bc2dacd3f804458652eb7eda818510bbabc">axonai-pmc-batch002-prod-0005970</span> | 5 | 2 | Explanation | Implements the published threshold as a short Python conditional with category labels. No input patient measurements. |

### [PMC13544557.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544557.1/PMC13544557.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 4978 | <span title="SHA-256 fa87635add6120df1bd768e468ced97a5d91557d4e09bb36bd0da5e0a1dd07f8">axonai-pmc-batch002-prod-0005253</span> | 4 | 11 | Wording | Published assay summaries, including a closely worded comparison of viability between conditions. |
| 6346 | <span title="SHA-256 a045dd671555ace069f7320102e76d39c33f4b6b0243de714790db1f5f0799b8">axonai-pmc-batch002-prod-0006684</span> | 7 | 7 | Explanation | Implements a reported calculation in Python. The prompt and docstring reuse short measurement descriptions. |

### [PMC13544558.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544558.1/PMC13544558.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 1109 | <span title="SHA-256 c49b8d194c086fd30c984297f88ff271746e9650fc19de2e8cb84452c0ac683e">axonai-pmc-batch002-prod-0001168</span> | 4 | 6 | Explanation | Interprets group summaries and significance values for two preservation methods. |

### [PMC13544560.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13544560.1/PMC13544560.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 5776 | <span title="SHA-256 2119ac4375e10680543530e8be24f06f3a8e34ccb6c3fe9e035c3bd0d5632743">axonai-pmc-batch002-prod-0006086</span> | 4 | 6 | Explanation | Synthesizes reported mechanisms and discusses a qualitative histology limitation. Contains no histology image. |
| 6482 | <span title="SHA-256 b008cf925becb581e7d1b03372c2a2c856001b79fed9bb1dd390ba692a9abd0a">axonai-pmc-batch002-prod-0006825</span> | 11 | 13 | Wording | Explains a reported inflammatory association using a clause from the paper's discussion. |

### [PMC13408521.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13408521.1/PMC13408521.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 7040 | <span title="SHA-256 4b4c13471c48723754216df3e18eb3f4b35c1c0b9c7753fd10e6256ac2110813">axonai20k-f8cfcaca8dacd0d2a3efd2a0</span> | 3 | 18 | Facts | Reports published group averages, units and significance levels. Numerical notation accounts for much of the matching text. |
| 8595 | <span title="SHA-256 5d8da0dc054363962ddcbf956893507c53e3722294c05df14ae52c103b163ce9">axonai20k-ff326f299b16e6d2385004c7</span> | 6 | 20 | Wording | Repeats results prose reporting biomarker differences between groups. |
| 9045 | <span title="SHA-256 a9f09768c01100de2a601e5eba03f188f42e48808170e02c70212d4b60e2756d">axonai20k-e8f3a957554347fbe1706482</span> | 5 | 14 | Wording | Reuses a clause reporting a null hormone association within an account of published correlations. |

### [PMC13408527.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13408527.1/PMC13408527.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 2004 | <span title="SHA-256 88bb8a68995b4dc7cdbce37e1f37d8fd84d1086271e88d9787f95140fd75c948">axonai-pmc-batch002-prod-0002112</span> | 4 | 6 | Explanation | Connects named network hubs and published expression and survival findings in an explanation. |
| 6262 | <span title="SHA-256 e85f2099bdca04a16e20d3193f07dc09990967efdb7f555f9a42e817d71d73ac">axonai-pmc-batch002-prod-0006596</span> | 3 | 16 | Wording | Reuses a discussion clause about drug properties and relates it to the paper's reported docking results. |
| 7041 | <span title="SHA-256 7f8cbc54e7dfc2408ea7842b76ea1da48aef23900e0c33f612328ae529dfd272">axonai20k-20cc5b1632b78d129b4d6a8b</span> | 11 | 13 | Wording | Prompt follows the authors' conclusion and the answer mirrors part of the pathway description. |
| 8221 | <span title="SHA-256 d64f85981aec9f27e29e2ca50c1200ed92767b689fe7874a5d5a70114b7699ab">axonai20k-0e5d73fed33c5449868308d7</span> | 4 | 5 | Explanation | Implements a short classifier for reported group labels. Contains no interaction-network download or edge list. |
| 9359 | <span title="SHA-256 8a05ca8c8a6b6ea956858c39a5fda14226159ee4b992ba38c6b2e264dad8f4ed">axonai20k-8a0bf1ba26fcb601595bc6aa</span> | 5 | 8 | Explanation | Summarizes the article's analysis stages and research gap using technical terms and some source phrasing. |

### [PMC13437122.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13437122.1/PMC13437122.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 3729 | <span title="SHA-256 b440ac73b219f44da7ac67782212b0cb626c43e6e9cd33da8e2dc0e98aaf9075">axonai-pmc-batch002-prod-0003927</span> | 13 | 30 | Wording | Prompt and answer reproduce mechanistic prose and the order of the review's explanation. |
| 5132 | <span title="SHA-256 573ba24f9221090b800b1df80ee3344595e469cef58656921a9cd639d84d19ee">axonai-pmc-batch002-prod-0005415</span> | 8 | 7 | Explanation | Organizes reported mechanisms into an explanation of two disease processes. Includes named pathways and technical phrases. |
| 7628 | <span title="SHA-256 2565c8aed50169677deb0046bee6ef4b805a89fccfc9cb6609b772e218fddca5">axonai20k-0deafb2ddbe1fbc748ec21cf</span> | 5 | 14 | Wording | Restates the review's selected interventions in its published order. |
| 9597 | <span title="SHA-256 a738172fc5f06d12c5fed51a85799c50111e9cc9643421c09b25b41d2e3429a4">axonai20k-7bb9b8db1b96c9147e9b976d</span> | 7 | 6 | Facts | Summarizes the paper's recommendations for monitoring and validation using short methods phrases. |

### [PMC13437126.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13437126.1/PMC13437126.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 41 | <span title="SHA-256 49a2a06e8b8648fe02a1e62ce22cb105ba22ebe740daf8f111c70150c807e7a6">axonai-pmc-batch002-prod-0000042</span> | 10 | 11 | Wording | Interprets a cell-model comparison while retaining a clause from the abstract. |
| 3251 | <span title="SHA-256 d137ffad47115bb318b5d2edbcc7a08b76113686f0b8f3de509648779c765e77">axonai-pmc-batch002-prod-0003430</span> | 3 | 9 | Explanation | Links published assay, validation and cell-model stages. Contains no omics matrix. |
| 7968 | <span title="SHA-256 372f61859b183a5ce91c6c97715cc79e66d5f6448db51e5cb1c1684655c3fbff">axonai20k-a2b0e8d6dc88c11bd9d8aa18</span> | 3 | 7 | Explanation | Develops the paper's proposed follow-up design and cohort rationale in partly shared technical language. |
| 10381 | <span title="SHA-256 e330ef35dcde2fc410004179e3ae247bb509352a679af7630a2a6660fbfdfc2e">axonai20k-948d14436929f54fa33827ee</span> | 4 | 10 | Facts | Gives the post hoc test and software version. The matching phrase is a conventional software report. |

### [PMC13504950.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13504950.1/PMC13504950.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 596 | <span title="SHA-256 a97906489b380c0752b1c863d256c7d1539b5407624296120b7360cf487446e9">axonai-pmc-batch002-prod-0000628</span> | 8 | 15 | Wording | Synthesizes weekday and weekend associations and closely repeats the outcome comparison. |
| 6104 | <span title="SHA-256 16b723f4b4e303a8ea2e9664494bb3850cfb304c9c58c9af37d21d96d95be34c">axonai-pmc-batch002-prod-0006430</span> | 12 | 16 | Wording | Prompt follows the abstract's study-design phrasing and the answer repeats part of its result. |

### [PMC13504960.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13504960.1/PMC13504960.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 753 | <span title="SHA-256 bf8c2d267b0ecd2c8465c6428af1905c2df32938fbe76754b1a819c80edc01b7">axonai-pmc-batch002-prod-0000793</span> | 4 | 14 | Wording | Explicitly quotes the narrative review's adoption prerequisites within a critique of the evidence. |
| 948 | <span title="SHA-256 f7636a40f6a3a52337aaef3af882f620b106f129ce692cf70c69f2e77c81d2cb">axonai-pmc-batch002-prod-0000999</span> | 5 | 7 | Explanation | Synthesizes infrastructure and governance concerns using the review's headings and technical terms. Contains no athlete stream. |

### [PMC13539385.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13539385.1/PMC13539385.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 3740 | <span title="SHA-256 276d7fcdb1c7ec691c30da5dc7787e1e6efaf267f72f4fff504be05e3819c295">axonai-pmc-batch002-prod-0003938</span> | 4 | 13 | Explanation | Interprets published experimental summaries, including group medians and significance values. Contains no individual survival observations. |
| 4115 | <span title="SHA-256 43051e6788de42ceb42ceda2512bd7a7d585cb7aff01a2b9c2afa78361ae4caa">axonai-pmc-batch002-prod-0004331</span> | 3 | 3 | Explanation | Implements a short lookup of study labels in Python. Contains no genome sequence or source program. |

### [PMC12807375.1](https://pmc-oa-opendata.s3.amazonaws.com/PMC12807375.1/PMC12807375.1.xml)

| Frozen line | Record ID | Prompt | Answer | Finding | Material and source relationship |
| ---: | --- | ---: | ---: | --- | --- |
| 2802 | <span title="SHA-256 d25111325fa5d78222ec4c07870c54587a2f7a0ed458719f4bfdd6c24954cc7b">axonai-pmc-batch002-prod-0002942</span> | 3 | 9 | Wording | Uses quoted phrasing from the pigeonpea highlight and evaluates what remains to be validated. |
| 5868 | <span title="SHA-256 5c98ad2a862de930affc7f1a364e2b62dd0c41e18f257ed7f1a11a559014df55">axonai-pmc-batch002-prod-0006181</span> | 3 | 7 | Explanation | Explains the legume annotation decision rule using the editorial's reported reference-set size. Contains no gene models. |
| 8355 | <span title="SHA-256 91c5dc6b712e14e288710b006459b20306f331e047d181f56043d479bc93c1d5">axonai20k-ec4502735f8b7933fa388be0</span> | 3 | 20 | Wording | Reproduces editorial prose about the ancestry of quinoa in an answer crediting Young et al. |
| 8898 | <span title="SHA-256 ebdebbcfc710199b4fa24ea0f3520182a48be19651cbedf8ec96a95e88c2b40b">axonai20k-4229275592a87d7c88c5f94e</span> | 5 | 20 | Wording | Closely repeats the editorial's explanation of the annotation rule credited to Fernandez et al. |
