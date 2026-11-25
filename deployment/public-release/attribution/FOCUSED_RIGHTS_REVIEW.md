# Focused rights review for the canonical BF16 candidate

This review covers the 26 previously open license cases (64 examples) and the unsigned editorial (four examples). Seven article licenses covering 22 examples are now verified as CC BY 4.0. The editorial has a publisher-specified credit. Nineteen article terms or versions covering 42 examples remain open. These are metadata dispositions. Training-example and weight redistribution each require a separate determination.

## Material used in training

The recovered 23,478,493-byte JSONL matches the frozen training digest `641b43e32f3d69fa6fbf74754772e976777a26383c27de2692b90670c5dc5d07`. All 10,684 record IDs match the freeze manifest in order. Its `manifest_sha256` field, `b0810d15a4aa2968b03690a63a6d38d0ba9f9a9162acd357d08f77dbe772c8a4`, identifies the approved scope manifest. The review matched the 68 scoped prompts and answers to the 27 article PMCIDs and DOIs and compared them with version-1 PMC full texts. The training loss supervised answer and terminal tokens and masked the prompt. Redistributing a serialized example would share both. [Record-by-record evidence and decisions](SCOPED_EXAMPLE_REVIEW.md) · [Extraction receipt](../../../engineering-maintenance/frozen-v1-training-extraction-v1-20281029/EXTRACTION_RECEIPT.json) · [Training provenance](../../reproducibility/README.md)

The article tables count the primary record findings. **Wording** identifies quotation, closely matching prose or a distinctive selection that warrants review. **Explanation** covers newly phrased interpretation or implementation. **Facts** covers predominantly published facts and technical terms. The linked records explain mixed cases, including conventional wording and numerical notation. All 19 open article terms or versions remain flagged.

## Confirmed noncommercial restriction

The final page of the Discoveries article and the publisher's issue terms both condition reuse on noncommercial use. The PMC article-text notice includes the same condition. The versioned CC BY URL occurs in a mining annotation. The exact standard license instrument remains unverified. The restriction concerns use of the article. Separate clearance would be needed for any independently credited content actually reproduced. [Article PDF](https://discoveriesjournals.org/discoveries/D.2026.02.RA.ZehraA.pdf) · [Issue terms](https://www.discoveriesjournal.com/all-issues/2026-volume-14-issue-2-april-june)

**Examples.** Three of the four records contain quoted or closely matched article wording. The fourth chiefly reports treatment facts. Determine the purpose and terms of the proposed dataset distribution. For reuse requiring permission, obtain a basis covering the intended use, including downstream commercial uses if the proposed terms allow them.

**Weights.** Refer the evidenced noncommercial condition to counsel. Assess the basis for source use, the reach of the applicable terms, and any retention or reproducibility of protected expression in the BF16 model. A free download or private staging destination does not decide those questions.

| Exact source article | PMCID | Examples | Actual material |
| --- | --- | ---: | --- |
| [The Dual Role of Uric Acid: Pathological Implications Across Chronic Diseases and Current Approaches to Hyperuricemia Management](https://doi.org/10.15190/d.2026.10) | [PMC13428630](https://pmc.ncbi.nlm.nih.gov/articles/PMC13428630/) | 4 | [3 wording · 1 facts](SCOPED_EXAMPLE_REVIEW.md#pmc134286301) |

## Accepted manuscripts and separate data rights

Both Wiley PDFs identify themselves as author manuscripts. Their notices link to general publisher terms. The Crossref records do not provide a CC license for either version of record. The geriatrics publisher credits the American Geriatrics Society and reserves rights. The proteomics publisher separately states that the South African National Health Laboratory Service data were used under license and require permission for access. That restriction concerns underlying data. Neither a finding reported in the paper nor a biomedical question about it establishes use of the licensed records. [Geriatrics publisher](https://agsjournals.onlinelibrary.wiley.com/doi/abs/10.1111/jgs.70421) · [Proteomics article and data statement](https://onlinelibrary.wiley.com/doi/10.1002/prca.202300015)

The Cell Stem Cell PDF explicitly applies CC BY to the accepted manuscript. The manuscript prose omits a version. Elsevier's Crossref deposit gives CC BY 4.0 for the version of record. Both grants can be true. Preserve their distinct scopes and match the actual source version before selecting a versioned grant. [Accepted manuscript](https://pmc-oa-opendata.s3.amazonaws.com/PMC13432921.1/PMC13432921.1.pdf) · [Publisher deposit](https://api.crossref.org/works/10.1016/j.stem.2025.01.011)

**Examples.** The examples contain manuscript prose, methods and aggregate results. The PMC locators omit historical revision identifiers. One Cell Stem Cell answer matches a complete abstract sentence after normalization. For the Wiley manuscripts, establish the specific reuse permission or other legal basis. The four proteomics records contain study-level descriptions and published statistics. No NHLS participant records or assay matrix appear. For the Cell Stem Cell records, apply the grant for the version actually used and supply its required credits.

**Weights.** Review any relevant acquisition and use terms for each manuscript. Consider data restrictions separately if the source mapping establishes use of controlled data. Determine what, if any, protected expression is embodied in or recoverable from the weights.

| Exact source article | PMCID | Examples | Actual material |
| --- | --- | ---: | --- |
| [Association of Geriatric Emergency Department Care With Hospitalization and Mortality in Older Adults](https://doi.org/10.1111/jgs.70421) | [PMC13428643](https://pmc.ncbi.nlm.nih.gov/articles/PMC13428643/) | 5 | [4 wording · 1 facts](SCOPED_EXAMPLE_REVIEW.md#pmc134286431) |
| [Associations of inflammation-related proteome with demographic and clinical characteristics of people with HIV in South Africa](https://doi.org/10.1002/prca.202300015) | [PMC13432066](https://pmc.ncbi.nlm.nih.gov/articles/PMC13432066/) | 4 | [2 wording · 1 explanation · 1 facts](SCOPED_EXAMPLE_REVIEW.md#pmc134320661) |
| [Sustained NF-κB activation allows mutant alveolar stem cells to co-opt a regeneration program for tumor initiation](https://doi.org/10.1016/j.stem.2025.01.011) | [PMC13432921](https://pmc.ncbi.nlm.nih.gov/articles/PMC13432921/) | 4 | [4 wording](SCOPED_EXAMPLE_REVIEW.md#pmc134329211) |

## CC BY confirmed, version missing

JAMA links the article's CC-BY statement to its own permission page. That page permits reuse of article content with credit to author and journal and omits the version. Each of the fourteen JBRA article notices in PMC grants CC BY for the article text and credits its authors and source. The corresponding CC BY 4.0 URLs are specifically marked for mining. Checking all fourteen DOI-linked publisher PDFs did not establish a general version. Thirteen were byte-identical to the PMC PDFs. For PMC13544265, the two PDFs identify the same article and authors but their hashes differ. A general publisher copyright footer does not cancel an article-specific CC grant. [JAMA article](https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2853474) · [JAMA permission](https://jamanetwork.com/pages/cc-by-license-permissions) · [JBRA publisher](https://www.jbra.com.br/web/)

**Examples.** Preserve the published unversioned grant and article credit. Establish the version if needed for the proposed distribution, indicate changes, and check the rights in anything quoted or adapted. An article-text license alone does not establish rights to patient records, separately credited figures, or supplementary data.

**Weights.** The evidence supports the CC BY license family. The remaining version question does not itself establish whether conditions apply to learned parameters. Determine that from the relevant law, source use, model behavior and proposed release.

| Exact source article | PMCID | Examples | Actual material |
| --- | --- | ---: | --- |
| [Specialist Referral for Cardiovascular Risk in Patients With Prostate Cancer](https://doi.org/10.1001/jamainternmed.2026.4773) | [PMC13527639](https://pmc.ncbi.nlm.nih.gov/articles/PMC13527639/) | 2 | [2 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135276391) |
| [The effect of depression, anxiety, and cognitive flexibility on pregnancy outcomes in women with unexplained infertility](https://doi.org/10.5935/1518-0557.20260032) | [PMC13544255](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544255/) | 2 | [2 wording](SCOPED_EXAMPLE_REVIEW.md#pmc135442551) |
| [Mental Health in Fertile and Infertile Women During the COVID-19 Pandemic: A Comparative Cohort Study](https://doi.org/10.5935/1518-0557.20260035) | [PMC13544257](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544257/) | 1 | [1 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135442571) |
| [Comparison of three embryo culture media: A randomized clinical trial with sibling oocytes](https://doi.org/10.5935/1518-0557.20260025) | [PMC13544258](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544258/) | 2 | [2 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135442581) |
| [Chlorogenic Acid Attenuates Busulfan-Induced Azoospermia in Rats: Antioxidant-Mediated Restoration of Spermatogenesis and Steroidogenesis - A Stereological and Molecular Analysis](https://doi.org/10.5935/1518-0557.20260037) | [PMC13544259](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544259/) | 2 | [1 wording · 1 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135442591) |
| [GLP-1 Receptor Agonists and Fertility: What Is Known So Far?](https://doi.org/10.5935/1518-0557.20260028) | [PMC13544260](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544260/) | 1 | [1 wording](SCOPED_EXAMPLE_REVIEW.md#pmc135442601) |
| [Ovarian hyperstimulation syndrome: severity predictors](https://doi.org/10.5935/1518-0557.20260034) | [PMC13544262](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544262/) | 2 | [1 wording · 1 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135442621) |
| [Influence of chlamydia infection history on tubal patency: A pilot study](https://doi.org/10.5935/1518-0557.20260036) | [PMC13544263](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544263/) | 2 | [2 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135442631) |
| [The Role of Artificial Intelligence in Addressing Menopausal Mental Health Challenges](https://doi.org/10.5935/1518-0557.20260019) | [PMC13544265](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544265/) | 1 | [1 facts](SCOPED_EXAMPLE_REVIEW.md#pmc135442651) |
| [Blastomere/trophectoderm biopsy for preimplantation genetic testing does not affect early HCG levels](https://doi.org/10.5935/1518-0557.20260029) | [PMC13544266](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544266/) | 2 | [1 wording · 1 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135442661) |
| [Blastocyst Morphology in Oocyte Donors stimulated During the Late Follicular Phase with either rFSH:rLH or HP-hMG in a PPOS protocol: a Pilot Randomized Controlled Trial](https://doi.org/10.5935/1518-0557.20260030) | [PMC13544267](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544267/) | 2 | [2 wording](SCOPED_EXAMPLE_REVIEW.md#pmc135442671) |
| [Does Endometrial Thickness Predict Chemical Pregnancy in Frozen Embryo Transfer? Analysis of 522 artificial cycles performed by a single physician](https://doi.org/10.5935/1518-0557.20260038) | [PMC13544269](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544269/) | 1 | [1 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135442691) |
| [In vitro effect of fresh and frozen seminal plasma on the proliferation and viability of periodontal ligament fibroblasts](https://doi.org/10.5935/1518-0557.20260048) | [PMC13544557](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544557/) | 2 | [1 wording · 1 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135445571) |
| [Impact of Cooling Curve Type on Sperm DNA Integrity During Cryopreservation](https://doi.org/10.5935/1518-0557.20260047) | [PMC13544558](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544558/) | 1 | [1 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135445581) |
| [Ethanolic Citrus sinensis peel mitigates resorcinol bis(diphenyl phosphate)-induced testicular toxicity in Wistar rats through oxido-inflammatory pathways](https://doi.org/10.5935/1518-0557.20260044) | [PMC13544560](https://pmc.ncbi.nlm.nih.gov/articles/PMC13544560/) | 2 | [1 wording · 1 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135445601) |

## Article-text CC BY 4.0 confirmed

The two Advanced Pharmaceutical Bulletin article PDFs name CC BY. The publisher's journal licensing statement explicitly specifies version 4.0. For the two Aging and Disease articles, the copyright notice on the first page of each typeset PDF links to CC BY 4.0, despite the PMC mining annotation naming 2.0. The PDFs credit the named authors in 2025 and the JATS rights notices give 2026. Preserve the respective notices for the version used. The Biology of Sport publisher identifies CC BY 4.0 for journal articles, consistent with the two article grants. The microPublication article's copyright paragraph explicitly names CC BY 4.0 International. [APB licensing](https://apb.tbzmed.ac.ir/ContactsandEditorialOffice) · [Zhao PDF, page 1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13437122.1/PMC13437122.1.pdf) · [Corbacho-Alonso PDF, page 1](https://pmc-oa-opendata.s3.amazonaws.com/PMC13437126.1/PMC13437126.1.pdf) · [Biology of Sport licensing](https://pjp.termedia.pl/Journal/Biology_of_Sport-78/office) · [microPublication copyright](https://micropublication.org/journals/biology/micropub.biology.002343/)

**Examples.** Credit the article, provide the license link and identify changes for shared covered material. Check separate credits when an example reuses a third-party item or research dataset. The linked record findings identify which prompts and answers use article wording.

**Weights.** The article-text license version is settled. AxonDAO's review must still decide the basis and obligations for the particular weight release. No additional weight restriction was established by this version correction.

| Exact source article | PMCID | Examples | Actual material |
| --- | --- | ---: | --- |
| [Hidden Hazards: Blood Lead and Cadmium Levels Associated with Biochemical and Hematological Markers](https://doi.org/10.34172/apb.025.46563) | [PMC13408521](https://pmc.ncbi.nlm.nih.gov/articles/PMC13408521/) | 3 | [2 wording · 1 facts](SCOPED_EXAMPLE_REVIEW.md#pmc134085211) |
| [Apigenin as a Multi-Targeted Agent in Gastrointestinal Cancers: A Systems Pharmacology Approach](https://doi.org/10.34172/apb.025.45970) | [PMC13408527](https://pmc.ncbi.nlm.nih.gov/articles/PMC13408527/) | 5 | [2 wording · 3 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc134085271) |
| [Crossing Pathological Boundaries: Multi-Target Restoration of the Neurovascular Unit in Alzheimer's and Vascular Dementia—From Modern Therapeutics to Traditional Chinese Medicine](https://doi.org/10.14336/AD.2025.0801) | [PMC13437122](https://pmc.ncbi.nlm.nih.gov/articles/PMC13437122/) | 4 | [2 wording · 1 explanation · 1 facts](SCOPED_EXAMPLE_REVIEW.md#pmc134371221) |
| [Apolipoprotein CII: Cornerstone of Pathophysiological Interplay in Aortic Stenosis and type 2 Diabetes Mellitus](https://doi.org/10.14336/AD.2025.0518) | [PMC13437126](https://pmc.ncbi.nlm.nih.gov/articles/PMC13437126/) | 4 | [1 wording · 2 explanation · 1 facts](SCOPED_EXAMPLE_REVIEW.md#pmc134371261) |
| [Weekday-specific associations of movement behaviors with body composition in preschool children: A compositional data analysis](https://doi.org/10.5114/biolsport.2026.159563) | [PMC13504950](https://pmc.ncbi.nlm.nih.gov/articles/PMC13504950/) | 2 | [2 wording](SCOPED_EXAMPLE_REVIEW.md#pmc135049501) |
| [Digital twins in sports science: applications for performance enhancement, injury prevention, and rehabilitation through advanced big data analytics and deep learning methodologies – a comprehensive narrative review](https://doi.org/10.5114/biolsport.2026.161106) | [PMC13504960](https://pmc.ncbi.nlm.nih.gov/articles/PMC13504960/) | 2 | [1 wording · 1 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135049601) |
| [Caenorhabditis Intervention Testing Program: all-trans retinoic acid lifespan extension in Caenorhabditis elegans requires 5-Fluoro-2'-deoxyuridine (FUdR)](https://doi.org/10.17912/micropub.biology.002343) | [PMC13539385](https://pmc.ncbi.nlm.nih.gov/articles/PMC13539385/) | 2 | [2 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc135393851) |

## Unsigned editorial credit

Wiley's own citation gives the year, title, journal, volume, article number and DOI without an author. Use the title-first credit in the candidate, retain the published copyright notice, and leave the author list empty. This resolves the bibliographic discrepancy. The authors of the studies summarized by the editorial are not its byline. [Publisher citation](https://acsess.onlinelibrary.wiley.com/doi/full/10.1002/tpg2.20386)

**Examples.** Attribute any reused editorial expression to this title under CC BY 4.0. The four records correspond to three editorial highlights. Credit the studies for the findings they report. Two answers reproduce extended wording found in the editorial. [Halladakeri et al.](https://doi.org/10.1002/tpg2.20342) · [Young et al.](https://doi.org/10.1002/tpg2.20349) · [Fernandez et al.](https://doi.org/10.1002/tpg2.20377)

**Weights.** The byline question adds no evidenced restriction specific to the weights. Consider any actual reuse of protected expression in the same release-specific assessment.

| Exact source article | PMCID | Examples | Actual material |
| --- | --- | ---: | --- |
| [Research highlights, The Plant Genome, Volume 16, Issue 3](https://doi.org/10.1002/tpg2.20386) | [PMC12807375](https://pmc.ncbi.nlm.nih.gov/articles/PMC12807375/) | 4 | [3 wording · 1 explanation](SCOPED_EXAMPLE_REVIEW.md#pmc128073751) |

## Decisions reserved to AxonDAO and counsel

For a dataset release, decide which verified transformations and credits cover the actual records, whether an exception or permission is required for a specific use, and what downstream uses the proposed terms permit. For the BF16 weights, decide the applicable jurisdictions, the legal basis and relevant contracts, the evidence concerning retention or emission of protected expression, and the adapter/derivative licensing terms. The historical internal-use approval, the source-code license, the number of flagged rows and the existence of a private model repository do not resolve these questions. [Original rights record](../../../models/merged/AxonAI-v1-27B/RELEASE_RIGHTS_STATUS.inherited.json) · [PMC copyright guidance](https://pmc.ncbi.nlm.nih.gov/about/copyright/) · [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en) · [U.S. Copyright Office report on generative AI training](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf)
