import gc
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoModelForMultimodalLM,
    AutoProcessor,
    AutoTokenizer,
)


###############################################################################
# PATHS
###############################################################################

ROOT = Path("/data/axonai")

GENERATOR_MODEL = (
    ROOT
    / "models/base/Qwen3.8-27B"
)

VERIFIER_MODEL = (
    ROOT
    / "models/base/Qwen3-32B"
)

BIOMEDICAL_POOL = (
    ROOT
    / "datasets/axonai-v1/generation-pools/"
      "pmc-batch002/biomedical-units.jsonl"
)

LITERATURE_POOL = (
    ROOT
    / "datasets/axonai-v1/generation-pools/"
      "pmc-batch002/literature-reasoning-units.jsonl"
)

ARTICLE_SPLITS = (
    ROOT
    / "datasets/axonai-v1/manifests/splits/"
      "batch-002-pmc-article-splits.jsonl"
)

OUT = (
    ROOT
    / "datasets/axonai-v1/generated/"
      "pmc-batch002-production"
)

ATTEMPTS_OUT = (
    OUT
    / "generation-attempts.jsonl"
)

CANDIDATES_OUT = (
    OUT
    / "candidates.jsonl"
)

VERIFY_OUT = (
    OUT
    / "verification-results.jsonl"
)

ACCEPTED_DETAIL_OUT = (
    OUT
    / "accepted-detailed.jsonl"
)

REJECTED_OUT = (
    OUT
    / "rejected.jsonl"
)

PROGRESS_OUT = (
    OUT
    / "progress.json"
)

REPORT_OUT = (
    OUT
    / "report.json"
)

CONFIG_OUT = (
    OUT
    / "production-config.json"
)

FINAL_SFT = (
    ROOT
    / "datasets/axonai-v1/sft/train/"
      "pmc-batch002-7000.jsonl"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


###############################################################################
# GLOBAL PRODUCTION TARGET
###############################################################################

TOTAL_ACCEPTED_TARGET = 7000

DOMAIN_TARGETS = {
    "biomedical": 5000,
    "literature_reasoning": 2000,
}


###############################################################################
# BIOMEDICAL TASK QUOTAS
#
# 5 * 556 + 4 * 555 = 5000
###############################################################################

TASK_QUOTAS = {
    "biomedical": {
        "scientific_qa": 556,
        "reasoning": 556,
        "result_interpretation": 556,
        "experimental_design": 556,
        "statistical_analysis": 556,
        "uncertainty_assessment": 555,
        "bioinformatics_workflow": 555,
        "coding": 555,
        "evidence_assessment": 555,
    },

    "literature_reasoning": {
        "literature_reasoning": 2000,
    },
}


###############################################################################
# FROZEN STRATUM TARGETS
#
# These are the previously audited PMC Batch 002 generation-plan targets.
###############################################################################

STRATUM_QUOTAS = {
    "biomedical": {
        "genomics_genetics": 450,
        "transcriptomics_rnaseq": 450,
        "proteomics": 350,
        "metabolomics": 300,
        "bioinformatics_computational_biology": 400,
        "molecular_cell_biology": 400,
        "immunology": 350,
        "cancer_biology": 400,
        "neuroscience": 350,
        "microbiology_microbiome": 400,
        "structural_biology": 300,
        "pharmacology_drug_discovery": 350,
        "clinical_research_methods": 250,
        "systems_biology_multiomics": 250,
    },

    "literature_reasoning": {
        "genomics_genetics": 180,
        "transcriptomics_rnaseq": 180,
        "proteomics": 140,
        "metabolomics": 120,
        "bioinformatics_computational_biology": 160,
        "molecular_cell_biology": 160,
        "immunology": 140,
        "cancer_biology": 160,
        "neuroscience": 140,
        "microbiology_microbiome": 160,
        "structural_biology": 120,
        "pharmacology_drug_discovery": 140,
        "clinical_research_methods": 100,
        "systems_biology_multiomics": 100,
    },
}


###############################################################################
# PRODUCTION PARAMETERS
###############################################################################

GENERATION_BATCH_VALID_TARGET = 128

MAX_ACCEPTED_PER_ARTICLE = 2

QUESTION_NEAR_DUPLICATE_THRESHOLD = 0.88

MIN_QUESTION_QUALITY = 4
MIN_ANSWER_QUALITY = 4

RNG_SEED = 2026090607000


###############################################################################
# OUTPUT TOKEN BUDGETS
#
# V4.1 established that 1200 solves coding truncation.
###############################################################################

GENERATION_BUDGETS = {
    "scientific_qa": 600,
    "reasoning": 650,
    "result_interpretation": 650,
    "experimental_design": 700,
    "statistical_analysis": 700,
    "uncertainty_assessment": 650,
    "bioinformatics_workflow": 800,
    "coding": 1200,
    "evidence_assessment": 650,
    "literature_reasoning": 750,
}


###############################################################################
# TAXONOMY — FROZEN AFTER V3/V4
###############################################################################

TASK_TYPES = [
    "scientific_qa",
    "reasoning",
    "result_interpretation",
    "experimental_design",
    "statistical_analysis",
    "uncertainty_assessment",
    "bioinformatics_workflow",
    "coding",
    "literature_reasoning",
    "evidence_assessment",
]


TASK_DEFINITIONS = {
    "scientific_qa":
        (
            "Factual or explanatory scientific QA where no more specialized "
            "cognitive operation predominates."
        ),

    "reasoning":
        (
            "Multi-step scientific inference requiring connection of at "
            "least two supplied facts or findings, and not better described "
            "by another specialized category."
        ),

    "result_interpretation":
        (
            "Interpretation of reported experimental or observational "
            "findings, trends, contrasts, trajectories, or outcomes."
        ),

    "experimental_design":
        (
            "Reasoning about experiment construction, controls, measurements, "
            "sampling, interventions, replication, confounding, or validity."
        ),

    "statistical_analysis":
        (
            "Reasoning about statistical method choice, assumptions, "
            "uncertainty, estimates, effect interpretation, or consequences "
            "of analytical choices. Merely naming a statistical test is "
            "scientific_qa."
        ),

    "uncertainty_assessment":
        (
            "Determining what remains uncertain, unsupported, ambiguous, "
            "confounded, or cannot be concluded from supplied evidence."
        ),

    "bioinformatics_workflow":
        (
            "Reasoning about computational-biology workflow stages, ordering, "
            "inputs and outputs, transformations, tool or method choice, or "
            "why specific processing stages are required."
        ),

    "coding":
        (
            "Implementation, executable or near-executable code, pseudocode, "
            "algorithms, parsing, scientific computation, or concrete data "
            "manipulation."
        ),

    "literature_reasoning":
        (
            "Synthesis across multiple findings, claims, sections, or lines "
            "of scientific evidence."
        ),

    "evidence_assessment":
        (
            "Judging how strongly evidence supports a claim, including "
            "strengths, weaknesses, alternatives, and adequacy."
        ),
}


###############################################################################
# TASK-SPECIFIC GENERATION CONTRACTS — FROZEN FROM V4/V4.1
###############################################################################

TASK_CONTRACTS = {
    "scientific_qa":
        """
Generate a SCIENTIFIC QA example.

The question should primarily require factual or explanatory scientific
understanding.

Good examples include:
- explaining a biological mechanism explicitly supported by the evidence,
- explaining the role of a component,
- identifying and explaining a reported relationship.

DO NOT generate a question whose dominant operation is:
- interpreting a reported result,
- designing an experiment,
- statistical reasoning,
- uncertainty assessment,
- evidence-strength assessment,
- computational workflow reasoning,
- coding,
- synthesis across several separate findings.

If another specialized task clearly fits better, return a skip object.
""",

    "reasoning":
        """
Generate a genuine MULTI-STEP SCIENTIFIC REASONING problem.

The question MUST require connecting at least two distinct facts or findings
from the supplied evidence to derive an inference.

The answer should explicitly make the inferential connection.

DO NOT generate:
- simple factual retrieval,
- a direct restatement,
- a mere summary,
- a result-interpretation task better labelled result_interpretation,
- an uncertainty task better labelled uncertainty_assessment,
- an evidence-strength task better labelled evidence_assessment.

If the evidence cannot support genuine multi-step reasoning, return skip.
""",

    "result_interpretation":
        """
Generate a genuine RESULT INTERPRETATION problem.

The question MUST require interpreting a reported result, comparison,
trajectory, trend, association, change, or experimental outcome.

The answer must explain what the reported finding means biologically or
scientifically while staying within the evidence.

DO NOT merely ask what numerical result was reported.
DO NOT invent mechanisms that are not supported by the evidence.

If no interpretable reported result is available, return skip.
""",

    "experimental_design":
        """
Generate a genuine EXPERIMENTAL DESIGN reasoning problem.

The question MUST require reasoning about one or more of:
- controls,
- measurements,
- sampling,
- interventions,
- replication,
- confounding,
- experimental validity,
- design choices,
- how a design could distinguish competing explanations.

DO NOT merely ask:
- which reagent was used,
- which assay was used,
- what materials were used,
- what procedure was performed.

Any proposed design element must be grounded in the supplied evidence.

If the evidence cannot support genuine design reasoning, return skip.
""",

    "statistical_analysis":
        """
Generate a genuine STATISTICAL ANALYSIS reasoning problem.

The question MUST require reasoning about at least one of:
- why a statistical model or test is appropriate,
- assumptions,
- repeated measurements or dependence,
- covariates,
- interpretation of an estimate,
- confidence or uncertainty,
- effect interpretation,
- consequences of an analytical choice.

DO NOT merely ask:
- which statistical test was used,
- which p-value threshold was used,
- which software produced statistics.

If the evidence does not support genuine statistical reasoning, return skip.
""",

    "uncertainty_assessment":
        """
Generate a genuine UNCERTAINTY ASSESSMENT problem.

The question MUST require identifying what remains uncertain, unsupported,
confounded, ambiguous, or not justified by the supplied evidence.

CRITICAL:
Do NOT invent plausible limitations from general scientific knowledge.
Every limitation or uncertainty in the answer must be explicitly supported
by the evidence or directly inferable from stated evidence.

Clearly distinguish observation from inference.

If the evidence does not support a meaningful uncertainty question, return
skip.
""",

    "bioinformatics_workflow":
        """
Generate a genuine BIOINFORMATICS WORKFLOW reasoning problem.

The question MUST involve at least two computational stages or operations,
such as:
- ordering analysis steps,
- connecting one stage's output to the next stage's input,
- explaining why preprocessing precedes downstream analysis,
- reasoning about alignment, assembly, annotation, quantification,
  normalization, variant analysis, quality control, or similar stages,
- selecting analysis stages based on biological/computational goals.

DO NOT merely ask:
- which software was used,
- which database was used,
- what a single named program does.

Do not invent workflow relationships not established by the evidence.

If the source does not contain sufficient computational workflow evidence,
return skip.
""",

    "coding":
        """
Generate a genuine CODING / IMPLEMENTATION problem.

The question MUST require concrete implementation logic, for example:
- executable Python,
- compact near-executable pseudocode,
- parsing scientific data,
- iterating over records,
- implementing an algorithm,
- performing a scientific calculation,
- transforming biological data.

The answer SHOULD contain actual code when possible.

Keep code compact and complete rather than writing a long tutorial.

DO NOT merely ask:
- which software was used,
- what a tool does conceptually,
- which command was reported.

If the source cannot support a grounded coding problem, return skip.
""",

    "evidence_assessment":
        """
Generate a genuine EVIDENCE ASSESSMENT problem.

The question MUST require judging how strongly the supplied evidence supports
a scientific claim.

It may require considering:
- strength of support,
- competing explanations,
- limitations explicitly present in the evidence,
- whether a conclusion is justified,
- what evidence distinguishes alternatives.

DO NOT invent additional limitations or outside evidence.

If the supplied material cannot support meaningful evidence assessment,
return skip.
""",

    "literature_reasoning":
        """
Generate a genuine LITERATURE REASONING problem.

The question MUST require synthesizing or contrasting at least two distinct
findings, claims, observations, sections, or lines of evidence from the
supplied scientific material.

The answer must perform that synthesis explicitly.

DO NOT generate a single-fact retrieval question.
DO NOT add external literature or scientific facts.
DO NOT convert a simple reported result into artificial multi-step reasoning.

If the evidence cannot support genuine synthesis, return skip.
""",
}


###############################################################################
# GENERATOR SYSTEM PROMPT
###############################################################################

GENERATOR_SYSTEM = (
    "You generate high-quality supervised fine-tuning examples for AxonAI, "
    "a rigorous scientific research assistant. Use ONLY the supplied source "
    "evidence. Never add plausible outside scientific knowledge. Never invent "
    "locations, populations, sample counts, numerical values, pathways, "
    "mechanisms, statistical results, software details, experimental "
    "limitations, replication claims, or conclusions. The requested task "
    "type is a hard cognitive constraint. If the source cannot support that "
    "task, return a skip object rather than forcing a weak example."
)


###############################################################################
# FINAL SFT SYSTEM PROMPT
###############################################################################

SFT_SYSTEM_PROMPT = (
    "You are AxonAI, a rigorous scientific research assistant specializing "
    "in biomedical science, bioinformatics, scientific computing, research "
    "methodology, and evidence-based reasoning. Distinguish observation from "
    "inference, preserve uncertainty, avoid unsupported claims, and reason "
    "carefully from available scientific evidence."
)


###############################################################################
# SOURCE-AWARE PRIORITIZATION TERMS
#
# These are candidate prioritizers ONLY.
# They NEVER bypass generator skips or V3 verification.
###############################################################################

WORKFLOW_TERMS = {
    "rna-seq": 5,
    "rna seq": 5,
    "sequencing": 2,
    "next-generation sequencing": 4,
    "bioinformatics": 5,
    "computational": 2,
    "pipeline": 5,
    "workflow": 5,
    "alignment": 4,
    "aligner": 4,
    "read mapping": 5,
    "mapped reads": 3,
    "reference genome": 3,
    "genome assembly": 4,
    "assembly": 2,
    "annotation": 2,
    "variant calling": 5,
    "variant detection": 4,
    "differential expression": 5,
    "expression analysis": 3,
    "quantification": 2,
    "normalization": 3,
    "quality control": 3,
    "quality filtering": 4,
    "trimming": 3,
    "fastq": 5,
    "fasta": 4,
    "samtools": 5,
    "bcftools": 5,
    "bowtie": 5,
    "bwa": 5,
    "hisat": 5,
    "star aligner": 5,
    "featurecounts": 5,
    "htseq": 5,
    "deseq": 5,
    "edger": 5,
    "blast": 4,
    "multiple sequence alignment": 4,
    "phylogenetic": 2,
    "metagenomic": 4,
    "metagenomics": 4,
    "single-cell": 2,
    "single cell": 2,
    "scrna-seq": 5,
    "mass spectrometry": 2,
}


CODING_TERMS = {
    "algorithm": 4,
    "script": 4,
    "python": 5,
    "r package": 3,
    "software": 1,
    "code": 5,
    "computational": 2,
    "pipeline": 3,
    "fasta": 4,
    "fastq": 4,
    "sequence": 1,
    "matrix": 2,
    "data processing": 3,
    "implemented": 2,
    "calculate": 2,
    "computed": 2,
    "analysis": 1,
}


STATISTICAL_TERMS = {
    "regression": 4,
    "linear model": 4,
    "mixed-effects": 5,
    "mixed effects": 5,
    "anova": 4,
    "confidence interval": 4,
    "odds ratio": 3,
    "hazard ratio": 3,
    "statistical model": 5,
    "multivariate": 3,
    "covariate": 4,
    "random effect": 5,
    "fixed effect": 4,
    "repeated measures": 5,
    "multiple testing": 4,
    "false discovery": 4,
    "adjusted p": 3,
    "p-value": 2,
    "p value": 2,
    "significance": 1,
}


###############################################################################
# HELPERS
###############################################################################

def utc_now():
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


def read_jsonl(path):
    if not path.exists():
        return []

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def append_jsonl(path, row):
    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()


def write_jsonl_atomic(path, rows):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

        f.flush()

    tmp.replace(
        path
    )


def write_json_atomic(path, obj):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tmp.replace(
        path
    )


def pick(record, *keys, default=None):
    for key in keys:
        if (
            key in record
            and record[
                key
            ]
            is not None
        ):
            return record[
                key
            ]

    return default


def context_of(record):
    value = pick(
        record,
        "context_text",
        "source_context",
        "text",
        "content",
        "context",
    )

    if not value:
        raise KeyError(
            "No source context field found."
        )

    return value


def pmcid_of(record):
    return str(
        pick(
            record,
            "pmcid",
            "PMC",
            "article_id",
            default="unknown",
        )
    )


def stratum_of(record):
    return str(
        pick(
            record,
            "stratum",
            "primary_stratum",
            default="unknown",
        )
    )


def source_unit_id_of(record):
    value = pick(
        record,
        "unit_id",
        "source_unit_id",
        "id",
    )

    if value is None:
        raw = (
            pmcid_of(
                record
            )
            + "\n"
            + context_of(
                record
            )
        )

        value = (
            hashlib.sha256(
                raw.encode(
                    "utf-8"
                )
            )
            .hexdigest()[:24]
        )

    return str(
        value
    )


def normalize_question(text):
    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-z0-9 ]+",
        "",
        text,
    )

    return text.strip()


def question_similarity(a, b):
    a_tokens = set(
        normalize_question(
            a
        ).split()
    )

    b_tokens = set(
        normalize_question(
            b
        ).split()
    )

    if (
        not a_tokens
        or not b_tokens
    ):
        return 0.0

    return (
        2.0
        * len(
            a_tokens
            & b_tokens
        )
        / (
            len(
                a_tokens
            )
            + len(
                b_tokens
            )
        )
    )


def extract_json(text):
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.S,
    ).strip()

    if text.startswith(
        "```"
    ):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.I,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    try:
        obj = json.loads(
            text
        )

        if isinstance(
            obj,
            dict,
        ):
            return obj

    except Exception:
        pass

    start = text.find(
        "{"
    )

    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(
        start,
        len(text),
    ):
        ch = text[i]

        if in_string:
            if escaped:
                escaped = False

            elif ch == "\\":
                escaped = True

            elif ch == '"':
                in_string = False

            continue

        if ch == '"':
            in_string = True

        elif ch == "{":
            depth += 1

        elif ch == "}":
            depth -= 1

            if depth == 0:
                try:
                    obj = json.loads(
                        text[
                            start:
                            i + 1
                        ]
                    )

                except Exception:
                    return None

                return (
                    obj
                    if isinstance(
                        obj,
                        dict,
                    )
                    else None
                )

    return None


def keyword_score(
    text,
    terms,
):
    text = text.lower()

    score = 0

    for term, weight in (
        terms.items()
    ):
        if term in text:
            score += weight

    return score


def source_priority_score(
    task,
    record,
):
    text = context_of(
        record
    )

    if (
        task
        == "bioinformatics_workflow"
    ):
        return keyword_score(
            text,
            WORKFLOW_TERMS,
        )

    if task == "coding":
        return keyword_score(
            text,
            CODING_TERMS,
        )

    if (
        task
        == "statistical_analysis"
    ):
        return keyword_score(
            text,
            STATISTICAL_TERMS,
        )

    return 0


###############################################################################
# CONFIGURATION FINGERPRINT
#
# Prevent accidental resume with materially different quotas/config.
###############################################################################

PRODUCTION_CONFIG = {
    "version":
        "pmc-batch002-production-v1",

    "target":
        TOTAL_ACCEPTED_TARGET,

    "base_model":
        "Qwen/Qwen3.8-27B",

    "verifier":
        "Qwen/Qwen3-32B-V3-two-stage",

    "domain_targets":
        DOMAIN_TARGETS,

    "task_quotas":
        TASK_QUOTAS,

    "stratum_quotas":
        STRATUM_QUOTAS,

    "generation_budgets":
        GENERATION_BUDGETS,

    "max_accepted_per_article":
        MAX_ACCEPTED_PER_ARTICLE,

    "near_duplicate_threshold":
        QUESTION_NEAR_DUPLICATE_THRESHOLD,

    "generation_batch_valid_target":
        GENERATION_BATCH_VALID_TARGET,

    "seed":
        RNG_SEED,
}


config_payload = json.dumps(
    PRODUCTION_CONFIG,
    sort_keys=True,
    separators=(
        ",",
        ":",
    ),
)

CONFIG_HASH = (
    hashlib.sha256(
        config_payload.encode(
            "utf-8"
        )
    )
    .hexdigest()
)


if CONFIG_OUT.exists():
    existing = json.loads(
        CONFIG_OUT.read_text(
            encoding="utf-8"
        )
    )

    if (
        existing.get(
            "config_hash"
        )
        != CONFIG_HASH
    ):
        raise RuntimeError(
            "Existing production workspace uses a different configuration. "
            "Refusing to resume into incompatible state."
        )

else:
    config_record = {
        **PRODUCTION_CONFIG,
        "config_hash":
            CONFIG_HASH,

        "created_utc":
            utc_now(),
    }

    write_json_atomic(
        CONFIG_OUT,
        config_record,
    )


###############################################################################
# QUOTA SANITY
###############################################################################

assert (
    sum(
        DOMAIN_TARGETS.values()
    )
    == TOTAL_ACCEPTED_TARGET
)

for domain, target in (
    DOMAIN_TARGETS.items()
):
    assert (
        sum(
            TASK_QUOTAS[
                domain
            ].values()
        )
        == target
    )

    assert (
        sum(
            STRATUM_QUOTAS[
                domain
            ].values()
        )
        == target
    )


###############################################################################
# LOAD ARTICLE SPLIT MANIFEST
###############################################################################

article_split = {}

if ARTICLE_SPLITS.exists():
    for row in read_jsonl(
        ARTICLE_SPLITS
    ):
        pmcid = str(
            pick(
                row,
                "pmcid",
                "PMC",
                "article_id",
                default="",
            )
        )

        split = pick(
            row,
            "split",
        )

        if pmcid:
            article_split[
                pmcid
            ] = split


###############################################################################
# LOAD / VALIDATE SOURCE POOLS
###############################################################################

biomedical_pool = read_jsonl(
    BIOMEDICAL_POOL
)

literature_pool = read_jsonl(
    LITERATURE_POOL
)

assert biomedical_pool
assert literature_pool


def validate_pool(
    domain,
    pool,
):
    explicit_non_train = 0
    invalid_license = 0
    unknown_strata = Counter()

    for row in pool:
        split = pick(
            row,
            "split",
            default="train",
        )

        if split != "train":
            explicit_non_train += 1

        license_name = pick(
            row,
            "license",
        )

        if (
            license_name
            not in {
                "CC-BY",
                "CC0",
            }
        ):
            invalid_license += 1

        stratum = stratum_of(
            row
        )

        if (
            stratum
            not in STRATUM_QUOTAS[
                domain
            ]
        ):
            unknown_strata[
                stratum
            ] += 1

        pmcid = pmcid_of(
            row
        )

        if (
            article_split
            and pmcid
            in article_split
            and article_split[
                pmcid
            ]
            != "train"
        ):
            explicit_non_train += 1

    if explicit_non_train:
        raise RuntimeError(
            f"{domain}: source pool contains non-train records."
        )

    if invalid_license:
        raise RuntimeError(
            f"{domain}: source pool contains invalid licenses."
        )

    if unknown_strata:
        raise RuntimeError(
            f"{domain}: unknown strata: "
            f"{dict(unknown_strata)}"
        )


validate_pool(
    "biomedical",
    biomedical_pool,
)

validate_pool(
    "literature_reasoning",
    literature_pool,
)


POOL_BY_DOMAIN = {
    "biomedical":
        biomedical_pool,

    "literature_reasoning":
        literature_pool,
}


###############################################################################
# BUILD SOURCE INDICES BY DOMAIN / STRATUM / TASK
###############################################################################

rng = random.Random(
    RNG_SEED
)

SOURCE_INDICES = {}


for domain, pool in (
    POOL_BY_DOMAIN.items()
):
    SOURCE_INDICES[
        domain
    ] = {}

    for task in TASK_QUOTAS[
        domain
    ]:
        SOURCE_INDICES[
            domain
        ][
            task
        ] = {}

        for stratum in STRATUM_QUOTAS[
            domain
        ]:
            rows = [
                row
                for row in pool
                if (
                    stratum_of(
                        row
                    )
                    == stratum
                )
            ]

            decorated = []

            for row in rows:
                score = source_priority_score(
                    task,
                    row,
                )

                random_key = rng.random()

                decorated.append(
                    (
                        -score,
                        random_key,
                        row,
                    )
                )

            decorated.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                )
            )

            SOURCE_INDICES[
                domain
            ][
                task
            ][
                stratum
            ] = [
                row
                for _, _, row
                in decorated
            ]


###############################################################################
# LOAD AUTHORITATIVE EXISTING STATE
###############################################################################

attempts = read_jsonl(
    ATTEMPTS_OUT
)

candidates = read_jsonl(
    CANDIDATES_OUT
)

verifications = read_jsonl(
    VERIFY_OUT
)


candidate_by_id = {
    row[
        "candidate_id"
    ]:
        row

    for row in candidates
}


verification_by_id = {}

for row in verifications:
    verification_by_id[
        row[
            "candidate_id"
        ]
    ] = row


used_generation_keys = {
    (
        row[
            "domain"
        ],
        row[
            "task_type"
        ],
        str(
            row[
                "source_unit_id"
            ]
        ),
    )
    for row in attempts
    if row.get(
        "source_unit_id"
    )
}


###############################################################################
# ACCEPTED STATE DERIVATION
###############################################################################

def derive_accepted_state():
    accepted_candidates = []

    accepted_by_domain = Counter()
    accepted_by_task = Counter()
    accepted_by_stratum = Counter()
    accepted_by_article = Counter()

    accepted_questions_by_task = defaultdict(
        list
    )

    for candidate_id, verification in (
        verification_by_id.items()
    ):
        if not verification.get(
            "accepted"
        ):
            continue

        candidate = candidate_by_id.get(
            candidate_id
        )

        if candidate is None:
            raise RuntimeError(
                f"Accepted verification references missing candidate "
                f"{candidate_id}"
            )

        accepted_candidates.append(
            candidate
        )

        domain = candidate[
            "domain"
        ]

        task = candidate[
            "task_type"
        ]

        stratum = candidate[
            "stratum"
        ]

        pmcid = candidate[
            "pmcid"
        ]

        accepted_by_domain[
            domain
        ] += 1

        accepted_by_task[
            (
                domain,
                task,
            )
        ] += 1

        accepted_by_stratum[
            (
                domain,
                stratum,
            )
        ] += 1

        accepted_by_article[
            pmcid
        ] += 1

        accepted_questions_by_task[
            task
        ].append(
            candidate[
                "question"
            ]
        )

    return {
        "candidates":
            accepted_candidates,

        "by_domain":
            accepted_by_domain,

        "by_task":
            accepted_by_task,

        "by_stratum":
            accepted_by_stratum,

        "by_article":
            accepted_by_article,

        "questions_by_task":
            accepted_questions_by_task,
    }


###############################################################################
# MATERIALIZE ACCEPTED / REJECTED / FINAL SFT FROM AUTHORITATIVE STATE
###############################################################################

def materialize_outputs():
    accepted_detail = []
    rejected = []
    final_sft = []

    for candidate_id, verification in sorted(
        verification_by_id.items()
    ):
        candidate = candidate_by_id.get(
            candidate_id
        )

        if candidate is None:
            continue

        if verification.get(
            "accepted"
        ):
            grounding = verification[
                "grounding"
            ]

            task_audit = verification[
                "task_audit"
            ]

            accepted_record = {
                **candidate,

                "verification":
                    verification,
            }

            accepted_detail.append(
                accepted_record
            )

            q = task_audit[
                "question_quality"
            ]

            a = grounding[
                "answer_quality"
            ]

            sft = {
                "id":
                    (
                        "axonai-pmc-batch002-"
                        + candidate_id
                    ),

                "messages": [
                    {
                        "role":
                            "system",

                        "content":
                            SFT_SYSTEM_PROMPT,
                    },

                    {
                        "role":
                            "user",

                        "content":
                            candidate[
                                "question"
                            ],
                    },

                    {
                        "role":
                            "assistant",

                        "content":
                            candidate[
                                "answer"
                            ],
                    },
                ],

                "domain":
                    candidate[
                        "domain"
                    ],

                "task_type":
                    candidate[
                        "task_type"
                    ],

                "source":
                    "PubMed Central",

                "license":
                    candidate[
                        "license"
                    ],

                "quality_score":
                    (
                        q
                        + a
                    )
                    / 10.0,

                "provenance": {
                    "pmcid":
                        candidate[
                            "pmcid"
                        ],

                    "doi":
                        candidate.get(
                            "doi"
                        ),

                    "pmid":
                        candidate.get(
                            "pmid"
                        ),

                    "primary_stratum":
                        candidate[
                            "stratum"
                        ],

                    "source_unit_id":
                        candidate[
                            "source_unit_id"
                        ],

                    "source_chunk_ids":
                        candidate.get(
                            "source_chunk_ids",
                            [],
                        ),

                    "generator":
                        "Qwen/Qwen3.8-27B",

                    "verifier":
                        "Qwen/Qwen3-32B-V3-two-stage",

                    "generation_contract":
                        "V4/V4.1-frozen",

                    "split":
                        "train",
                },

                "verification": {
                    "fully_grounded":
                        grounding[
                            "fully_grounded"
                        ],

                    "unsupported_claims":
                        grounding[
                            "unsupported_claims"
                        ],

                    "answer_quality":
                        a,

                    "best_task_type":
                        task_audit[
                            "best_task_type"
                        ],

                    "assigned_task_is_good_match":
                        task_audit[
                            "assigned_task_is_good_match"
                        ],

                    "question_quality":
                        q,

                    "max_question_similarity":
                        verification.get(
                            "max_question_similarity",
                            0.0,
                        ),
                },
            }

            final_sft.append(
                sft
            )

        else:
            rejected.append({
                **candidate,

                "verification":
                    verification,
            })

    write_jsonl_atomic(
        ACCEPTED_DETAIL_OUT,
        accepted_detail,
    )

    write_jsonl_atomic(
        REJECTED_OUT,
        rejected,
    )

    write_jsonl_atomic(
        FINAL_SFT,
        final_sft,
    )


###############################################################################
# PROGRESS REPORT
###############################################################################

def current_progress():
    state = derive_accepted_state()

    total = len(
        state[
            "candidates"
        ]
    )

    progress = {
        "updated_utc":
            utc_now(),

        "target":
            TOTAL_ACCEPTED_TARGET,

        "accepted":
            total,

        "remaining":
            TOTAL_ACCEPTED_TARGET
            - total,

        "generation_attempts":
            len(
                attempts
            ),

        "valid_candidates":
            len(
                candidates
            ),

        "verified_candidates":
            len(
                verification_by_id
            ),

        "accepted_by_domain": {
            domain:
                state[
                    "by_domain"
                ][
                    domain
                ]

            for domain
            in DOMAIN_TARGETS
        },

        "accepted_by_task": {
            domain: {
                task:
                    state[
                        "by_task"
                    ][
                        (
                            domain,
                            task,
                        )
                    ]

                for task
                in TASK_QUOTAS[
                    domain
                ]
            }

            for domain
            in DOMAIN_TARGETS
        },

        "accepted_by_stratum": {
            domain: {
                stratum:
                    state[
                        "by_stratum"
                    ][
                        (
                            domain,
                            stratum,
                        )
                    ]

                for stratum
                in STRATUM_QUOTAS[
                    domain
                ]
            }

            for domain
            in DOMAIN_TARGETS
        },
    }

    write_json_atomic(
        PROGRESS_OUT,
        progress,
    )

    return progress


###############################################################################
# INITIAL STATE
###############################################################################

materialize_outputs()

progress = current_progress()


print("=" * 88)
print("AXONAI PMC BATCH 002 — 7,000 ACCEPTED PRODUCTION")
print("=" * 88)

print(
    "Configuration hash:",
    CONFIG_HASH,
)

print(
    "Existing generation attempts:",
    len(
        attempts
    ),
)

print(
    "Existing valid candidates:",
    len(
        candidates
    ),
)

print(
    "Existing verified candidates:",
    len(
        verification_by_id
    ),
)

print(
    "Existing accepted:",
    progress[
        "accepted"
    ],
    "/",
    TOTAL_ACCEPTED_TARGET,
)

print(
    "Remaining:",
    progress[
        "remaining"
    ],
)


###############################################################################
# SOURCE SELECTION
###############################################################################

source_cursor = defaultdict(
    int
)


def choose_source(
    domain,
    task,
    stratum,
    accepted_by_article,
    cycle_article_reservations,
):
    rows = SOURCE_INDICES[
        domain
    ][
        task
    ][
        stratum
    ]

    key = (
        domain,
        task,
        stratum,
    )

    start = source_cursor[
        key
    ]

    for offset in range(
        len(
            rows
        )
    ):
        index = (
            start
            + offset
        ) % len(
            rows
        )

        row = rows[
            index
        ]

        unit_id = source_unit_id_of(
            row
        )

        generation_key = (
            domain,
            task,
            unit_id,
        )

        if (
            generation_key
            in used_generation_keys
        ):
            continue

        pmcid = pmcid_of(
            row
        )

        if (
            accepted_by_article[
                pmcid
            ]
            + cycle_article_reservations[
                pmcid
            ]
            >= MAX_ACCEPTED_PER_ARTICLE
        ):
            continue

        score = source_priority_score(
            task,
            row,
        )

        if (
            task
            == "bioinformatics_workflow"
            and score < 5
        ):
            continue

        source_cursor[
            key
        ] = (
            index
            + 1
        ) % len(
            rows
        )

        return (
            row,
            score,
        )

    return (
        None,
        None,
    )


###############################################################################
# COMBINATION CHOICE
#
# Satisfy task quotas and stratum quotas simultaneously.
###############################################################################

def choose_generation_combination(
    state,
    cycle_task_valid,
    cycle_stratum_valid,
    blocked,
):
    combinations = []

    for domain in DOMAIN_TARGETS:

        for task, task_target in (
            TASK_QUOTAS[
                domain
            ].items()
        ):
            task_accepted = (
                state[
                    "by_task"
                ][
                    (
                        domain,
                        task,
                    )
                ]
            )

            task_remaining = (
                task_target
                - task_accepted
                - cycle_task_valid[
                    (
                        domain,
                        task,
                    )
                ]
            )

            if task_remaining <= 0:
                continue

            task_fraction = (
                task_remaining
                / task_target
            )

            for stratum, stratum_target in (
                STRATUM_QUOTAS[
                    domain
                ].items()
            ):
                combo = (
                    domain,
                    task,
                    stratum,
                )

                if combo in blocked:
                    continue

                stratum_accepted = (
                    state[
                        "by_stratum"
                    ][
                        (
                            domain,
                            stratum,
                        )
                    ]
                )

                stratum_remaining = (
                    stratum_target
                    - stratum_accepted
                    - cycle_stratum_valid[
                        (
                            domain,
                            stratum,
                        )
                    ]
                )

                if stratum_remaining <= 0:
                    continue

                stratum_fraction = (
                    stratum_remaining
                    / stratum_target
                )

                priority = (
                    task_fraction
                    + stratum_fraction
                )

                combinations.append(
                    (
                        -priority,
                        -task_remaining,
                        -stratum_remaining,
                        combo,
                    )
                )

    if not combinations:
        return None

    combinations.sort()

    return combinations[
        0
    ][
        3
    ]


###############################################################################
# GENERATOR
###############################################################################

def load_generator():
    print()
    print(
        "Loading Qwen3.8-27B generator..."
    )

    processor = AutoProcessor.from_pretrained(
        GENERATOR_MODEL,
        local_files_only=True,
    )

    model = (
        AutoModelForMultimodalLM
        .from_pretrained(
            GENERATOR_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    )

    model.eval()

    device = next(
        model.parameters()
    ).device

    print(
        "Generator loaded."
    )

    return (
        processor,
        model,
        device,
    )


def unload_generator(
    processor,
    model,
):
    print(
        "Unloading generator..."
    )

    del model
    del processor

    gc.collect()
    torch.cuda.empty_cache()

    print(
        "Generator unloaded."
    )


def generate_one(
    processor,
    model,
    device,
    domain,
    task,
    source,
):
    evidence = context_of(
        source
    )

    prompt = f"""
Create ONE supervised fine-tuning example.

DATASET DOMAIN:
{domain}

REQUESTED TASK TYPE:
{task}

TASK DEFINITION:
{TASK_DEFINITIONS[task]}

TASK-SPECIFIC CONTRACT:
{TASK_CONTRACTS[task]}

SOURCE EVIDENCE:
----------------
{evidence}
----------------

STRICT GROUNDING RULES:

1. Use ONLY information contained in SOURCE EVIDENCE.
2. Do not add plausible outside scientific knowledge.
3. Do not invent geography, patient populations, sample counts, numerical
   values, pathways, mechanisms, software details, statistical values,
   experimental limitations, replication claims, or conclusions.
4. If a limitation or inference is not supported by SOURCE EVIDENCE, do not
   introduce it.
5. The generated question MUST predominantly require REQUESTED TASK TYPE.
6. The answer must directly answer the question.
7. Prefer one focused, high-information question over several unrelated
   subquestions.
8. If SOURCE EVIDENCE cannot support the requested cognitive operation,
   return a skip object rather than forcing an example.
9. Keep the answer concise while still complete.
10. For coding, provide compact complete implementation logic rather than a
    long tutorial.

Return ONLY one of:

{{
  "question": "...",
  "answer": "..."
}}

OR:

{{
  "skip": true,
  "reason": "source does not support requested task"
}}
""".strip()

    messages = [
        {
            "role":
                "system",

            "content":
                GENERATOR_SYSTEM,
        },

        {
            "role":
                "user",

            "content":
                prompt,
        },
    ]

    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = processor(
        text=rendered,
        return_tensors="pt",
    )

    inputs = {
        key:
            value.to(
                device
            )
            if torch.is_tensor(
                value
            )
            else value

        for key, value
        in inputs.items()
    }

    input_len = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=
                GENERATION_BUDGETS[
                    task
                ],
            do_sample=True,
            temperature=0.60,
            top_p=0.90,
            pad_token_id=
                processor.tokenizer.pad_token_id,
        )

    raw = processor.tokenizer.decode(
        output[
            0,
            input_len:,
        ],
        skip_special_tokens=True,
    ).strip()

    return (
        extract_json(
            raw
        ),
        raw,
    )


###############################################################################
# VERIFIER
###############################################################################

def load_verifier():
    print()
    print(
        "Loading Qwen3-32B V3 verifier..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        VERIFIER_MODEL,
        local_files_only=True,
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            VERIFIER_MODEL,
            dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    )

    model.eval()

    device = next(
        model.parameters()
    ).device

    print(
        "Verifier loaded."
    )

    return (
        tokenizer,
        model,
        device,
    )


def unload_verifier(
    tokenizer,
    model,
):
    print(
        "Unloading verifier..."
    )

    del model
    del tokenizer

    gc.collect()
    torch.cuda.empty_cache()

    print(
        "Verifier unloaded."
    )


def verifier_call(
    tokenizer,
    model,
    device,
    system,
    prompt,
    max_new_tokens,
):
    messages = [
        {
            "role":
                "system",

            "content":
                system,
        },

        {
            "role":
                "user",

            "content":
                prompt,
        },
    ]

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        rendered,
        return_tensors="pt",
    )

    inputs = {
        key:
            value.to(
                device
            )

        for key, value
        in inputs.items()
    }

    input_len = (
        inputs[
            "input_ids"
        ].shape[1]
    )

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=
                max_new_tokens,
            do_sample=False,
            pad_token_id=
                tokenizer.pad_token_id,
        )

    raw = tokenizer.decode(
        output[
            0,
            input_len:,
        ],
        skip_special_tokens=True,
    ).strip()

    return (
        extract_json(
            raw
        ),
        raw,
    )


###############################################################################
# V3 GROUNDING GATE
###############################################################################

def grounding_audit(
    tokenizer,
    model,
    device,
    candidate,
):
    prompt = f"""
SOURCE EVIDENCE:
----------------
{candidate["source_context"]}
----------------

QUESTION:
{candidate["question"]}

ANSWER:
{candidate["answer"]}

Audit every substantive claim in ANSWER against SOURCE EVIDENCE.

A claim is unsupported if supplied evidence is insufficient to support it,
even if the claim sounds scientifically plausible or may be true elsewhere.

Inspect particularly carefully:
- geography,
- populations,
- sample counts,
- numerical values,
- methods,
- pathways,
- mechanisms,
- causal claims,
- replication claims,
- limitations,
- conclusions,
- relationships between computational workflow stages.

Return ONLY JSON:

{{
  "fully_grounded": true,
  "unsupported_claims": [],
  "answer_quality": 5,
  "reason": "brief explanation"
}}

Rules:

- fully_grounded MUST be false if even one material claim is unsupported.
- unsupported_claims must list every material unsupported claim detected.
- answer_quality:
  5 = accurate, complete and appropriately calibrated
  4 = good with minor weakness
  3 = materially incomplete or overstated
  2 = poor
  1 = incorrect
""".strip()

    return verifier_call(
        tokenizer,
        model,
        device,
        (
            "You perform strict claim-level "
            "scientific grounding audits."
        ),
        prompt,
        420,
    )


###############################################################################
# V3 TASK GATE
###############################################################################

def task_audit(
    tokenizer,
    model,
    device,
    candidate,
):
    definitions = "\n".join(
        (
            f"- {task}: "
            f"{TASK_DEFINITIONS[task]}"
        )

        for task
        in TASK_TYPES
    )

    prompt = f"""
QUESTION:
{candidate["question"]}

ASSIGNED TASK TYPE:
{candidate["task_type"]}

Determine the cognitive operation actually required by QUESTION.

TASK TAXONOMY:

{definitions}

Rules:

1. Classify the required cognitive operation, not the scientific topic.

2. scientific_qa is the factual/explanatory fallback when no specialized
   operation predominates.

3. Merely naming a statistical test is scientific_qa.
   statistical_analysis requires genuine statistical reasoning.

4. experimental_design requires reasoning about controls, measurements,
   interventions, sampling, replication, confounding, or validity.

5. result_interpretation means interpreting reported findings, trajectories,
   trends, contrasts, or outcomes.

6. uncertainty_assessment focuses on uncertainty or limits of inference.

7. evidence_assessment judges strength or adequacy of evidence.

8. literature_reasoning synthesizes multiple findings or lines of evidence.

9. bioinformatics_workflow requires computational biological pipeline logic.

10. coding requires concrete implementation or algorithmic operations.

Return ONLY JSON:

{{
  "best_task_type": "exact task label",
  "second_best_task_type": "exact task label",
  "assigned_task_is_good_match": true,
  "question_quality": 5,
  "reason": "brief explanation"
}}

QUESTION QUALITY:
5 = strong, unambiguous instance of assigned task
4 = good match with minor ambiguity
3 = useful but materially mismatched or shallow
2 = poor
1 = unusable
""".strip()

    return verifier_call(
        tokenizer,
        model,
        device,
        (
            "You classify scientific questions "
            "by required cognitive operation."
        ),
        prompt,
        380,
    )


###############################################################################
# GENERATE ONE PRODUCTION BATCH
###############################################################################

def generate_batch():
    state = derive_accepted_state()

    total_accepted = len(
        state[
            "candidates"
        ]
    )

    remaining = (
        TOTAL_ACCEPTED_TARGET
        - total_accepted
    )

    if remaining <= 0:
        return 0

    valid_target = min(
        GENERATION_BATCH_VALID_TARGET,
        remaining,
    )

    cycle_task_valid = Counter()
    cycle_stratum_valid = Counter()
    cycle_article_reservations = Counter()

    blocked = set()

    valid_created = 0

    processor, model, device = (
        load_generator()
    )

    try:
        while (
            valid_created
            < valid_target
        ):
            combo = (
                choose_generation_combination(
                    state,
                    cycle_task_valid,
                    cycle_stratum_valid,
                    blocked,
                )
            )

            if combo is None:
                break

            (
                domain,
                task,
                stratum,
            ) = combo

            source, source_score = (
                choose_source(
                    domain,
                    task,
                    stratum,
                    state[
                        "by_article"
                    ],
                    cycle_article_reservations,
                )
            )

            if source is None:
                blocked.add(
                    combo
                )

                print(
                    "BLOCKED source combination:",
                    domain,
                    task,
                    stratum,
                    flush=True,
                )

                continue

            unit_id = source_unit_id_of(
                source
            )

            pmcid = pmcid_of(
                source
            )

            generation_key = (
                domain,
                task,
                unit_id,
            )

            used_generation_keys.add(
                generation_key
            )

            attempt_number = (
                len(
                    attempts
                )
                + 1
            )

            parsed, raw = generate_one(
                processor,
                model,
                device,
                domain,
                task,
                source,
            )

            attempt = {
                "attempt_id":
                    (
                        f"pmc-prod-attempt-"
                        f"{attempt_number:07d}"
                    ),

                "created_utc":
                    utc_now(),

                "domain":
                    domain,

                "task_type":
                    task,

                "stratum":
                    stratum,

                "pmcid":
                    pmcid,

                "source_unit_id":
                    unit_id,

                "source_priority_score":
                    source_score,

                "max_new_tokens":
                    GENERATION_BUDGETS[
                        task
                    ],
            }

            if parsed is None:
                attempt[
                    "status"
                ] = "invalid_json"

                attempt[
                    "raw"
                ] = raw

                attempts.append(
                    attempt
                )

                append_jsonl(
                    ATTEMPTS_OUT,
                    attempt,
                )

                print(
                    f"GEN INVALID_JSON "
                    f"{domain}/{task}/{stratum} "
                    f"accepted={total_accepted}/"
                    f"{TOTAL_ACCEPTED_TARGET}",
                    flush=True,
                )

                continue

            if (
                parsed.get(
                    "skip"
                )
                is True
            ):
                attempt[
                    "status"
                ] = "source_task_skip"

                attempt[
                    "skip_reason"
                ] = parsed.get(
                    "reason"
                )

                attempts.append(
                    attempt
                )

                append_jsonl(
                    ATTEMPTS_OUT,
                    attempt,
                )

                print(
                    f"GEN SKIP "
                    f"{domain}/{task}/{stratum}",
                    flush=True,
                )

                continue

            question = parsed.get(
                "question"
            )

            answer = parsed.get(
                "answer"
            )

            if (
                not isinstance(
                    question,
                    str,
                )
                or not question.strip()
                or not isinstance(
                    answer,
                    str,
                )
                or not answer.strip()
            ):
                attempt[
                    "status"
                ] = "invalid_schema"

                attempt[
                    "raw"
                ] = raw

                attempts.append(
                    attempt
                )

                append_jsonl(
                    ATTEMPTS_OUT,
                    attempt,
                )

                print(
                    f"GEN INVALID_SCHEMA "
                    f"{domain}/{task}/{stratum}",
                    flush=True,
                )

                continue

            candidate_number = (
                len(
                    candidates
                )
                + 1
            )

            candidate_id = (
                f"prod-{candidate_number:07d}"
            )

            candidate = {
                "candidate_id":
                    candidate_id,

                "created_utc":
                    utc_now(),

                "domain":
                    domain,

                "task_type":
                    task,

                "stratum":
                    stratum,

                "pmcid":
                    pmcid,

                "doi":
                    pick(
                        source,
                        "doi",
                    ),

                "pmid":
                    pick(
                        source,
                        "pmid",
                    ),

                "license":
                    pick(
                        source,
                        "license",
                    ),

                "source_unit_id":
                    unit_id,

                "source_chunk_ids":
                    pick(
                        source,
                        "source_chunk_ids",
                        "chunk_ids",
                        default=[],
                    ),

                "source_context":
                    context_of(
                        source
                    ),

                "question":
                    question.strip(),

                "answer":
                    answer.strip(),

                "generator":
                    "Qwen/Qwen3.8-27B",

                "generation_contract":
                    "V4/V4.1-frozen",

                "generation_max_new_tokens":
                    GENERATION_BUDGETS[
                        task
                    ],
            }

            candidates.append(
                candidate
            )

            candidate_by_id[
                candidate_id
            ] = candidate

            append_jsonl(
                CANDIDATES_OUT,
                candidate,
            )

            attempt[
                "status"
            ] = "valid_candidate"

            attempt[
                "candidate_id"
            ] = candidate_id

            attempts.append(
                attempt
            )

            append_jsonl(
                ATTEMPTS_OUT,
                attempt,
            )

            valid_created += 1

            cycle_task_valid[
                (
                    domain,
                    task,
                )
            ] += 1

            cycle_stratum_valid[
                (
                    domain,
                    stratum,
                )
            ] += 1

            cycle_article_reservations[
                pmcid
            ] += 1

            print(
                f"GEN VALID "
                f"{valid_created}/{valid_target} "
                f"{domain}/{task}/{stratum} "
                f"{candidate_id}",
                flush=True,
            )

    finally:
        unload_generator(
            processor,
            model,
        )

    return valid_created


###############################################################################
# VERIFY ALL PENDING CANDIDATES
###############################################################################

def verify_pending():
    pending = [
        row
        for row in candidates
        if (
            row[
                "candidate_id"
            ]
            not in verification_by_id
        )
    ]

    if not pending:
        return 0

    print()
    print(
        "Pending candidates:",
        len(
            pending
        ),
    )

    tokenizer, model, device = (
        load_verifier()
    )

    verified_now = 0

    try:
        for candidate in pending:
            candidate_id = candidate[
                "candidate_id"
            ]

            state = derive_accepted_state()

            total_accepted = len(
                state[
                    "candidates"
                ]
            )

            if (
                total_accepted
                >= TOTAL_ACCEPTED_TARGET
            ):
                break

            domain = candidate[
                "domain"
            ]

            task = candidate[
                "task_type"
            ]

            stratum = candidate[
                "stratum"
            ]

            pmcid = candidate[
                "pmcid"
            ]

            reasons = []

            ###################################################################
            # QUOTA / ARTICLE PRECHECK
            ###################################################################

            if (
                state[
                    "by_task"
                ][
                    (
                        domain,
                        task,
                    )
                ]
                >= TASK_QUOTAS[
                    domain
                ][
                    task
                ]
            ):
                reasons.append(
                    "task_quota_already_full"
                )

            if (
                state[
                    "by_stratum"
                ][
                    (
                        domain,
                        stratum,
                    )
                ]
                >= STRATUM_QUOTAS[
                    domain
                ][
                    stratum
                ]
            ):
                reasons.append(
                    "stratum_quota_already_full"
                )

            if (
                state[
                    "by_article"
                ][
                    pmcid
                ]
                >= MAX_ACCEPTED_PER_ARTICLE
            ):
                reasons.append(
                    "article_acceptance_cap"
                )

            ###################################################################
            # EXACT / NEAR DUPLICATE PRECHECK
            ###################################################################

            max_similarity = 0.0

            normalized = normalize_question(
                candidate[
                    "question"
                ]
            )

            for previous in (
                state[
                    "questions_by_task"
                ][
                    task
                ]
            ):
                if (
                    normalize_question(
                        previous
                    )
                    == normalized
                ):
                    max_similarity = 1.0

                    reasons.append(
                        "duplicate_question"
                    )

                    break

                similarity = (
                    question_similarity(
                        candidate[
                            "question"
                        ],
                        previous,
                    )
                )

                if (
                    similarity
                    > max_similarity
                ):
                    max_similarity = (
                        similarity
                    )

                if (
                    similarity
                    >= QUESTION_NEAR_DUPLICATE_THRESHOLD
                ):
                    reasons.append(
                        "near_duplicate_question"
                    )

                    break

            ###################################################################
            # If precheck failed, record without wasting two model calls.
            ###################################################################

            if reasons:
                verification = {
                    "candidate_id":
                        candidate_id,

                    "verified_utc":
                        utc_now(),

                    "accepted":
                        False,

                    "rejection_reasons":
                        reasons,

                    "grounding":
                        None,

                    "task_audit":
                        None,

                    "max_question_similarity":
                        max_similarity,

                    "verifier_skipped":
                        True,
                }

                verification_by_id[
                    candidate_id
                ] = verification

                verifications.append(
                    verification
                )

                append_jsonl(
                    VERIFY_OUT,
                    verification,
                )

                verified_now += 1

                print(
                    f"VERIFY REJECT {candidate_id} "
                    f"{','.join(reasons)}",
                    flush=True,
                )

                continue

            ###################################################################
            # V3 GROUNDING
            ###################################################################

            (
                grounding,
                grounding_raw,
            ) = grounding_audit(
                tokenizer,
                model,
                device,
                candidate,
            )

            ###################################################################
            # V3 TASK CLASSIFICATION
            ###################################################################

            (
                task_result,
                task_raw,
            ) = task_audit(
                tokenizer,
                model,
                device,
                candidate,
            )

            if (
                grounding is None
                or task_result is None
            ):
                reasons.append(
                    "verifier_invalid_json"
                )

            else:
                unsupported = (
                    grounding.get(
                        "unsupported_claims"
                    )
                )

                answer_quality = (
                    grounding.get(
                        "answer_quality"
                    )
                )

                question_quality = (
                    task_result.get(
                        "question_quality"
                    )
                )

                if (
                    grounding.get(
                        "fully_grounded"
                    )
                    is not True
                ):
                    reasons.append(
                        "ungrounded_answer"
                    )

                if (
                    not isinstance(
                        unsupported,
                        list,
                    )
                ):
                    reasons.append(
                        "invalid_unsupported_claims_field"
                    )

                elif unsupported:
                    reasons.append(
                        "unsupported_claims"
                    )

                if (
                    not isinstance(
                        answer_quality,
                        int,
                    )
                    or answer_quality
                    < MIN_ANSWER_QUALITY
                ):
                    reasons.append(
                        "answer_quality_below_threshold"
                    )

                if (
                    task_result.get(
                        "assigned_task_is_good_match"
                    )
                    is not True
                ):
                    reasons.append(
                        "assigned_task_mismatch"
                    )

                if (
                    task_result.get(
                        "best_task_type"
                    )
                    != task
                ):
                    reasons.append(
                        "best_task_differs_from_assigned"
                    )

                if (
                    not isinstance(
                        question_quality,
                        int,
                    )
                    or question_quality
                    < MIN_QUESTION_QUALITY
                ):
                    reasons.append(
                        "question_quality_below_threshold"
                    )

            accepted = (
                len(
                    reasons
                )
                == 0
            )

            verification = {
                "candidate_id":
                    candidate_id,

                "verified_utc":
                    utc_now(),

                "accepted":
                    accepted,

                "rejection_reasons":
                    reasons,

                "grounding":
                    grounding,

                "task_audit":
                    task_result,

                "max_question_similarity":
                    max_similarity,

                "grounding_raw":
                    grounding_raw,

                "task_raw":
                    task_raw,

                "verifier_skipped":
                    False,
            }

            ###################################################################
            # AUTHORITATIVE APPEND
            ###################################################################

            verification_by_id[
                candidate_id
            ] = verification

            verifications.append(
                verification
            )

            append_jsonl(
                VERIFY_OUT,
                verification,
            )

            verified_now += 1

            if accepted:
                new_total = (
                    total_accepted
                    + 1
                )

                print(
                    f"VERIFY ACCEPT {candidate_id} "
                    f"{new_total}/"
                    f"{TOTAL_ACCEPTED_TARGET} "
                    f"{domain}/{task}/{stratum}",
                    flush=True,
                )

            else:
                print(
                    f"VERIFY REJECT {candidate_id} "
                    f"{','.join(reasons)}",
                    flush=True,
                )

            ###################################################################
            # PERIODIC DURABLE MATERIALIZATION
            ###################################################################

            if (
                verified_now
                % 25
                == 0
            ):
                materialize_outputs()

                current_progress()

    finally:
        unload_verifier(
            tokenizer,
            model,
        )

    materialize_outputs()

    current_progress()

    return verified_now


###############################################################################
# FINAL REPORT BUILDER
###############################################################################

def build_final_report():
    state = derive_accepted_state()

    accepted_total = len(
        state[
            "candidates"
        ]
    )

    attempt_status = Counter(
        row.get(
            "status",
            "unknown",
        )
        for row in attempts
    )

    rejection_reasons = Counter()

    accepted_quality = Counter()

    verifier_calls = 0

    verifier_skips = 0

    for verification in (
        verification_by_id.values()
    ):
        if verification.get(
            "verifier_skipped"
        ):
            verifier_skips += 1

        else:
            verifier_calls += 2

        if not verification.get(
            "accepted"
        ):
            for reason in verification.get(
                "rejection_reasons",
                []
            ):
                rejection_reasons[
                    reason
                ] += 1

        else:
            grounding = verification[
                "grounding"
            ]

            task_result = verification[
                "task_audit"
            ]

            score = (
                grounding[
                    "answer_quality"
                ]
                + task_result[
                    "question_quality"
                ]
            ) / 10.0

            accepted_quality[
                score
            ] += 1

    report = {
        "name":
            "AxonAI PMC Batch 002 Production",

        "config_hash":
            CONFIG_HASH,

        "completed_utc":
            utc_now(),

        "target":
            TOTAL_ACCEPTED_TARGET,

        "accepted":
            accepted_total,

        "complete":
            (
                accepted_total
                == TOTAL_ACCEPTED_TARGET
            ),

        "generation_attempts":
            len(
                attempts
            ),

        "valid_candidates":
            len(
                candidates
            ),

        "verified_candidates":
            len(
                verification_by_id
            ),

        "estimated_verifier_model_calls":
            verifier_calls,

        "verifier_precheck_skips":
            verifier_skips,

        "generation_status":
            dict(
                attempt_status
            ),

        "rejection_reasons":
            dict(
                rejection_reasons
            ),

        "quality_distribution": {
            str(
                key
            ):
                value

            for key, value
            in sorted(
                accepted_quality.items()
            )
        },

        "accepted_by_domain": {
            domain:
                state[
                    "by_domain"
                ][
                    domain
                ]

            for domain
            in DOMAIN_TARGETS
        },

        "accepted_by_task": {
            domain: {
                task:
                    state[
                        "by_task"
                    ][
                        (
                            domain,
                            task,
                        )
                    ]

                for task
                in TASK_QUOTAS[
                    domain
                ]
            }

            for domain
            in DOMAIN_TARGETS
        },

        "accepted_by_stratum": {
            domain: {
                stratum:
                    state[
                        "by_stratum"
                    ][
                        (
                            domain,
                            stratum,
                        )
                    ]

                for stratum
                in STRATUM_QUOTAS[
                    domain
                ]
            }

            for domain
            in DOMAIN_TARGETS
        },

        "max_accepted_per_article":
            max(
                state[
                    "by_article"
                ].values(),
                default=0,
            ),

        "unique_accepted_articles":
            len(
                state[
                    "by_article"
                ]
            ),

        "final_sft":
            str(
                FINAL_SFT
            ),

        "generator":
            "Qwen/Qwen3.8-27B",

        "verifier":
            "Qwen/Qwen3-32B-V3-two-stage",

        "generation_contract":
            "V4/V4.1-frozen",

        "thinking_disabled":
            True,
    }

    write_json_atomic(
        REPORT_OUT,
        report,
    )

    return report


###############################################################################
# MAIN RESUMABLE PRODUCTION LOOP
###############################################################################

cycle = 0

started = time.time()


while True:
    cycle += 1

    state = derive_accepted_state()

    accepted_total = len(
        state[
            "candidates"
        ]
    )

    if (
        accepted_total
        >= TOTAL_ACCEPTED_TARGET
    ):
        break

    print()
    print("#" * 88)
    print(
        f"PRODUCTION CYCLE {cycle}"
    )
    print("#" * 88)

    print(
        "Accepted:",
        accepted_total,
        "/",
        TOTAL_ACCEPTED_TARGET,
    )

    print(
        "Remaining:",
        TOTAL_ACCEPTED_TARGET
        - accepted_total,
    )

    ###########################################################################
    # Always verify durable pending candidates first on resume.
    ###########################################################################

    pending_count = sum(
        row[
            "candidate_id"
        ]
        not in verification_by_id
        for row in candidates
    )

    if pending_count:
        print(
            "Found pending candidates from previous generation:",
            pending_count,
        )

        verify_pending()

        state = derive_accepted_state()

        if (
            len(
                state[
                    "candidates"
                ]
            )
            >= TOTAL_ACCEPTED_TARGET
        ):
            break

    ###########################################################################
    # Generate next valid candidate batch.
    ###########################################################################

    generated = generate_batch()

    if generated == 0:
        raise RuntimeError(
            "Unable to generate any additional valid candidates while "
            "production quotas remain. Inspect blocked source/task/stratum "
            "combinations before continuing."
        )

    ###########################################################################
    # Verify generated candidates.
    ###########################################################################

    verify_pending()

    ###########################################################################
    # Materialize and print cycle progress.
    ###########################################################################

    materialize_outputs()

    progress = current_progress()

    elapsed_hours = (
        time.time()
        - started
    ) / 3600.0

    print()
    print(
        "=== CYCLE PROGRESS ==="
    )

    print(
        "Accepted:",
        progress[
            "accepted"
        ],
        "/",
        TOTAL_ACCEPTED_TARGET,
    )

    print(
        "Remaining:",
        progress[
            "remaining"
        ],
    )

    print(
        "Generation attempts:",
        len(
            attempts
        ),
    )

    print(
        "Valid candidates:",
        len(
            candidates
        ),
    )

    print(
        "Verified candidates:",
        len(
            verification_by_id
        ),
    )

    print(
        "Elapsed this invocation:",
        f"{elapsed_hours:.2f} h",
    )


###############################################################################
# FINAL MATERIALIZATION
###############################################################################

materialize_outputs()

progress = current_progress()

report = build_final_report()


###############################################################################
# HARD PRODUCTION ASSERTIONS
###############################################################################

final_rows = read_jsonl(
    FINAL_SFT
)

assert (
    len(
        final_rows
    )
    == TOTAL_ACCEPTED_TARGET
)

assert (
    report[
        "accepted"
    ]
    == TOTAL_ACCEPTED_TARGET
)

assert (
    report[
        "accepted_by_domain"
    ]
    == DOMAIN_TARGETS
)


for domain in DOMAIN_TARGETS:

    assert (
        report[
            "accepted_by_task"
        ][
            domain
        ]
        == TASK_QUOTAS[
            domain
        ]
    )

    assert (
        report[
            "accepted_by_stratum"
        ][
            domain
        ]
        == STRATUM_QUOTAS[
            domain
        ]
    )


###############################################################################
# UNIQUE SFT IDS
###############################################################################

ids = [
    row[
        "id"
    ]
    for row in final_rows
]

assert (
    len(
        ids
    )
    == len(
        set(
            ids
        )
    )
)


###############################################################################
# ACCEPTED DATA VALIDITY
###############################################################################

pmcid_counts = Counter()

normalized_questions = set()

source_units = set()


for row in final_rows:

    assert (
        row[
            "license"
        ]
        in {
            "CC-BY",
            "CC0",
        }
    )

    assert (
        row[
            "provenance"
        ][
            "split"
        ]
        == "train"
    )

    verification = row[
        "verification"
    ]

    assert (
        verification[
            "fully_grounded"
        ]
        is True
    )

    assert (
        verification[
            "unsupported_claims"
        ]
        == []
    )

    assert (
        verification[
            "answer_quality"
        ]
        >= MIN_ANSWER_QUALITY
    )

    assert (
        verification[
            "assigned_task_is_good_match"
        ]
        is True
    )

    assert (
        verification[
            "best_task_type"
        ]
        == row[
            "task_type"
        ]
    )

    assert (
        verification[
            "question_quality"
        ]
        >= MIN_QUESTION_QUALITY
    )

    pmcid = row[
        "provenance"
    ][
        "pmcid"
    ]

    pmcid_counts[
        pmcid
    ] += 1

    if article_split:
        assert (
            article_split.get(
                pmcid
            )
            == "train"
        )

    normalized_question = (
        normalize_question(
            row[
                "messages"
            ][
                1
            ][
                "content"
            ]
        )
    )

    assert (
        normalized_question
        not in normalized_questions
    )

    normalized_questions.add(
        normalized_question
    )

    source_units.add(
        row[
            "provenance"
        ][
            "source_unit_id"
        ]
    )


assert (
    max(
        pmcid_counts.values(),
        default=0,
    )
    <= MAX_ACCEPTED_PER_ARTICLE
)


###############################################################################
# FINAL OUTPUT
###############################################################################

print()
print("=" * 88)
print("AXONAI PMC BATCH 002 PRODUCTION COMPLETE")
print("=" * 88)

print(
    "Accepted SFT examples:",
    len(
        final_rows
    ),
)

print(
    "Generation attempts:",
    len(
        attempts
    ),
)

print(
    "Valid candidates:",
    len(
        candidates
    ),
)

print(
    "Verified candidates:",
    len(
        verification_by_id
    ),
)

print(
    "Unique accepted articles:",
    len(
        pmcid_counts
    ),
)

print(
    "Unique source units:",
    len(
        source_units
    ),
)

print(
    "Maximum accepted/article:",
    max(
        pmcid_counts.values(),
        default=0,
    ),
)

print()
print(
    "=== DOMAIN COUNTS ==="
)

for domain in DOMAIN_TARGETS:
    print(
        f"{domain:28s} "
        f"{report['accepted_by_domain'][domain]:4d}/"
        f"{DOMAIN_TARGETS[domain]:4d}"
    )


print()
print(
    "=== BIOMEDICAL TASK COUNTS ==="
)

for task, target in (
    TASK_QUOTAS[
        "biomedical"
    ].items()
):
    value = (
        report[
            "accepted_by_task"
        ][
            "biomedical"
        ][
            task
        ]
    )

    print(
        f"{task:28s} "
        f"{value:4d}/"
        f"{target:4d}"
    )


print()
print(
    "=== LITERATURE REASONING ==="
)

print(
    "literature_reasoning           "
    f"{report['accepted_by_task']['literature_reasoning']['literature_reasoning']:4d}/"
    f"{TASK_QUOTAS['literature_reasoning']['literature_reasoning']:4d}"
)


print()
print(
    "=== GENERATION STATUS ==="
)

for status, count in sorted(
    report[
        "generation_status"
    ].items()
):
    print(
        f"{status:28s} "
        f"{count}"
    )


print()
print(
    "=== VERIFIER REJECTION REASONS ==="
)

for reason, count in sorted(
    report[
        "rejection_reasons"
    ].items(),
    key=lambda item: (
        -item[1],
        item[0],
    ),
):
    print(
        f"{reason:36s} "
        f"{count}"
    )


print()
print(
    "Final SFT:"
)

print(
    FINAL_SFT
)

print()
print(
    "Detailed accepted:"
)

print(
    ACCEPTED_DETAIL_OUT
)

print()
print(
    "Report:"
)

print(
    REPORT_OUT
)

print()
print(
    "AxonAI PMC Batch 002 7,000-example production corpus: VALID"
)
