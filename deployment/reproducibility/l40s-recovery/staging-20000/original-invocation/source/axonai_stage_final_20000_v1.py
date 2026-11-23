#!/usr/bin/env python3
"""AxonAI-v1 final 20,000 SFT staging + read-only input audit.

No original file, accepted checkpoint or SQLite database is modified.
Builds a NEW versioned staging dataset and reports conservative structural,
provenance, licence, duplicate and concentration checks. No automatic final
training-corpus freeze or fine-tuning is performed.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import os
import re
import tarfile
import unicodedata
from pathlib import Path

VERSION = "axonai-v1-final-20000-staging-audit-v1"
TARGET = "axonai-20k-slot-06926"
EXPECTED_BASELINE = (5800, 7000, 7172, 28, 11, 1)
EXPECTED_FROZEN11_SHA = "504c86a85b3dd047c27faf1c39408c1ae058f0ce24d15d469f2330a8dc3fa4e8"
EXPECTED_SINGLE_SHA = "2bb726b3e2823e03a26df65912f13b1ae3e1ce38dd9146440b11a75c41de1973"
CANONICAL = {"id", "messages", "domain", "task_type", "source", "license", "quality_score", "provenance", "verification"}


def need(test, msg):
    if not test:
        raise RuntimeError(msg)


def sha_path(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def sha_bytes(content):
    return hashlib.sha256(content).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def jsonlines(path):
    with path.open("r", encoding="utf-8") as f:
        for no, line in enumerate(f,1):
            if line.strip():
                obj=json.loads(line)
                need(type(obj)==dict,f"{path}:{no}: JSON object expected")
                yield obj


def member_data(path,suffix):
    with tarfile.open(path,"r:gz") as tf:
        matches=[m for m in tf.getmembers() if m.isfile() and m.name.endswith(suffix)
                 and not Path(m.name).is_absolute() and ".." not in Path(m.name).parts]
        need(len(matches)==1, f"Archive {path.name}: expected exactly one {suffix}, got {len(matches)}")
        f=tf.extractfile(matches[0])
        need(f is not None,f"Unreadable {suffix}")
        return f.read()


def archive_json(path,suffix):
    return json.loads(member_data(path,suffix))


def archive_rows(path,suffix):
    return [json.loads(l) for l in member_data(path,suffix).splitlines() if l.strip()]


def verify_archive(path):
    with tarfile.open(path,"r:gz") as tf:
        file_map={Path(m.name).name:m for m in tf.getmembers() if m.isfile()}
        if "SHA256SUMS" not in file_map:
            raise RuntimeError("Archive has no SHA256SUMS: "+path.name)
        checks=tf.extractfile(file_map["SHA256SUMS"]).read().decode().splitlines()
        for item in checks:
            expected,name=item.strip().split(None,1)
            name=Path(name.strip()).name
            need(name in file_map,"Missing archive checksum target "+name)
            data=tf.extractfile(file_map[name]).read()
            need(sha_bytes(data)==expected,"Bad archived checksum: "+name)


def question_answer(row):
    msgs=row.get("messages")
    need(isinstance(msgs,list) and len(msgs)==3,"Expected system/user/assistant messages")
    need([m.get("role") for m in msgs]==["system","user","assistant"],
         "Unexpected message roles")
    for m in msgs:
        need(isinstance(m.get("content"),str) and m["content"].strip(),"Empty/nontext message content")
    return normalize(msgs[1]["content"]),normalize(msgs[2]["content"])


def validate_record(row,origin):
    need(set(row)==CANONICAL,f"{origin}: canonical schema mismatch")
    need(isinstance(row["id"],str) and row["id"],"Missing record id")
    need(isinstance(row["domain"],str) and row["domain"],"Missing domain")
    need(isinstance(row["task_type"],str) and row["task_type"],"Missing task type")
    need(isinstance(row["provenance"],dict),"Missing provenance")
    need(row["provenance"].get("split")=="train",f"{origin}: not train split")
    v=row["verification"]
    need(isinstance(v,dict) and v.get("fully_grounded") is True
         and v.get("unsupported_claims")==[]
         and v.get("assigned_task_is_good_match") is True
         and v.get("question_quality",0)>=4
         and v.get("answer_quality",0)>=4,
         f"{origin}: verification metadata does not pass")
    return question_answer(row)


def atomic_json(path,obj):
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        json.dump(obj,f,indent=2,ensure_ascii=False)
        f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def atomic_jsonl(path,records):
    tmp=path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def get_shingles(q):
    words=re.findall(r"\w+",q)
    k=4 if len(words)>=8 else 3 if len(words)>=3 else 2
    if len(words)<2:
        return {q}
    return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}


def near_duplicates(qs, ids, max_candidates=250000):
    """Candidate-generation on word shingles + exact SequenceMatcher >= .90.

    Conservative review screen, *not* proof of no semantic paraphrases.
    Excludes very high-frequency generic shingles to avoid quadratic fanout.
    """
    shingles=[get_shingles(x) for x in qs]
    inverted=collections.defaultdict(list)
    for i,items in enumerate(shingles):
        for item in items:
            inverted[item].append(i)
    possible=collections.Counter()
    skipped_generic=0
    for shingle, postings in inverted.items():
        if len(postings)>120:
            skipped_generic+=1
            continue
        for n,i in enumerate(postings):
            for j in postings[:n]:
                possible[(j,i)]+=1
    need(len(possible)<=max_candidates,
         f"Near-duplicate candidate volume too large ({len(possible)}); audit must be partitioned")
    hits=[]
    for (i,j),inter in possible.items():
        total=len(shingles[i])+len(shingles[j])
        if total and 2*inter/total<0.48:
            continue
        ratio=difflib.SequenceMatcher(None,qs[i],qs[j],autojunk=False).ratio()
        if ratio>=0.90:
            hits.append({"id_1":ids[i],"id_2":ids[j],"ratio":round(ratio,6),
                         "question_1":qs[i][:1000],"question_2":qs[j][:1000]})
    hits.sort(key=lambda x:(-x["ratio"],x["id_1"],x["id_2"]))
    return hits,{"method":"4-word (or shorter-question 3-word) shingle candidate screen + SequenceMatcher 0.90",
                 "pairs_considered":len(possible),"skipped_generic_shingles":skipped_generic,
                 "is_exhaustive_semantic_paraphrase_check":False}


def main():
    ap=argparse.ArgumentParser()
    for key in ("batch001","pmc","production","recovered28_archive","supersession_plan",
                "previous11_archive","single_archive","source12_archive","source1_archive",
                "output_dir"):
        ap.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    ap.add_argument("--skip-near-screen",action="store_true")
    a=ap.parse_args()
    for key,p in vars(a).items():
        if key not in ("output_dir","skip_near_screen"):
            need(p.is_file(),"Missing input "+str(p))
    need(not a.output_dir.exists(),"Refusing to overwrite existing staging directory")
    for path in (a.recovered28_archive,a.supersession_plan,a.previous11_archive,
                 a.single_archive,a.source12_archive,a.source1_archive):
        verify_archive(path)
    p11_summary=archive_json(a.previous11_archive,"summary.json")
    p1_summary=archive_json(a.single_archive,"summary.json")
    src12_audit=archive_json(a.source12_archive,"hard-audit-report.json")
    src1_audit=archive_json(a.source1_archive,"audit.json")
    plan_summary=archive_json(a.supersession_plan,"summary.json")
    need(p11_summary["provisionally_accepted"]==11
         and p11_summary["unresolved_slots"]==[TARGET],"Prior 11 not preserved")
    need(p1_summary["status"]=="ACCEPTED" and p1_summary["provisionally_accepted"]==1
         and p1_summary["slot_id"]==TARGET,"Single slot not accepted")
    need(src12_audit["status"]=="FROZEN_12_DUPLICATE_REPLACEMENT_SOURCES_PASS"
         and src12_audit["source_manifest_sha256"]==EXPECTED_FROZEN11_SHA,
         "12-source freeze mismatch")
    need(src1_audit["status"]=="SINGLE_SOURCE_FREEZE_PASS"
         and src1_audit["source_manifest_sha256"]==EXPECTED_SINGLE_SHA,
         "Single source freeze mismatch")
    need(plan_summary["victim_production_records"]==12,"Unexpected supersession plan")
    source12=archive_rows(a.source12_archive,"frozen-12-duplicate-replacement-sources.jsonl")
    source1=archive_rows(a.source1_archive,"frozen-06926-replacement-source.jsonl")
    need(len(source12)==12 and len(source1)==1,"Unexpected frozen-source counts")
    source_map={r["slot_id"]:r for r in source12}
    need(len(source_map)==12 and source1[0]["slot_id"]==TARGET,"Frozen slot identities mismatch")
    source_map[TARGET]=source1[0]
    victims=archive_rows(a.supersession_plan,"12-proposed-production-supersessions.jsonl")
    victim_map={r["replacement_slot_id"]:r for r in victims}
    need(len(victim_map)==12 and set(source_map)==set(victim_map),"Slot coverage mismatch")
    removed_ids={r["supersede_record_id"] for r in victims}
    need(len(removed_ids)==12,"Supersession ID collision")
    old_p=archive_rows(a.recovered28_archive,"provisional-recovered-28.jsonl")
    eleven=archive_rows(a.previous11_archive,"provisional-duplicate-replacements-12.jsonl")
    one=archive_rows(a.single_archive,"provisional-single-06926.jsonl")
    need(len(old_p)==28 and len(eleven)==11 and len(one)==1,"Recovery input count mismatch")
    correction=eleven+one
    need(len({r["provenance"]["frozen_slot_id"] for r in correction})==12,
         "Duplicate replacement slot")
    need({r["provenance"]["frozen_slot_id"] for r in correction}==set(victim_map),
         "Incorrect replacement slot coverage")
    for row in correction:
        prov=row["provenance"]
        sid=prov["frozen_slot_id"]
        src=source_map[sid]
        expected=victim_map[sid]
        need(row["domain"]==src["domain"]==expected["domain"]
             and row["task_type"]==src["task_type"]==expected["task_type"],
             "Domain/task changed: "+sid)
        need(prov.get("source_unit_id")==src["source_unit_id"]
             and prov.get("source_sha256")==src["source_sha256"]
             and prov.get("source_family")==src["source_family"],
             "Frozen source evidence changed: "+sid)
        need(prov.get("recovery_manifest_sha256")==(
            EXPECTED_SINGLE_SHA if sid==TARGET else EXPECTED_FROZEN11_SHA),
             "Wrong recovery manifest hash: "+sid)
        need(prov.get("supersedes_production_record_id")==expected["supersede_record_id"]
             and prov.get("retained_duplicate_record_id")==expected["retain_record_id"],
             "Original/retained record identity mismatch: "+sid)
        need(row["license"]==src["license"],"Replacement licence mismatch: "+sid)
        validate_record(row,"replacement:"+sid)
    print("RECOVERIES_OK: exactly 28 inclusive historical + 11 preserved + 1 new; 12 supersession identities pinned",flush=True)

    dataset=[]
    origin_rows=collections.Counter()
    removed=[]
    original_task=collections.Counter()
    for name,path,expected in [("batch001",a.batch001,5800),("pmc",a.pmc,7000),
                               ("production",a.production,7172)]:
        count=0
        for row in jsonlines(path):
            count+=1
            original_task[(row["domain"],row["task_type"])]+=1
            if name=="production" and row["id"] in removed_ids:
                removed.append(row)
                continue
            dataset.append((name,row))
            origin_rows[name]+=1
        need(count==expected,f"{name} expected {expected}, got {count}")
    need(len(removed)==12 and {x["id"] for x in removed}==removed_ids,
         "Supersession records absent or repeated")
    for row in old_p:
        dataset.append(("historical_recoveries_28",row)); origin_rows["historical_recoveries_28"]+=1
        original_task[(row["domain"],row["task_type"])]+=1
    for row in correction:
        dataset.append(("duplicate_replacements_12",row));origin_rows["duplicate_replacements_12"]+=1
    need(origin_rows==collections.Counter({
        "batch001":5800,"pmc":7000,"production":7160,
        "historical_recoveries_28":28,"duplicate_replacements_12":12}),
        "Final dataset component counts differ")
    need(len(dataset)==20000,"Staging must contain exactly 20000 records")
    final_task=collections.Counter((r["domain"],r["task_type"]) for _,r in dataset)
    need(final_task==original_task,"Original 20k domain/task quotas not preserved")

    ids=set(); qs=set(); qs_list=[]; ids_list=[]
    lengths=collections.Counter();licence=collections.Counter();domains=collections.Counter()
    sources=collections.Counter();proof_sources=collections.Counter(); weak_metadata=[]
    for origin,row in dataset:
        q,_=validate_record(row,origin)
        need(row["id"] not in ids,"Duplicate record id: "+row["id"])
        need(q not in qs,"Exact duplicate user question: "+q[:140])
        ids.add(row["id"]);qs.add(q);qs_list.append(q);ids_list.append(row["id"])
        lengths[origin]+=1
        licence[str(row.get("license"))]+=1
        domains[row["domain"]]+=1
        proof_sources[row.get("source")]+=1
        p=row["provenance"]
        if p.get("source_unit_id"):
            sources[(origin,p.get("source_unit_id"))]+=1
        else:
            weak_metadata.append({"id":row["id"],"origin":origin,"issue":"missing source_unit_id"})
    print("STRUCTURAL_PASS: 20000 unique IDs and normalized questions; train split, verified, canonical",flush=True)
    near_flags=[]
    screen_details={"run":False,"note":"Opted out of preliminary near-duplicate candidate screening"}
    if not a.skip_near_screen:
        print("Starting conservative near-duplicate question screening (not semantic exhaustive)...",flush=True)
        near_flags,screen_details=near_duplicates(qs_list,ids_list)
        screen_details["run"]=True
    report={
        "version":VERSION,
        "status":("STAGING_COMPLETE_REVIEW_REQUIRED" if near_flags or weak_metadata
                  else "STAGING_PRELIMINARY_PASS_FINAL_REVIEW_REQUIRED"),
        "records":len(dataset),"unique_record_ids":len(ids),
        "unique_normalized_questions":len(qs),
        "origin_counts":dict(origin_rows),
        "superseded_production_records_excluded":12,
        "historic_recovery_archive_included_once":True,
        "duplicate_replacements_from_old_archive":11,
        "single_slot_replacement_from_new_archive":1,
        "domain_task_quotas_preserved":True,
        "domain_counts":dict(domains),
        "licence_value_counts":dict(licence),
        "source_issuer_counts":dict(proof_sources),
        "max_units_per_origin":{"|".join(map(str,k)):v for k,v in sources.most_common(10)},
        "near_duplicate_pairs_flagged":len(near_flags),
        "near_duplicate_method":screen_details,
        "provenance_missing_unit_records":weak_metadata[:100],
        "original_training_files_modified":False,
        "original_production_sqlite_modified":False,
        "ready_for_training":False,
        "next_gate":"INDEPENDENT_FINAL_HARD_AUDIT_AND_EXPLICIT_VERSIONED_FREEZE",
        "cautions":[
            "This is a versioned staging corpus, NOT a model-training release.",
            "The 2/unit and 12/document concentration caps were approved per generation batch; global concentrations require disclosure.",
            "This lexical near-duplicate screen is not an exhaustive semantic paraphrase search.",
            "Licences and provenance must be audited against primary sources prior to distribution.",
            "Final model-appropriate token length audit and representative difficult-case QA remain to be completed."
        ]
    }
    a.output_dir.mkdir(parents=True)
    staged=a.output_dir/"axonai-v1-20000-STAGING-NOT-FROZEN.jsonl"
    atomic_jsonl(staged,[r for _,r in dataset])
    report["staged_dataset_sha256"]=sha_path(staged)
    atomic_json(a.output_dir/"staging-audit-report.json",report)
    atomic_json(a.output_dir/"near-question-review.json",near_flags)
    atomic_json(a.output_dir/"superseded-production-record-ids.json",sorted(removed_ids))
    inputs={name:sha_path(getattr(a,name)) for name in (
        "batch001","pmc","production","recovered28_archive","supersession_plan",
        "previous11_archive","single_archive","source12_archive","source1_archive")}
    atomic_json(a.output_dir/"input-sha256.json",inputs)
    with (a.output_dir/"SHA256SUMS").open("w",encoding="utf-8") as f:
        for path in sorted(a.output_dir.iterdir()):
            if path.is_file() and path.name!="SHA256SUMS":
                f.write(sha_path(path)+"  "+path.name+"\n")
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print("STAGING_DIR:",a.output_dir,flush=True)

if __name__=="__main__":
    main()
