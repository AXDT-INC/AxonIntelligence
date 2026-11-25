"""Reconcile indexed publisher permission statements, keeping scope and uncertainty explicit."""
import argparse
from collections import Counter
import hashlib
import json
import re
import time
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from pathlib import Path

from fetch import CACHE, INVENTORY, REGISTER_SHA, text

OUT = Path("/attribution")
PUBLISHER = Path("/cache/crossref")
CC = re.compile(r"creativecommons\.org/((?:licenses/(?:by(?:-[a-z]+)*)|publicdomain/zero)/\d\.\d(?:/(?:igo|[a-z]{2})(?=/))?)", re.I)
REVIEWS = json.loads(Path(__file__).with_name("publisher-review.json").read_bytes())
ZERO = "https://creativecommons.org/publicdomain/zero/1.0/"
DATA = re.compile(r"(?:applies to|applies specifically to) the data (?:made available in|associated with) this article", re.I)
RESTRICT = re.compile(r"non[ -]?commercial|no[ -]?derivatives|share[ -]?alike|not used for commercial purposes|accepted manuscript", re.I)


def cc_urls(value):
    return {"https://creativecommons.org/"+s.lower()+"/" for s in CC.findall(value)}


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    tmp.replace(path)


def publisher_deposit(pmcid, doi, online):
    path = PUBLISHER / (pmcid+".json")
    if path.is_file():
        return read_json(path)
    if not online or not doi:
        return {}
    url = "https://api.crossref.org/works/" + quote(doi, safe="/")
    for delay in (1, 4, 12):
        time.sleep(delay)
        try:
            req = Request(url, headers={"User-Agent": "AxonAI-attribution/1"})
            with urlopen(req, timeout=20) as r:
                record = json.load(r)["message"]
            if record["DOI"].lower() != doi.lower():
                raise ValueError("Publisher DOI mismatch")
            entry = {"source": url, "licenses": record.get("license", []),
                     "authors": [a.get("name") or " ".join(filter(None, (a.get("family"),a.get("given")))) for a in record.get("author", [])]}
            write_json(path, entry)
            return entry
        except OSError as exc:
            if getattr(exc, "code", None) == 404:
                break
    return {}


def reconcile(row, meta, online):
    issues = []
    pmcid = row["pmcid"]
    out = {key:row[key] for key in ("pmcid","source_url","training_record_count")}
    out.update(article_license_url=None, data_waiver_url=None,
               historical_license_url=row["license_url"], changes=[])
    if not meta:
        out.update(authors=row["authors"], doi=row["doi"], title=row["title"], issues=["PMC metadata unavailable; bibliographic fields inherited"])
        return out
    assert meta["pmcid"] == pmcid
    out.update(pmc_version=meta["version"], metadata_source=meta["source"], retrieved=meta["retrieved"])
    permissions = [ET.fromstring(p) for p in meta["permissions"]]
    out["copyright_notices"] = [text(c) for p in permissions for c in p.findall("copyright-statement")]
    licenses = [l for p in permissions for l in p.findall("license")]
    paragraphs = [p for l in licenses for p in l.findall("license-p")]
    notice = " ".join(text(p) for p in paragraphs)
    out["permission_notice"] = notice
    all_urls = set().union(*(cc_urls(ET.tostring(l,encoding="unicode")) for l in licenses))
    pub_urls = set().union(*(cc_urls(ET.tostring(p, encoding="unicode")) for p in paragraphs))
    if DATA.search(notice) and ZERO in pub_urls:
        out["data_waiver_url"] = ZERO
        all_urls.discard(ZERO)
        pub_urls.discard(ZERO)
    candidate = next(iter(all_urls)) if len(all_urls)==1 else None
    version = candidate.rstrip("/").split("/")[-1] if candidate else ""
    explicit = candidate and (candidate in pub_urls or (version and re.search(r"(?<![0-9.])"+re.escape(version)+r"(?![0-9])",notice)))
    scope_warning = bool(RESTRICT.search(notice.replace("posting of the Accepted Manuscript", "posting of the manuscript")))
    multiple = len(licenses)!=1 and pmcid!="PMC13541222"
    ambiguous = not explicit or scope_warning or multiple
    authors = meta["authors"]
    if ambiguous or not authors or not all(authors):
        dep = publisher_deposit(pmcid, meta["doi"], online)
        if dep:
            out["publisher_deposit"] = dep["source"]
            # Only licensing for the version of record can corroborate the article grant.
            vor = set().union(*(cc_urls(x["URL"]) for x in dep["licenses"] if x.get("content-version")=="vor"))
            out["publisher_vor_cc_urls"] = sorted(vor)
            unspecified = set().union(*(cc_urls(x["URL"]) for x in dep["licenses"] if x.get("content-version")=="unspecified"))
            out["publisher_unspecified_cc_urls"] = sorted(unspecified)
            matching = vor=={candidate} or (not vor and unspecified=={candidate} and "attribution" in notice.lower())
            if matching and not multiple and not scope_warning:
                ambiguous = False
            elif vor and vor!={candidate}:
                ambiguous = True
            if not any(authors) and dep["authors"] and all(dep["authors"]):
                authors = dep["authors"]
    reviewed = REVIEWS.get(meta["version"])
    if reviewed:
        assert not meta["doi"] or meta["doi"].lower()==reviewed["doi"].lower()
        out["publisher_evidence"] = reviewed["source"]
        if "open_issue" in reviewed:
            ambiguous = True
        else:
            assert candidate == reviewed.get("pmc_annotation", reviewed["license"]) and not scope_warning
            if candidate != reviewed["license"]:
                out["pmc_textmining_license_url"] = candidate
            candidate = reviewed["license"]
            ambiguous = False
            out["publisher_review"] = reviewed["basis"]
            if "credit_line" in reviewed:
                out["credit_line"] = reviewed["credit_line"]
    if pmcid == "PMC13543777" and "in the public domain" in notice:
        out["article_rights_statement"] = "US public domain (publisher statement)"
        out["publisher_evidence"] = "https://www.cdc.gov/pcd/issues/2026/pdf/26_0004.pdf"
        out["changes"].append("article_license")
    elif ambiguous:
        issues.append((reviewed or {}).get("open_issue", "Article license or version requires review against the publisher's applicable version"))
        out["pmc_license_candidates"] = sorted(all_urls)
    else:
        out["article_license_url"] = candidate
    if (not authors or not all(authors)) and not out.get("credit_line"):
        issues.append("Incomplete named or group authorship")
    if meta["doi"] and row["doi"] and meta["doi"].lower()!=row["doi"].lower():
        issues.append("Historical and current DOI differ; check source identity")
    if not meta["title"]:
        issues.append("Title unavailable")
    out.update(doi=(reviewed or {}).get("doi") or meta["doi"] or row["doi"], title=meta["title"] or row["title"], authors=[a for a in authors if a])
    if out["authors"]!=row["authors"]:
        out["historical_author_credits"] = row["authors"]
        out["group_members"] = list(filter(None,meta.get("group_members",[])))
    for key in ("authors","title","doi"):
        if out[key]!=row[key]:
            out["changes"].append(key)
    if out["article_license_url"] and out["article_license_url"]!=row["license_url"]:
        out["changes"].append("article_license")
    if out["data_waiver_url"]:
        out["changes"].append("separate_data_waiver")
    out["issues"] = issues
    return out


def main(online):
    assert hashlib.sha256(INVENTORY.read_bytes()).hexdigest() == REGISTER_SHA
    rows = read_json(INVENTORY)["sources"]
    receipt_path = Path("/input/extraction.json")
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest()=="c16e15e0bb1228c4f4b29ab8e0f64339c71b7a8807247dc1db3ec3d947c62b4a"
    receipt = read_json(receipt_path)
    assert len({r["pmcid"] for r in rows})==len(rows)==4230
    assert receipt["id_coverage_exact"] and sum(r["training_record_count"] for r in rows)==receipt["record_count"]==10684
    assert receipt["extracted_jsonl_sha256"]=="641b43e32f3d69fa6fbf74754772e976777a26383c27de2692b90670c5dc5d07"
    if online:
        PUBLISHER.mkdir(exist_ok=True)
    journals = read_json(CACHE/"journal.json") if (CACHE/"journal.json").exists() else {}
    result = []
    for row in sorted(rows,key=lambda r:(r["license"]!="CC0",r["pmcid"])):
        path = CACHE / (row["pmcid"]+".json")
        entry = reconcile(row, read_json(path) if path.exists() else None, online)
        entry["journal_citation"] = journals.get(row["pmcid"])
        if not entry["journal_citation"]:
            entry["issues"].append("Missing journal citation")
        result.append(entry)
    write_json(OUT/"SOURCE_ATTRIBUTIONS.rc.json", dict(historical_register_sha256=REGISTER_SHA,
               extracted_training_sha256=receipt["extracted_jsonl_sha256"], training_records=10684,
               status="Release candidate. Bibliographic metadata only. Redistribution decisions pending.",
               methodology="Publisher-supplied PMC JATS article-level permissions and contributor lists; limited Crossref version-of-record corroboration. Unspecified data rights are unknown. Credit-line exceptions apply. Evidence describes the cited version; historical extraction did not pin article revision IDs.",
               source_transformation="Published literature informed scientific instruction/response examples used in supervised fine-tuning. The model packages contain no training examples.", sources=result))
    summary = dict(total=len(result), records=sum(r["training_record_count"] for r in result),
                   pmc_records=sum("pmc_version" in r for r in result),
                   article_terms=dict(Counter(r["article_license_url"] or r.get("article_rights_statement","unresolved") for r in result)),
                   historical_cc0=dict(Counter(r["article_license_url"] or "unresolved" for r in result if r["historical_license_url"]==ZERO)),
                   changes=dict(Counter(k for r in result for k in r["changes"])),
                   missing_authors_recovered=sum(not old["authors"] and bool(new["authors"]) for old,new in zip(sorted(rows,key=lambda r:r["pmcid"]),sorted(result,key=lambda r:r["pmcid"]))),
                   issues={r["pmcid"]:r["issues"] for r in result if r["issues"]})
    write_json(OUT/"RECONCILIATION.json",summary)
    print({k:v for k,v in summary.items() if k!="issues"}, "flagged",len(summary["issues"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publisher",action="store_true", help="Allow rate-limited Crossref lookup for ambiguous records")
    main(parser.parse_args().publisher)
