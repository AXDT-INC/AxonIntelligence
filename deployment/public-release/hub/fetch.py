"""Cache publisher-supplied JATS front matter from PMC EFetch, CC0 draft labels first."""
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from gzip import GzipFile
from pathlib import Path
from urllib.request import Request, urlopen

CACHE = Path("/cache/pmc")
INVENTORY = Path("/input/register.json")
REGISTER_SHA = "cbddcc47e0be80c8c24c44fb20ac012481a700819b217a595b7fc8007abbb340"
API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def text(node):
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def author(contrib):
    names = contrib.findall("name") + contrib.findall("name-alternatives/name")
    if names:
        name = next((n for n in names if n.get(LANG) == "en"), names[0])
        return " ".join(filter(None, [text(name.find(k)) for k in ("prefix", "surname", "given-names", "suffix")]))
    for tag in ("collab", "collab-alternatives/collab", "string-name", "anonymous"):
        node = contrib.find(tag)
        if node is not None:
            node = ET.fromstring(ET.tostring(node))
            for ref in list(node.findall("xref")) + list(node.findall("contrib-group")):
                node.remove(ref)
            return text(node) or ("Anonymous" if tag == "anonymous" else "")
    return ""


def fetch(ids):
    url = API + "?db=pmc&retmode=xml&tool=axonai_attribution&id=" + ",".join(i[3:] for i in ids)
    req = Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "AxonAI-attribution/1"})
    with urlopen(req, timeout=90) as response:
        stream = GzipFile(fileobj=response) if response.headers.get("Content-Encoding") == "gzip" else response
        for _, article in ET.iterparse(stream, events=("end",)):
            if article.tag != "article":
                continue
            meta = article.find("front/article-meta")
            if meta is None:
                article.clear()
                continue
            keys = {e.get("pub-id-type"): text(e) for e in meta.findall("article-id")}
            pmcid = keys.get("pmcid")
            if pmcid in ids:
                authors = [author(c) for c in meta.findall("contrib-group/contrib") if c.get("contrib-type") == "author"]
                group_members = [author(c) for g in meta.findall("contrib-group") for parent in g.findall("contrib")
                                 for c in parent.findall(".//contrib-group/contrib") if c.get("contrib-type")=="author"]
                permissions = [ET.tostring(e, encoding="unicode") for e in meta.findall("permissions")]
                row = dict(pmcid=pmcid, version=keys.get("pmcid-ver"), doi=keys.get("doi"),
                           title=text(meta.find("title-group/article-title")), authors=authors, group_members=group_members,
                           permissions=permissions, retrieved=datetime.now(timezone.utc).date().isoformat(),
                           article_type=article.get("article-type"), source=API+"?db=pmc&retmode=xml&id="+pmcid[3:])
                dest = CACHE / (pmcid + ".json")
                dest.with_suffix(".tmp").write_text(json.dumps(row, ensure_ascii=False))
                dest.with_suffix(".tmp").replace(dest)
            article.clear()



def fetch_citations(rows):
    """Retain journal credit lines from the PMC summary, including electronic page numbers."""
    wanted = {r["pmcid"] for r in rows}
    path = CACHE / "journal.json"
    citations = json.loads(path.read_bytes()) if path.is_file() else {}
    missing = sorted(wanted - citations.keys())
    for start in range(0, len(missing), 100):
        batch = missing[start:start + 100]
        url = API.replace("efetch", "esummary") + "?db=pmc&retmode=json&tool=axonai_attribution&id=" + ",".join(n[3:] for n in batch)
        for delay in (0.5, 4, 12):
            time.sleep(delay)
            try:
                with urlopen(url, timeout=30) as response:
                    result = json.load(response)["result"]
                for uid in result["uids"]:
                    r = result[uid]
                    pmcid = next((i["value"] for i in r["articleids"] if i["idtype"]=="pmcid"), None)
                    if pmcid in batch:
                        citations[pmcid] = {k:r.get(k) for k in ("fulljournalname","source","pubdate","volume","issue","pages")}
                path.with_suffix(".tmp").write_text(json.dumps(citations, ensure_ascii=False))
                path.with_suffix(".tmp").replace(path)
                break
            except (OSError, KeyError, ValueError) as exc:
                print("PMC citation batch", start, type(exc).__name__)
                if getattr(exc, "code", None) == 429:
                    time.sleep(30)
    print(f"{len(citations)} journal citations cached")


def main():
    data = INVENTORY.read_bytes()
    assert hashlib.sha256(data).hexdigest() == REGISTER_SHA
    rows = sorted(json.loads(data)["sources"], key=lambda r: (r["license"] != "CC0", r["pmcid"]))
    CACHE.mkdir(parents=True, exist_ok=True)
    missing = []
    for r in rows:
        path = CACHE / (r["pmcid"] + ".json")
        if path.is_file():
            cached = json.loads(path.read_bytes())
            if "group_members" in cached or cached["authors"] == r["authors"]:
                continue
        missing.append(r["pmcid"])
    for offset in range(0, len(missing), 35):
        batch = missing[offset:offset+35]
        for attempt in range(3):
            time.sleep(max(0.5, 3 * attempt))
            try:
                fetch(batch)
                break
            except (OSError, ET.ParseError, EOFError) as exc:
                print("PMC batch", offset, type(exc).__name__, flush=True)
                if getattr(exc, "code", None) == 429:
                    time.sleep(max(20, int(exc.headers.get("Retry-After", 20))))
        if offset % 350 == 0:
            print(f"{len(list(CACHE.glob('PMC*.json')))} metadata records cached", flush=True)
    print(f"{len(list(CACHE.glob('PMC*.json')))} metadata records cached")

    fetch_citations(rows)


if __name__ == "__main__":
    main()
