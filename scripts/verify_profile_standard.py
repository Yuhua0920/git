"""Download exact ETSI Release17 PDF and verify A/B/C table rows against code."""
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import pymupdf
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src import tdl_profiles as profiles
from src.lmmse_train import sha256_file

PDF=ROOT/"outputs/t14a_standard_tr38901_v170000.pdf"
AUDIT=ROOT/"outputs/t14a_standard_profile_audit.json"


def canonical_rows(rows):
    return "\n".join(f"{i},{d:.4f},{p:.1f},{f}" for i,d,p,f in rows)+"\n"


def audit_standard():
    if not PDF.exists():
        request=urllib.request.Request(profiles.STANDARD_URL,headers={"User-Agent":"Research-profile-audit/1.0"})
        with urllib.request.urlopen(request,timeout=60) as response:
            data=response.read()
        if not data.startswith(b"%PDF"):
            raise ValueError("official source did not return PDF")
        PDF.write_bytes(data)
    with pymupdf.open(PDF) as document:
        if document.page_count!=100 or "V17.0.0" not in document[0].get_text():
            raise ValueError("wrong 3GPP release")
        blocks={"A":document[74].get_text().split("Table 7.7.2-1.")[1].split("Table 7.7.2-2.")[0],
            "B":document[74].get_text().split("Table 7.7.2-2.")[1],
            "C":document[75].get_text().split("Table 7.7.2-3.")[1].split("Table 7.7.2-4.")[0]}
        records={}
        for profile,text in blocks.items():
            found=re.findall(r"(?m)^\s*(\d+)\s+([\d.]+)\s+(-?[\d.]+)\s+(Rayleigh)\s*$",text)
            rows=[(int(i),float(d),float(p),f) for i,d,p,f in found]
            expected=[(i+1,d,p,"Rayleigh") for i,(d,p) in enumerate(zip(profiles.NORMALIZED_DELAYS[profile],profiles.POWERS_DB[profile]))]
            if rows!=expected:
                raise ValueError(f"TDL-{profile} table differs from code: extracted {len(rows)} rows; expected {len(expected)}")
            digest=hashlib.sha256(canonical_rows(rows).encode("utf8")).hexdigest()
            records[profile]={"table":profiles.TABLES[profile],"printed_page":74 if profile in ("A","B") else 75,
                "tap_count":len(rows),"rows":[{"tap":i,"normalized_delay":d,"power_db":p,"fading":f} for i,d,p,f in rows],
                "canonical_format":"tap,delay four decimals,power one decimal,fading; UTF8 LF","canonical_table_sha256":digest,"all_checks_passed":True,
                **profiles.profile_diagnostics(profile)}
    report={"source":profiles.STANDARD_URL,"standard":profiles.STANDARD_REFERENCE,"release":17,
        "PDF":str(PDF),"PDF_sha256":sha256_file(PDF),"extraction":"PyMuPDF table text + numeric row regex; exact per-row comparison with module",
        "profiles":records,"all_checks_passed":True}
    AUDIT.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
    return report


if __name__=="__main__":
    r=audit_standard()
    print("Official PDF SHA256:",r["PDF_sha256"])
    for p,v in r["profiles"].items():
        print(p,v["tap_count"],v["canonical_table_sha256"],"RMS ns",v["actual_RMS_delay_ns"])
    print("standard profile audit PASS")
