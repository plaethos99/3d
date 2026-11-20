"""Index of the installed Daz content: what can be loaded, what it is, what it fits, with thumbnails.

Usage: python pipeline/daz/daz_library.py [--library "C:/Users/Public/Documents/My DAZ 3D Library"]
       [--out tools/daz/library.json] [--find "outfit genesis 9"]
Reads every product's install metadata (Runtime/Support/*.dsx: the product, each loadable asset with its content
type, categories and the figure it fits) and writes one JSON list, one entry per loadable asset:
  {"path": "/People/Genesis 9/...duf" (as Daz's content manager finds it), "product", "artists", "kind",
   "content_type", "categories", "fits" (e.g. "/Genesis 9/Base"), "thumb" (the .png beside it, if any), "words"}
kind: figure, character, wardrobe, hair, material, pose, expression, shape, prop, environment, light, render
setting, script or other. Licensing for games (Daz's Interactive Licence) is per product and not in these files:
the index says "interactive_licence": "check" for every product.
--find: print the entries matching all the words (a quick search). Prints DAZ_LIBRARY_OK.
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def opt(k, default):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else default


library = opt("--library", r"C:\Users\Public\Documents\My DAZ 3D Library")
out = os.path.join(ROOT, opt("--out", os.path.join("tools", "daz", "library.json")))


def kind_of(ctype, cats, path):
    c = (ctype or "").lower()
    cat = " ".join(cats).lower()
    p = path.lower()
    if c.startswith("preset/materials") or c.startswith("preset/shader") or c.startswith("shader"):
        return "material"
    if c.startswith("follower/hair") or ("/hair/" in p and c.startswith(("follower", "preset/wearables"))):
        return "hair"
    if c.startswith("follower/wardrobe") or c.startswith("preset/wearables") or "/wardrobe" in cat:
        return "wardrobe"
    if c.startswith("actor/character") or c.startswith("preset/character"):
        return "character"
    if c.startswith("actor"):
        return "figure"
    if c.startswith("preset/materials") or c.startswith("preset/shader"):
        return "material"
    if c.startswith("preset/pose") and "expression" in cat:
        return "expression"
    if c.startswith("preset/pose") or c.startswith("preset/animation"):
        return "pose"
    if c.startswith("preset/shape") or c.startswith("preset/morph") or c.startswith("modifier"):
        return "shape"
    if c.startswith("prop") or c.startswith("follower/accessory") or c.startswith("follower/attachment"):
        return "prop"
    if c.startswith("set") or "environment" in cat or "/landscape" in cat or "/architecture" in cat:
        return "environment"
    if c.startswith("preset/light") or c.startswith("light"):
        return "light"
    if c.startswith("preset/render") or c.startswith("preset/camera"):
        return "render setting"
    if c.startswith("script"):
        return "script"
    return "other"


entries = []
support = os.path.join(library, "Runtime", "Support")
for f in sorted(os.listdir(support)):
    if not f.lower().endswith(".dsx"):
        continue
    try:
        tree = ET.parse(os.path.join(support, f))
    except ET.ParseError:
        continue
    for prod in tree.getroot().iter("Product"):
        pname = prod.get("VALUE")
        artists = [a.get("VALUE") for a in prod.iter("Artist")]
        for a in prod.iter("Asset"):
            rel = a.get("VALUE", "")
            if not rel.lower().endswith((".duf", ".dsa", ".dse")):
                continue
            ctype = (a.find("ContentType").get("VALUE") if a.find("ContentType") is not None else "")
            cats = [c.get("VALUE") for c in a.iter("Category")]
            fits = [c.get("VALUE") for c in a.iter("Compatibility")]
            thumb = os.path.splitext(os.path.join(library, rel))[0] + ".png"
            words = sorted(set(re.findall(r"[a-z0-9]+", (rel + " " + pname + " " + " ".join(cats)).lower())))
            entries.append({"path": "/" + rel.replace("\\", "/"), "product": pname, "artists": artists,
                            "kind": kind_of(ctype, cats, rel), "content_type": ctype, "categories": cats, "fits": fits,
                            "thumb": thumb if os.path.exists(thumb) else None, "words": words,
                            "interactive_licence": "check"})
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(entries, open(out, "w"), indent=0)
from collections import Counter  # noqa: E402

print("PRODUCTS", len({e["product"] for e in entries}), "| ASSETS", len(entries), "|",
      dict(Counter(e["kind"] for e in entries).most_common()))
if "--find" in sys.argv:
    want = opt("--find", "").lower().split()
    for e in entries:
        if all(w in e["words"] for w in want):
            print(f"  [{e['kind']}] {e['path']}  ({e['product']})")
print("DAZ_LIBRARY_OK", out)
