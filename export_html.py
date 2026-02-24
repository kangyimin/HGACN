import json


def export_html_report(explain_json_path, out_html, atom_png=None, pdb_path=None):
    with open(explain_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pair = data.get("pair", {})
    gate = data.get("experts", {})
    evidence = data.get("evidence", {})
    unc = data.get("uncertainty", {})

    html = []
    html.append("<html><head><meta charset='utf-8'><title>HGACN Explain</title></head><body>")
    html.append(f"<h2>DTI Explain: drug={pair.get('drug_id')} prot={pair.get('prot_id')}</h2>")
    html.append(f"<p>logit={data.get('logit'):.4f} prob={data.get('prob'):.4f}</p>")
    if atom_png:
        html.append(f"<h3>Drug atoms</h3><img src='{atom_png}' style='max-width:480px;'/>")
    if pdb_path:
        html.append(f"<h3>Protein residues</h3><p>PDB with B-factor: {pdb_path}</p>")
    html.append("<h3>Experts & Gate</h3>")
    html.append(f"<pre>{json.dumps(gate, indent=2)}</pre>")
    html.append("<h3>Evidence</h3>")
    html.append(f"<pre>{json.dumps(evidence, indent=2)}</pre>")
    html.append("<h3>Uncertainty</h3>")
    html.append(f"<pre>{json.dumps(unc, indent=2)}</pre>")
    html.append("</body></html>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
