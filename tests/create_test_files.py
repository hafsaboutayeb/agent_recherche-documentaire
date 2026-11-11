
from pathlib import Path
import csv
from docx import Document

# Dossier de destination
folder = Path("data/raw")
folder.mkdir(parents=True, exist_ok=True)

# 1. Fichier Markdown
(folder / "test.md").write_text(
    "# Intelligence artificielle\n\n"
    "Le RAG permet de rechercher des informations "
    "dans des documents pour générer des réponses.",
    encoding="utf-8"
)

# 2. Fichier HTML
(folder / "test.html").write_text(
    "<html><body>"
    "<h1>Le Machine Learning</h1>"
    "<p>Le Machine Learning permet aux machines "
    "d'apprendre à partir des données.</p>"
    "</body></html>",
    encoding="utf-8"
)

# 3. Fichier CSV
with (folder / "test.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["matiere", "description"]
    )
    writer.writeheader()
    writer.writerow({
        "matiere": "Intelligence artificielle",
        "description": "Etude des systèmes intelligents"
    })
    writer.writerow({
        "matiere": "RAG",
        "description": "Recherche augmentée par génération"
    })

# 4. Fichier Word DOCX
word = Document()
word.add_heading("Introduction au RAG", level=1)
word.add_paragraph(
    "Le RAG combine la recherche documentaire "
    "et la génération de texte."
)
word.save(folder / "test.docx")

print("Les quatre fichiers de test ont été créés !")

# 5. Fichier PDF
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

pdf_path = folder / "test.pdf"

pdf = canvas.Canvas(str(pdf_path), pagesize=A4)

pdf.setFont("Helvetica-Bold", 16)
pdf.drawString(50, 800, "Introduction a l'intelligence artificielle")

pdf.setFont("Helvetica", 12)
pdf.drawString(
    50, 760,
    "L'intelligence artificielle permet aux machines"
)
pdf.drawString(
    50, 740,
    "d'effectuer des taches qui necessitent normalement"
)
pdf.drawString(
    50, 720,
    "l'intelligence humaine."
)

pdf.drawString(
    50, 680,
    "Le RAG permet de rechercher des informations"
)
pdf.drawString(
    50, 660,
    "dans des documents pour generer des reponses."
)

pdf.save()

print("Le fichier PDF a ete cree !")
