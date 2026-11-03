"""Limpieza, división en artículos y verificación de citas."""

from __future__ import annotations

from app.normativa.citations import parse_answer, sources_by_document
from app.normativa.cleaning import clean_markdown
from app.normativa.corpus import load_catalog, load_corpus, render_corpus
from app.normativa.structure import parse_document

UNITS = {
    "INT08-art12": ("INT08", "Un estudiante será declarado egresado cuando cumpla: a) Declaratoria de aptitud con la documentación pertinente."),
    "EXT01-art84": ("EXT01", "La matrícula en una misma asignatura podrá ser hasta por tercera vez."),
}


def test_cleaning_removes_noise_and_keeps_first_letterhead() -> None:
    text = "CÓDIGO: R-01\nTexto\n<!-- image -->\n12\nCÓDIGO: R-01\nMás texto"
    assert clean_markdown(text) == "CÓDIGO: R-01\nTexto\nMás texto"


def test_article_glued_to_previous_paragraph_is_split() -> None:
    text = (
        "Artículo 1.- Ámbito.- Uno.\n\nArtículo 2.- Objeto.- Dos.\n\n"
        "Artículo 3.- Reconocimiento.- Tres. Artículo 4.- Homologación.- Hasta diez (10) años.\n\n"
        "Artículo 5.- Validación.- Según el artículo 4 de este reglamento."
    )
    units = parse_document("EXT02", text).units
    assert [u.unit_id for u in units] == ["EXT02-art1", "EXT02-art2", "EXT02-art3", "EXT02-art4", "EXT02-art5"]
    assert units[3].text.startswith("Artículo 4.- Homologación")


def test_quotes_are_verified_and_invented_ids_detected() -> None:
    text = (
        "Requisito [INT08-art12] y otro [INT08-art99].\n\nFuentes:\nReglamento de Titulación (INT08)\n"
        "- [INT08-art12] «Un estudiante sera declarado egresado ... declaratoria de aptitud»\n- [INT08-art99] «inventado»"
    )
    parsed = parse_answer(text, UNITS)
    assert parsed.unknown_ids == ["INT08-art99"]
    assert parsed.citations[0].quote_verified is True
    assert parsed.body == "Requisito [INT08-art12] y otro [INT08-art99]."


def test_external_sources_are_flagged_and_abstention_detected() -> None:
    parsed = parse_answer("Algo [INT08-art12].\n\nFuentes:\n- https://es.scribd.com/document/1", UNITS)
    assert parsed.external_sources == ["https://es.scribd.com/document/1"]
    assert parse_answer("No encontré en la normativa disponible información sobre becas.", UNITS).abstained


def test_sources_use_official_names_and_skip_unknown_ids() -> None:
    text = "X [EXT01-art84][INT08-art12][INT08-art99].\n\nFuentes:\n- [EXT01-art84] «hasta por tercera vez»"
    names = {"EXT01": "LOES", "INT08": "Reglamento de Titulación"}
    groups = sources_by_document(parse_answer(text, UNITS).citations, names)
    assert [g["document_name"] for g in groups] == ["LOES", "Reglamento de Titulación"]
    assert [u["unit_id"] for g in groups for u in g["units"]] == ["EXT01-art84", "INT08-art12"]


def test_hierarchy_variant_orders_by_rank(data_dir) -> None:
    catalog = load_catalog(data_dir / "catalogo_normativo.json")
    documents = load_corpus(data_dir, catalog)
    plain = render_corpus(documents, "A", catalog)
    ranked = render_corpus(documents, "AJ", catalog)
    assert plain.index("[EXT01]") < plain.index("[INT08]")
    assert "ESCALA DE JERARQUÍA" in ranked
    assert "Jerarquía 2: Ley orgánica" in ranked
    assert "[INT08-art12] **Artículo 12." in plain
