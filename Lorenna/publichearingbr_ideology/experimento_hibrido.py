"""Experimento reprodutível para pauta e recuperação de contradições.

O fluxo tem três etapas:
1. ``prepare`` cria amostras para anotação humana de pauta, postura e pares;
2. ``compare`` mede LDA/TF-IDF e embeddings contra essas anotações;
3. ``train-embedding`` ajusta um embedding com pares resumo-transcrição.

Exemplos:
    python experimento_hibrido.py prepare --input deputados.json
    python experimento_hibrido.py compare --input deputados.json \
        --falas-anotadas anotacoes_falas.json \
        --pares-anotados anotacoes_pares.json
    python experimento_hibrido.py train-embedding --input deputados.json \
        --output-model modelo_embedding_publichearingbr
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import joblib
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, normalized_mutual_info_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
PAUTAS = (
    "privatizacao", "papel_do_estado", "tributacao", "direitos_trabalhistas",
    "programas_sociais", "aborto", "armas", "drogas", "direitos_lgbt",
    "cotas", "meio_ambiente", "reforma_agraria", "outra",
)
STOPWORDS_PT = frozenset(
    "a o as os um uma uns umas de da das do dos em no na nos nas por para com "
    "sem sobre entre que e ou mas se ao aos à às como mais menos muito muito "
    "tambem também ja já nao não sim este esta estes estas esse essa esses essas "
    "aquele aquela aqueles aquelas seu sua seus suas meu minha meus minhas nós voces "
    "voces você vocês voce ele ela eles elas foi ser sao são era serão serao porque quando onde "
    "cada todos todas todo toda tem têm ha há".split()
)


@dataclass
class Fala:
    id: str
    deputado: str
    partido: str
    sessao_id: int | None
    assunto: str | None
    texto: str
    contexto: str


def load_falas(path: str) -> list[Fala]:
    with open(path, encoding="utf-8") as file:
        deputados = json.load(file)

    falas = []
    for deputado_index, deputado in enumerate(deputados):
        for fala_index, opiniao in enumerate(deputado.get("opinioes", [])):
            if isinstance(opiniao, dict):
                texto = opiniao.get("opiniao", "")
                trechos = opiniao.get("trechos_transcricao", [])
                contexto = "\n".join(t for t in trechos if isinstance(t, str))
                sessao_id = opiniao.get("sessao_id")
                assunto = opiniao.get("assunto")
            else:
                texto, contexto, sessao_id, assunto = opiniao, "", None, None
            if texto and texto.strip():
                falas.append(Fala(
                    id=f"{deputado_index}:{fala_index}",
                    deputado=deputado.get("nome", ""),
                    partido=deputado.get("partido", ""),
                    sessao_id=sessao_id,
                    assunto=assunto,
                    texto=texto.strip(),
                    contexto=contexto.strip(),
                ))
    return falas


def save_json(data: object, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def fala_para_anotacao(fala: Fala) -> dict:
    return {
        "id": fala.id,
        "deputado": fala.deputado,
        "partido": fala.partido,
        "sessao_id": fala.sessao_id,
        "assunto_origem": fala.assunto,
        "texto": fala.texto,
        "contexto": fala.contexto,
        "pauta": None,
        "postura": None,
        "opcoes_pauta": PAUTAS,
        "opcoes_postura": ("esquerda", "direita", "neutra"),
    }


def lda_topics(textos: list[str], n_topics: int) -> tuple[LatentDirichletAllocation, TfidfVectorizer, np.ndarray, list[dict]]:
    vectorizer = TfidfVectorizer(
        strip_accents="unicode", lowercase=True, min_df=2,
        stop_words=list(STOPWORDS_PT),
    )
    matrix = vectorizer.fit_transform(textos)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    assignments = lda.fit_transform(matrix).argmax(axis=1)
    terms = np.asarray(vectorizer.get_feature_names_out())
    topics = [
        {
            "topico": int(index),
            "termos": terms[component.argsort()[-12:][::-1]].tolist(),
            "falas": int((assignments == index).sum()),
        }
        for index, component in enumerate(lda.components_)
    ]
    return lda, vectorizer, assignments, topics


def candidate_pairs(vectors: np.ndarray, falas: list[Fala], per_deputado: int) -> list[dict]:
    por_deputado: dict[str, list[int]] = defaultdict(list)
    for index, fala in enumerate(falas):
        por_deputado[fala.deputado].append(index)

    pairs = {}
    for indices in por_deputado.values():
        if len(indices) < 2:
            continue
        local = vectors[indices]
        similarities = local @ local.T
        candidates = [
            (float(similarities[i, j]), indices[i], indices[j])
            for i in range(len(indices)) for j in range(i + 1, len(indices))
        ]
        for score, first, second in sorted(candidates, reverse=True)[:per_deputado]:
            key = (falas[first].id, falas[second].id)
            pairs[key] = {
                "fala_a_id": key[0], "fala_b_id": key[1],
                "deputado": falas[first].deputado,
                "similaridade": round(score, 4),
                "fala_a": falas[first].texto, "fala_b": falas[second].texto,
                "mesma_pauta": None, "contradicao": None,
            }
    return sorted(pairs.values(), key=lambda pair: pair["similaridade"], reverse=True)


def prepare(args: argparse.Namespace) -> None:
    falas = load_falas(args.input)
    rng = random.Random(42)
    amostra = rng.sample(falas, min(args.sample_size, len(falas)))
    save_json([fala_para_anotacao(fala) for fala in amostra], args.output_falas)

    _, vectorizer, _, _ = lda_topics([fala.texto for fala in falas], args.n_topics)
    matrix = vectorizer.transform([fala.texto for fala in falas])
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    vectors = matrix.toarray() / np.maximum(norms[:, None], 1e-12)
    save_json(candidate_pairs(vectors, falas, args.pairs_per_deputado), args.output_pares)
    print(f"[OK] {len(amostra)} falas para anotação: {args.output_falas}")
    print(f"[OK] pares candidatos: {args.output_pares}")


def load_annotations(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def supervised_topic_metrics(annotations: list[dict]) -> dict | None:
    labelled = [item for item in annotations if item.get("pauta") in PAUTAS]
    counts = Counter(item["pauta"] for item in labelled)
    if len(labelled) < 20 or min(counts.values()) < 2 or len(counts) < 2:
        return None

    texts = [item["texto"] for item in labelled]
    labels = [item["pauta"] for item in labelled]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels,
    )
    vectorizer = TfidfVectorizer(
        strip_accents="unicode", lowercase=True, ngram_range=(1, 2),
        stop_words=list(STOPWORDS_PT),
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(train_matrix, train_labels)
    predictions = classifier.predict(vectorizer.transform(test_texts))
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro", zero_division=0,
    )
    return {
        "amostras": len(labelled), "classes": dict(counts),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4), "f1_macro": round(float(f1), 4),
    }


def train_topic(args: argparse.Namespace) -> None:
    annotations = load_annotations(args.falas_anotadas)
    labelled = [item for item in annotations if item.get("pauta") in PAUTAS]
    counts = Counter(item["pauta"] for item in labelled)
    if len(labelled) < 20 or len(counts) < 2 or min(counts.values()) < 2:
        raise ValueError(
            "São necessárias pelo menos 20 falas anotadas, duas por pauta."
        )
    vectorizer = TfidfVectorizer(
        strip_accents="unicode", lowercase=True, ngram_range=(1, 2),
        stop_words=list(STOPWORDS_PT),
    )
    matrix = vectorizer.fit_transform([item["texto"] for item in labelled])
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(matrix, [item["pauta"] for item in labelled])
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier}, args.output_model)
    print(f"[OK] Classificador de pauta treinado com {len(labelled)} falas: "
          f"{args.output_model}")


def annotated_pair_metrics(candidates: list[dict], annotations: list[dict], label: str) -> dict | None:
    expected = {
        (item["fala_a_id"], item["fala_b_id"]): item[label]
        for item in annotations if item.get(label) in (True, False)
    }
    available = [pair for pair in candidates if (pair["fala_a_id"], pair["fala_b_id"]) in expected]
    if not available:
        return None
    threshold = np.median([pair["similaridade"] for pair in available])
    truth = [expected[(pair["fala_a_id"], pair["fala_b_id"])] for pair in available]
    predictions = [pair["similaridade"] >= threshold for pair in available]
    return {
        "pares": len(available), "threshold": round(float(threshold), 4),
        "f1": round(float(f1_score(truth, predictions, zero_division=0)), 4),
    }


def compare(args: argparse.Namespace) -> None:
    falas = load_falas(args.input)
    textos = [fala.texto for fala in falas]
    _, vectorizer, assignments, topics = lda_topics(textos, args.n_topics)
    matrix = vectorizer.transform(textos)
    tfidf_vectors = matrix.toarray()
    tfidf_vectors /= np.maximum(np.linalg.norm(tfidf_vectors, axis=1, keepdims=True), 1e-12)

    model = SentenceTransformer(args.embedding_model, device=args.device)
    embedding_vectors = model.encode(textos, normalize_embeddings=True, convert_to_numpy=True)
    tfidf_pairs = candidate_pairs(tfidf_vectors, falas, args.pairs_per_deputado)
    embedding_pairs = candidate_pairs(embedding_vectors, falas, args.pairs_per_deputado)

    report = {
        "dataset": {"falas": len(falas), "deputados": len({fala.deputado for fala in falas})},
        "lda": {"topicos": topics},
        "candidatos": {"tfidf": len(tfidf_pairs), "embedding": len(embedding_pairs)},
    }
    if args.falas_anotadas:
        annotations = load_annotations(args.falas_anotadas)
        labelled = [item for item in annotations if item.get("pauta") in PAUTAS]
        ids_to_index = {fala.id: index for index, fala in enumerate(falas)}
        valid = [item for item in labelled if item["id"] in ids_to_index]
        if valid:
            truth = [item["pauta"] for item in valid]
            predicted = [int(assignments[ids_to_index[item["id"]]]) for item in valid]
            report["lda"]["nmi_pauta_anotada"] = round(
                float(normalized_mutual_info_score(truth, predicted)), 4
            )
        report["classificador_supervisionado_tfidf"] = supervised_topic_metrics(annotations)
    if args.pares_anotados:
        annotations = load_annotations(args.pares_anotados)
        report["avaliacao_pares"] = {
            "tfidf_mesma_pauta": annotated_pair_metrics(tfidf_pairs, annotations, "mesma_pauta"),
            "embedding_mesma_pauta": annotated_pair_metrics(embedding_pairs, annotations, "mesma_pauta"),
            "tfidf_contradicao": annotated_pair_metrics(tfidf_pairs, annotations, "contradicao"),
            "embedding_contradicao": annotated_pair_metrics(embedding_pairs, annotations, "contradicao"),
        }
    save_json(report, args.output)
    save_json(tfidf_pairs, args.output_tfidf_pairs)
    save_json(embedding_pairs, args.output_embedding_pairs)
    print(f"[OK] Relatório comparativo: {args.output}")


def train_embedding(args: argparse.Namespace) -> None:
    falas = [fala for fala in load_falas(args.input) if fala.contexto]
    if not falas:
        raise ValueError("A entrada não possui trechos de transcrição para treino.")
    model = SentenceTransformer(args.embedding_model, device=args.device)
    examples = [InputExample(texts=[fala.texto, fala.contexto]) for fala in falas]
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    model.fit(
        train_objectives=[(loader, losses.MultipleNegativesRankingLoss(model))],
        epochs=args.epochs,
        warmup_steps=min(100, len(loader)),
        output_path=args.output_model,
        show_progress_bar=True,
    )
    print(f"[OK] Embedding ajustado com {len(falas)} pares: {args.output_model}")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser()
    subparsers = command.add_subparsers(dest="command", required=True)
    for name in ("prepare", "compare", "train-topic", "train-embedding"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--input", default="deputados.json")
        subparser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
        subparser.add_argument("--device", default=None)
        subparser.add_argument("--n-topics", type=int, default=13)
        subparser.add_argument("--pairs-per-deputado", type=int, default=3)
    prepare_parser = subparsers.choices["prepare"]
    prepare_parser.add_argument("--sample-size", type=int, default=200)
    prepare_parser.add_argument("--output-falas", default="anotacoes_falas.json")
    prepare_parser.add_argument("--output-pares", default="anotacoes_pares.json")
    compare_parser = subparsers.choices["compare"]
    compare_parser.add_argument("--falas-anotadas")
    compare_parser.add_argument("--pares-anotados")
    compare_parser.add_argument("--output", default="relatorio_experimento.json")
    compare_parser.add_argument("--output-tfidf-pairs", default="pares_tfidf.json")
    compare_parser.add_argument("--output-embedding-pairs", default="pares_embedding.json")
    train_parser = subparsers.choices["train-embedding"]
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--output-model", default="modelo_embedding_publichearingbr")
    topic_parser = subparsers.choices["train-topic"]
    topic_parser.add_argument("--falas-anotadas", required=True)
    topic_parser.add_argument("--output-model", default="modelo_pauta_tfidf.joblib")
    return command


if __name__ == "__main__":
    args = parser().parse_args()
    {
        "prepare": prepare,
        "compare": compare,
        "train-topic": train_topic,
        "train-embedding": train_embedding,
    }[args.command](args)
