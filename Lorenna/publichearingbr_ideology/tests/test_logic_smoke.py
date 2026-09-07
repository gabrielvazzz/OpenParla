"""
Teste de fumaça: valida a LÓGICA de decisão (centróides, margem,
threshold de similaridade, pareamento de contradições) usando
embeddings sintéticos/determinísticos, sem baixar os modelos reais do
Hugging Face (este ambiente de execução não tem acesso à internet
para isso). Isto NÃO valida a qualidade semântica do e5-instruct em si
— só que o código ao redor dele está correto.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ideology_classifier as ic
import contradiction_detector as cd
from ideology_classifier import IdeologyClassifier
from contradiction_detector import ContradictionDetector
from party_alignment import find_party_contradictions


# ---------------------------------------------------------------------
# 1) Testa IdeologyClassifier com um encoder falso e determinístico
# ---------------------------------------------------------------------

# Vocabulário mínimo -> vetor one-hot num espaço de 4 dims, pra
# simular "direções semânticas" bem separadas.
FAKE_DIM = 4
WORD_VECS = {
    "esquerda": np.array([1.0, 0.0, 0.0, 0.0]),
    "direita": np.array([0.0, 1.0, 0.0, 0.0]),
    "centro": np.array([0.0, 0.0, 1.0, 0.0]),
    "neutra": np.array([0.0, 0.0, 0.0, 1.0]),
}


def fake_encode(self, texts):
    """Substitui SentenceTransformer.encode: cada texto é mapeado pra
    um vetor a partir das palavras-chave que contém, sem baixar nenhum
    modelo."""
    out = []
    for t in texts:
        low = t.lower()
        vec = np.zeros(FAKE_DIM)
        for kw, v in WORD_VECS.items():
            if kw in low:
                vec = vec + v
        if np.linalg.norm(vec) == 0:
            vec = WORD_VECS["neutra"].copy()
        vec = vec / np.linalg.norm(vec)
        out.append(vec)
    return np.array(out)


class _FakeModel:
    def __init__(self, *a, **kw):
        pass

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        return fake_encode(None, texts)


def test_ideology_classifier():
    ic.SentenceTransformer = _FakeModel  # monkeypatch

    anchors = {
        "neutra": ["texto ancora neutra procedural"],
    }
    policy_anchors = {
        "pauta_teste": {
            "left": ["posição esquerda"],
            "right": ["posição direita"],
        }
    }

    clf = IdeologyClassifier(
        anchors=anchors,
        policy_anchors=policy_anchors,
        policy_evidence_threshold=0.1,
        topic_margin_threshold=0.1,
    )

    # Caso claro de "direita"
    r = clf.classify("Esta fala fala claramente de direita economica")
    assert r.label == "direita", f"esperado direita, veio {r.label}"

    # Caso claro de "esquerda"
    r = clf.classify("posição de esquerda sobre direitos sociais")
    assert r.label == "esquerda", f"esperado esquerda, veio {r.label}"

    # Fala sem nenhuma palavra-chave -> cai em "neutra" por similaridade
    r = clf.classify("Bom dia a todos, agradeco a presenca de todos aqui")
    assert r.label == "neutra", f"esperado neutra, veio {r.label}"

    # Fala vazia -> neutra
    r = clf.classify("   ")
    assert r.label == "neutra"

    # Partido de esquerda + fala inequivocamente à direita.
    opinioes = [{"opiniao": "posição direita", "sessao_id": 7, "assunto": "teste"}]
    conflitos = find_party_contradictions(
        "esquerda", opinioes, ["posição direita"], [clf.classify("posição direita")]
    )
    assert len(conflitos) == 1
    assert conflitos[0].sessao_id == 7

    # Mesmo lado não é contradição fala-partido.
    sem_conflito = find_party_contradictions(
        "direita", opinioes, ["posição direita"], [clf.classify("posição direita")]
    )
    assert sem_conflito == []

    print("[OK] IdeologyClassifier: todos os casos de teste passaram.")


# ---------------------------------------------------------------------
# 2) Testa ContradictionDetector com encoders/NLI falsos
# ---------------------------------------------------------------------

class _FakeTopicModel:
    def __init__(self, *a, **kw):
        pass

    def encode(self, texts, normalize_embeddings=True):
        # Duas "famílias" de tópico: textos que mencionam "aborto" ficam
        # perto um do outro; textos que mencionam "economia" formam
        # outra família; qualquer outra coisa fica isolada.
        out = []
        for t in texts:
            low = t.lower()
            if "aborto" in low:
                v = np.array([1.0, 0.0, 0.0])
            elif "economia" in low:
                v = np.array([0.0, 1.0, 0.0])
            else:
                v = np.array([0.0, 0.0, 1.0])
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def _fake_nli_pipeline(*a, **kw):
    def _one(item):
        if isinstance(item, dict):
            premise, hyp = item["text"], item["text_pair"]
        else:
            premise, hyp = item.split("</s></s>")
        p, h = premise.lower(), hyp.lower()
        # Regra sintética: se uma fala é "a favor do aborto" e a outra
        # é "contra o aborto", é contradição.
        if "aborto" in p and "aborto" in h:
            favor_p = "a favor" in p
            favor_h = "a favor" in h
            if favor_p != favor_h:
                return [{"label": "contradiction", "score": 0.95},
                        {"label": "entailment", "score": 0.02},
                        {"label": "neutral", "score": 0.03}]
            else:
                return [{"label": "entailment", "score": 0.9},
                        {"label": "contradiction", "score": 0.05},
                        {"label": "neutral", "score": 0.05}]
        return [{"label": "neutral", "score": 0.9},
                {"label": "contradiction", "score": 0.05},
                {"label": "entailment", "score": 0.05}]

    def _call(text, truncation=True, batch_size=None):
        if isinstance(text, list):
            return [_one(item) for item in text]
        return [_one(text)]

    return _call


def test_contradiction_detector():
    cd.SentenceTransformer = _FakeTopicModel  # monkeypatch
    cd.pipeline = lambda *a, **kw: _fake_nli_pipeline()  # monkeypatch

    det = ContradictionDetector(topic_sim_threshold=0.9, contradiction_threshold=0.7)

    opinioes = [
        "Sou a favor do aborto em qualquer circunstância.",   # 0
        "Sou totalmente contra o aborto, defendo a vida.",     # 1
        "A economia precisa de mais investimento estatal.",    # 2 (tópico diferente)
    ]

    pares = det.find_contradictions(opinioes)
    assert len(pares) == 1, f"esperado 1 par contraditorio, veio {len(pares)}"
    assert {pares[0].indice_a, pares[0].indice_b} == {0, 1}

    # Duas falas concordantes sobre o mesmo tema -> sem contradição
    opinioes_concordantes = [
        "Sou a favor do aborto em qualquer circunstância.",
        "Reafirmo que sou a favor do aborto como direito da mulher.",
    ]
    pares2 = det.find_contradictions(opinioes_concordantes)
    assert len(pares2) == 0

    # Uma única fala -> nunca há contradição
    assert det.find_contradictions(["única fala"]) == []

    # Score alto em apenas uma direção não deve virar falso positivo.
    def asymmetric_nli(inputs, truncation=True, batch_size=None):
        outputs = []
        for item in inputs:
            high = item["text"].startswith("Primeira")
            outputs.append([
                {"label": "contradiction", "score": 0.90 if high else 0.01},
                {"label": "neutral", "score": 0.09 if high else 0.98},
                {"label": "entailment", "score": 0.01},
            ])
        return outputs

    det.nli = asymmetric_nli
    assert det.find_contradictions(["Primeira fala", "Segunda fala"]) == []

    print("[OK] ContradictionDetector: todos os casos de teste passaram.")


if __name__ == "__main__":
    test_ideology_classifier()
    test_contradiction_detector()
    print("\n[OK] Smoke test completo: lógica de decisão validada "
          "(sem download de modelos reais).")
