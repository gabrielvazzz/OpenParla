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
from ideology_classifier import IdeologyClassifier, ClassificationResult
from contradiction_detector import ContradictionDetector
from party_alignment import (
    find_party_contradictions,
    find_party_divergences,
    party_position_for_issue,
)
from padroes_politicos import find_cross_issue_tensions, detect_defensive_speech
from indicios import (
    indicio_da_fala,
    party_divergences_from_indicios,
    cross_issue_tensions_from_indicios,
)


def fake_result(score, issue, strength, label=None):
    """Monta um ClassificationResult sintético sem passar pelo encoder."""
    if label is None:
        label = IdeologyClassifier._label_from_score(score)
    return ClassificationResult(
        label=label,
        scores={},
        margin=0.05,
        top1=0.05,
        ideology_score=score,
        issue=issue,
        issue_similarity=0.9,
        policy_evidence=0.05,
        topic_margin=0.05,
        stance_strength=strength,
    )


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


# ---------------------------------------------------------------------
# 3) Testa extensões: divergência partido-fala por pauta + padrões de
#    posicionamento (tensão transversal e fala defensiva)
# ---------------------------------------------------------------------

def test_party_divergences():
    opinioes = [{"opiniao": "x", "sessao_id": 1, "assunto": "a"}]
    textos = ["x"]

    # PL + fala forte à esquerda em armas -> "contradicao".
    r = fake_result(-0.8, "armas", strength=0.4, label="esquerda")
    d = find_party_divergences("PL", opinioes, textos, [r])
    assert len(d) == 1 and d[0].severidade == "contradicao"

    # UNIÃO (centro-direita) em direitos_lgbt + fala à esquerda, porém fraca:
    # não é contradição, é "tensao".
    r = fake_result(-0.6, "direitos_lgbt", strength=0.2, label="esquerda")
    d = find_party_divergences("UNIÃO", opinioes, textos, [r])
    assert len(d) == 1 and d[0].severidade == "tensao"

    # Fala fraca demais até para tensão -> nada.
    r = fake_result(-0.6, "direitos_lgbt", strength=0.05, label="esquerda")
    assert find_party_divergences("UNIÃO", opinioes, textos, [r]) == []

    # Mesmo lado do partido -> nada.
    r = fake_result(0.6, "armas", strength=0.4, label="direita")
    assert find_party_divergences("PL", opinioes, textos, [r]) == []

    # Partido de centro não tem polo definido -> nada.
    assert find_party_divergences("MDB", opinioes, textos, [r]) == []

    # Partido desconhecido -> nada.
    assert find_party_divergences("PARTIDO_X", opinioes, textos, [r]) == []

    # Posição por pauta com fallback global.
    assert party_position_for_issue("REPUBLICANOS", "armas") == 1.0
    assert party_position_for_issue("REPUBLICANOS", "tributacao") == 0.75
    assert party_position_for_issue("PT", "meio_ambiente") == -0.5
    assert party_position_for_issue("MDB", "aborto") == 0.0
    assert party_position_for_issue(None, "aborto") is None
    assert party_position_for_issue("INEXISTENTE", "aborto") is None

    print("[OK] Divergências partido-fala por pauta: casos passaram.")


def test_cross_issue_tensions():
    opinioes = [
        {"opiniao": "a", "sessao_id": 1, "assunto": "a"},
        {"opiniao": "b", "sessao_id": 2, "assunto": "b"},
    ]
    textos = ["a", "b"]

    # Esquerda em programas_sociais + direita em tributacao -> tensão.
    r1 = fake_result(-0.7, "programas_sociais", strength=0.3)
    r2 = fake_result(0.7, "tributacao", strength=0.3)
    t = find_cross_issue_tensions(opinioes, textos, [r1, r2])
    assert len(t) == 1
    assert t[0].pauta_esquerda == "programas_sociais"
    assert t[0].pauta_direita == "tributacao"

    # Apenas um lado posicionado -> nada.
    r2_neutro = fake_result(0.1, "tributacao", strength=0.3)
    assert find_cross_issue_tensions(opinioes, textos, [r1, r2_neutro]) == []

    # Posições fortes mas sem evidência (força) -> nada.
    r1_fraca = fake_result(-0.7, "programas_sociais", strength=0.05)
    assert find_cross_issue_tensions(opinioes, textos, [r1_fraca, r2]) == []

    print("[OK] Tensões transversais entre pautas: casos passaram.")


def test_defensive_speech():
    opinioes = [
        {"opiniao": "Os parlamentares não são 'antivacinas'.", "sessao_id": 15},
        {"opiniao": "Precisamos votar o relatório agora.", "sessao_id": 6},
        {"opiniao": "Não sou contra a vacinação, mas tenho ressalvas.", "sessao_id": 7},
    ]
    textos = [o["opiniao"] for o in opinioes]
    resultados = [
        fake_result(0.5, "saude_campanha", strength=0.2),
        fake_result(0.0, "procedimento", strength=0.0),
        fake_result(0.6, "saude_campanha", strength=0.3),
    ]

    d = detect_defensive_speech(opinioes, textos, resultados)
    assert len(d) == 2
    assert [x.indice for x in d] == [0, 2]
    assert d[0].termo_negado in ("antivacinas", "antivacina")
    assert d[0].sessao_id == 15
    assert d[1].termo_negado == "contra"
    assert d[1].pauta == "saude_campanha"

    print("[OK] Falas defensivas (negação de rótulo): casos passaram.")


def test_indicios_com_textos_reais():
    """Casos borderline do corpus real, detectados sem o classificador."""
    # Márcio Marinho (REPUBLICANOS): crítica a gastos com armas.
    m_marcio = "Defendeu apoio público para as artes marciais nas escolas. "\
        "'Se gasta tanto com armas, armando a população, armando o cidadão'."
    ind = indicio_da_fala(m_marcio)
    assert ind is not None and ind.issue == "armas" and ind.lado == -1

    # Mauricio do Vôlei (PL): defendeu a taxação das apostas.
    ind = indicio_da_fala("Defendeu a taxação das casas de apostas esportivas.")
    assert ind is not None and ind.issue == "tributacao" and ind.lado == -1

    # Danilo Forte (UNIÃO): subsídio social (esquerda) + resistência a
    # aumento de tributos (direita) em falas diferentes.
    f_social = "Há consenso sobre a migração da Tarifa Social de Energia "\
        "Elétrica para o orçamento da União. Hoje esse subsídio é custeado "\
        "pela conta de energia dos consumidores."
    ind = indicio_da_fala(f_social)
    assert ind is not None and ind.issue == "programas_sociais" and ind.lado == -1
    f_trib = "Matérias impopulares e que não têm a simpatia do Parlamento com "\
        "relação a aumento de tributos."
    ind = indicio_da_fala(f_trib)
    assert ind is not None and ind.issue == "tributacao" and ind.lado == +1

    # Ivan Valente (PSOL): moderação ambiental.
    ind = indicio_da_fala(
        "A discussão é complexa e é preciso equilibrar a necessidade de "
        "produção de petróleo com a sustentabilidade ambiental."
    )
    assert ind is not None and ind.issue == "meio_ambiente" and ind.lado == -1
    # E sua fala pró-Petrobras aponta para o polo oposto (direita ambiental).
    ind = indicio_da_fala("Defende a capacidade estratégica da Petrobras.")
    assert ind is not None and ind.issue == "meio_ambiente" and ind.lado == +1

    # Bia Kicis e Capitão Alberto Neto NÃO geram indício léxico de polo.
    assert indicio_da_fala("Os parlamentares não são 'antivacinas'.") is None
    assert indicio_da_fala(
        "Criticou a possibilidade de extinção do saque-aniversário."
    ) is None

    print("[OK] Indícios léxicos (casos reais): passaram.")


def test_indicios_partido_e_transversal():
    # Márcio Marinho: REPUBLICANOS (armas=+1) x crítica a armas -> tensao.
    o = [{"opiniao": t, "sessao_id": s, "assunto": "a"} for s, t in
         [(40, "Se gasta tanto com armas, armando a população.")]]
    txt = [x["opiniao"] for x in o]
    res = [fake_result(0.0, None, strength=0.0)]
    inc = [indicio_da_fala(t) for t in txt]
    d = party_divergences_from_indicios("REPUBLICANOS", o, txt, res, inc)
    assert len(d) == 1 and d[0].severidade == "tensao"
    assert d[0].origem == "indicio_lexical"
    assert d[0].pauta == "armas"

    # Danilo Forte: UNIÃO. Indício de subsídio (esq) x resistência a tributos
    # (dir) em falas diferentes -> tensão transversal.
    textos = [
        "Há consenso sobre a migração da Tarifa Social de Energia Elétrica "
        "para o orçamento da União.",
        "Matérias impopulares sem simpatia do Parlamento sobre aumento de "
        "tributos.",
        "Vamos analisar a legislação de outros países.",
    ]
    opinioes = [{"opiniao": t, "sessao_id": i + 1, "assunto": "a"}
                for i, t in enumerate(textos)]
    resultados = [fake_result(0.0, None, strength=0.0) for _ in textos]
    indicios = [indicio_da_fala(t) for t in textos]
    assert indicios[0].issue == "programas_sociais" and indicios[0].lado == -1
    assert indicios[1].issue == "tributacao" and indicios[1].lado == +1
    t = cross_issue_tensions_from_indicios(opinioes, textos, resultados, indicios)
    assert len(t) == 1
    assert t[0].pauta_esquerda == "programas_sociais"
    assert t[0].pauta_direita == "tributacao"

    # Capitão Alberto Neto (PL): sem indícios -> nada dispara.
    ca = [{"opiniao": "Criticou a possibilidade de extinção do "
                      "saque-aniversário.", "sessao_id": 3, "assunto": "a"}]
    cat = [x["opiniao"] for x in ca]
    cr = [fake_result(0.0, None, strength=0.0)]
    cin = [indicio_da_fala(x) for x in cat]
    assert party_divergences_from_indicios("PL", ca, cat, cr, cin) == []
    assert cross_issue_tensions_from_indicios(ca, cat, cr, cin) == []

    # Ivan Valente (PSOL=-1.0 em meio_ambiente) x fala pró-Petrobras -> tensao.
    iv = [{"opiniao": "Defende a capacidade estratégica da Petrobras.",
           "sessao_id": 44, "assunto": "a"}]
    ivt = [x["opiniao"] for x in iv]
    ivr = [fake_result(0.0, None, strength=0.0)]
    ivin = [indicio_da_fala(x) for x in ivt]
    d = party_divergences_from_indicios("PSOL", iv, ivt, ivr, ivin)
    assert len(d) == 1 and d[0].pauta == "meio_ambiente"
    assert d[0].score_fala > 0 and d[0].severidade == "tensao"

    print("[OK] Indícios: divergência partido e tensão transversal passaram.")


if __name__ == "__main__":
    test_ideology_classifier()
    test_contradiction_detector()
    test_party_divergences()
    test_cross_issue_tensions()
    test_defensive_speech()
    test_indicios_com_textos_reais()
    test_indicios_partido_e_transversal()
    print("\n[OK] Smoke test completo: lógica de decisão validada "
          "(sem download de modelos reais).")
