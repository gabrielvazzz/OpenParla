"""
anchors.py

Exemplos-âncora usados para construir os centróides de cada classe
ideológica. É um conjunto SEED, pensado para cobrir temas recorrentes
em audiências públicas e discursos na Câmara (papel do Estado na
economia, costumes, segurança pública, meio ambiente, direitos
sociais).

IMPORTANTE: isto é ponto de partida, não um conjunto validado. O ideal
é complementar/substituir estes exemplos com falas reais do próprio
corpus PublicHearingBR, rotuladas manualmente (idealmente por 2+
anotadores, com checagem de concordância), e então recalibrar os
thresholds em ideology_classifier.py em cima desse conjunto rotulado.
O uso de centróide multi-exemplo em vez de um único rótulo-palavra é
uma escolha de engenharia minha, não algo testado no paper do Touché
2025 (Munibuc/NV-Embed-v2), que comparava contra rótulos únicos.
"""

IDEOLOGY_ANCHORS: dict[str, list[str]] = {
    "esquerda": [
        "Defendo a estatização de setores estratégicos como energia e "
        "petróleo, para que o povo brasileiro seja dono dessas riquezas.",
        "É preciso ampliar os direitos trabalhistas e fortalecer os "
        "sindicatos, revertendo os pontos da reforma trabalhista que "
        "precarizaram o emprego.",
        "O Estado deve garantir moradia, saúde e educação públicas e "
        "gratuitas como direitos universais, financiados por tributação "
        "progressiva sobre grandes fortunas.",
        "Sou favorável à legalização do aborto e à autonomia das "
        "mulheres sobre seus próprios corpos.",
        "A reforma agrária é urgente para a redistribuição de terras e "
        "o combate à concentração fundiária no país.",
        "Defendo cotas raciais e políticas afirmativas como reparação "
        "histórica às populações negra e indígena.",
        "O agronegócio precisa ser regulado com rigor para proteger "
        "comunidades indígenas, quilombolas e o meio ambiente.",
        "Somos contrários a qualquer flexibilização das leis "
        "trabalhistas que precarize o emprego e reduza direitos.",
    ],
    "centro-esquerda": [
        "Sou a favor de um Estado regulador forte, mas que também "
        "estimule parcerias com o setor privado quando isso beneficiar "
        "a população.",
        "Defendo o fortalecimento do SUS com mais investimento público, "
        "sem excluir a complementaridade do setor privado na saúde.",
        "Apoio programas de transferência de renda combinados com "
        "qualificação profissional para reduzir a desigualdade.",
        "Sou favorável à descriminalização do porte de drogas para uso "
        "pessoal, mantendo o combate rigoroso ao tráfico.",
        "Defendo uma reforma tributária que taxe mais os mais ricos, "
        "mas sem inviabilizar a atividade produtiva das empresas.",
        "Acredito em metas ambientais rígidas, negociadas com "
        "produtores rurais para viabilizar a transição de forma justa.",
        "Sou a favor da união civil entre pessoas do mesmo sexo e do "
        "combate à discriminação em qualquer esfera.",
    ],
    "centro": [
        "O importante é buscar equilíbrio fiscal sem abrir mão de "
        "investimentos sociais essenciais para a população.",
        "Defendo que cada proposta seja avaliada pelo mérito técnico, "
        "sem amarras ideológicas de um lado ou de outro.",
        "Sou favorável a parcerias público-privadas caso a caso, "
        "conforme o resultado esperado para a população.",
        "Acredito em um Estado eficiente, nem inchado nem mínimo, "
        "focado em entregar serviços públicos de qualidade.",
        "Questões de costumes devem ser decididas prioritariamente pelo "
        "Judiciário e pelo debate social amplo, não por posições "
        "fechadas do Parlamento.",
        "Defendo o diálogo entre governo, empresários e trabalhadores "
        "para construir consensos nas reformas necessárias.",
    ],
    "centro-direita": [
        "Defendo a responsabilidade fiscal e a redução gradual do "
        "tamanho do Estado, mantendo uma rede de proteção social "
        "básica.",
        "Sou favorável à abertura comercial e à atração de investimento "
        "estrangeiro como motor de crescimento econômico.",
        "Apoio a simplificação tributária para estimular o "
        "empreendedorismo, com atenção à progressividade.",
        "Defendo parcerias público-privadas e concessões como forma de "
        "melhorar a infraestrutura do país.",
        "Sou a favor do livre mercado, mas reconheço que o Estado deve "
        "regular setores estratégicos e coibir monopólios.",
        "Apoio o agronegócio como vetor de desenvolvimento, com regras "
        "ambientais claras e previsíveis.",
    ],
    "direita": [
        "Sou a favor da privatização do patrimônio público e do livre "
        "comércio, com o mínimo de interferência do Estado na "
        "economia.",
        "Defendo a redução de impostos e a desburocratização como "
        "caminho para o crescimento econômico do país.",
        "Sou contrário à legalização do aborto e defendo a família "
        "tradicional como base da sociedade.",
        "Defendo o direito ao armamento da população como legítima "
        "defesa pessoal.",
        "Sou a favor do endurecimento das penas e da redução da "
        "maioridade penal para combater a criminalidade.",
        "O Estado gasta demais e precisa ser drasticamente reduzido, "
        "deixando a iniciativa privada resolver mais problemas.",
        "Defendo a flexibilização das leis trabalhistas para gerar mais "
        "empregos e competitividade para as empresas.",
    ],
    "neutra": [
        "Agradeço a presença de todos os convidados nesta audiência "
        "pública e passo a palavra ao próximo orador.",
        "Registro que o relatório foi protocolado na mesa diretora "
        "nesta data, conforme o regimento interno.",
        "Informo que a sessão será suspensa por quinze minutos para "
        "verificação de quórum.",
        "Cumprimento a comissão pelo excelente trabalho de organização "
        "deste evento.",
        "Solicito que a ata da reunião anterior seja anexada aos autos "
        "do processo.",
        "Parabenizo os técnicos que elaboraram o estudo apresentado "
        "nesta audiência pública.",
        "Informo que os documentos citados serão disponibilizados no "
        "portal da Câmara dos Deputados.",
        "Passamos agora à leitura do expediente do dia.",
    ],
}
