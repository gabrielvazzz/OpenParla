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


# Pares de posições opostas sobre a MESMA pauta. O classificador usa a
# diferença entre os dois polos, e não a similaridade absoluta com rótulos
# genéricos. Isso reduz o efeito observado no E5 em que todos os centróides
# políticos ficam com cosseno muito alto e quase idêntico.
#
# Em cada pauta, "left" deve conter a posição associada ao polo esquerdo e
# "right" sua oposição associada ao polo direito. Os exemplos precisam ser
# semanticamente paralelos para que o eixo represente postura, não tópico.
POLICY_STANCE_ANCHORS: dict[str, dict[str, list[str]]] = {
    "privatizacao": {
        "left": [
            "Sou contra privatizações e defendo empresas estatais fortes.",
            "Setores estratégicos devem permanecer sob controle público.",
            "O patrimônio público não deve ser transferido à iniciativa privada.",
        ],
        "right": [
            "Sou favorável à privatização das empresas públicas.",
            "A iniciativa privada deve administrar as empresas estatais.",
            "O patrimônio estatal deve ser vendido para reduzir o tamanho do Estado.",
        ],
    },
    "papel_do_estado": {
        "left": [
            "O Estado deve ampliar sua atuação na economia e nos serviços públicos.",
            "Defendo investimento público e planejamento estatal para desenvolver o país.",
            "O governo deve intervir na economia para reduzir desigualdades.",
        ],
        "right": [
            "O Estado deve reduzir sua atuação e deixar mais espaço ao mercado.",
            "Defendo menos intervenção estatal e mais liberdade econômica.",
            "O governo deve ser menor e a iniciativa privada deve liderar a economia.",
        ],
    },
    "tributacao": {
        "left": [
            "Defendo impostos maiores sobre ricos e grandes fortunas.",
            "A tributação deve ser progressiva para redistribuir renda.",
            "Grandes patrimônios e lucros devem pagar proporcionalmente mais impostos.",
        ],
        "right": [
            "Defendo a redução de impostos e uma carga tributária menor.",
            "É preciso reduzir tributos sobre empresas e patrimônio.",
            "Menos impostos estimulam investimento, emprego e crescimento.",
        ],
    },
    "direitos_trabalhistas": {
        "left": [
            "Os direitos trabalhistas devem ser ampliados e protegidos.",
            "Sou contra flexibilizar leis trabalhistas e reduzir direitos dos empregados.",
            "Sindicatos fortes e proteção legal são essenciais para os trabalhadores.",
        ],
        "right": [
            "As leis trabalhistas devem ser flexibilizadas para gerar empregos.",
            "Defendo menos encargos e liberdade de negociação entre empresa e empregado.",
            "A legislação trabalhista precisa ser reduzida para aumentar a competitividade.",
        ],
    },
    "programas_sociais": {
        "left": [
            "O governo deve ampliar programas de transferência de renda.",
            "Políticas sociais universais são essenciais para combater a desigualdade.",
            "O Estado deve aumentar os benefícios destinados à população vulnerável.",
        ],
        "right": [
            "Programas de transferência de renda devem ser reduzidos e temporários.",
            "Benefícios sociais excessivos geram dependência e devem ser limitados.",
            "A assistência estatal deve ser mínima e focada apenas em casos extremos.",
        ],
    },
    "aborto": {
        "left": [
            "Defendo a legalização do aborto e a autonomia das mulheres.",
            "O aborto deve ser descriminalizado e tratado como questão de saúde pública.",
            "A mulher deve ter o direito de decidir pela interrupção da gravidez.",
        ],
        "right": [
            "Sou contra a legalização do aborto e defendo a vida desde a concepção.",
            "O aborto deve continuar proibido pela legislação.",
            "A interrupção voluntária da gravidez não deve ser permitida.",
        ],
    },
    "armas": {
        "left": [
            "Sou contra ampliar o acesso da população a armas de fogo.",
            "O porte de armas deve ser mais restrito.",
            "É preciso controlar e reduzir a circulação de armas entre civis.",
        ],
        "right": [
            "Defendo o direito da população de possuir armas para legítima defesa.",
            "O cidadão deve ter acesso facilitado a armas de fogo.",
            "Sou favorável à ampliação do porte e da posse de armas.",
        ],
    },
    "drogas": {
        "left": [
            "Defendo a descriminalização do porte de drogas para uso pessoal.",
            "A política de drogas deve priorizar saúde pública e redução de danos.",
            "O usuário de drogas não deve ser tratado como criminoso.",
        ],
        "right": [
            "Sou contra descriminalizar o porte de drogas para uso pessoal.",
            "A política de drogas deve endurecer a repressão e as penas.",
            "Usuários e traficantes devem enfrentar leis mais rigorosas.",
        ],
    },
    "direitos_lgbt": {
        "left": [
            "Defendo igualdade de direitos e casamento para pessoas LGBT.",
            "O Estado deve combater a discriminação por orientação sexual e identidade de gênero.",
            "Famílias formadas por pessoas do mesmo sexo devem ter os mesmos direitos.",
        ],
        "right": [
            "Sou contra ampliar o casamento e a adoção para casais do mesmo sexo.",
            "O Estado deve preservar o conceito tradicional de família.",
            "Políticas de identidade de gênero não devem ser promovidas pelo governo.",
        ],
    },
    "cotas": {
        "left": [
            "Defendo cotas raciais e políticas afirmativas nas universidades.",
            "Ações afirmativas são necessárias para reparar desigualdades históricas.",
            "O Estado deve ampliar políticas específicas para grupos racialmente discriminados.",
        ],
        "right": [
            "Sou contra cotas raciais e defendo critérios sem distinção de raça.",
            "O acesso à universidade não deve usar critérios raciais.",
            "Políticas públicas não devem conceder benefícios com base em raça.",
        ],
    },
    "meio_ambiente": {
        "left": [
            "A proteção ambiental deve prevalecer mesmo com custos econômicos.",
            "O agronegócio precisa de fiscalização ambiental mais rigorosa.",
            "É necessário restringir atividades econômicas para proteger florestas e povos tradicionais.",
        ],
        "right": [
            "Regras ambientais devem ser flexibilizadas para permitir o desenvolvimento econômico.",
            "O agronegócio deve ter menos restrições e burocracia ambiental.",
            "A exploração econômica deve avançar sem excesso de fiscalização ambiental.",
        ],
    },
    "reforma_agraria": {
        "left": [
            "Defendo a reforma agrária e a redistribuição de terras improdutivas.",
            "Grandes propriedades improdutivas devem ser desapropriadas para assentamentos.",
            "É necessário reduzir a concentração fundiária no país.",
        ],
        "right": [
            "Sou contra desapropriações para reforma agrária e defendo a propriedade privada.",
            "A propriedade rural deve ser protegida contra ocupações e redistribuição estatal.",
            "O governo não deve tomar terras privadas para criar assentamentos.",
        ],
    },
}
