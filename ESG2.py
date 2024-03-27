import pandas as pd
import gspread
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import base64

st.set_page_config(
    page_title="Ideias",
    page_icon="🌱"
)
@st.cache_data()
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("esgfundo.png")
img2 = get_img_as_base64("esgfundo.png")

page_bg_img = f"""
<style>
header, footer {{
    visibility: hidden !important;
}}

#MainMenu {{
    visibility: visible !important;
    color: #F44D00;
}}

[data-testid="stAppViewContainer"] > .main {{
    background-image: url("data:fundoesg4k/png;base64,{img}");
    background-size: cover; 
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:esgfundo1/png;base64,{img2}");
    background-position: center; 
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}

.stTextInput>div>div>input[type="text"] {{
    background-color: #CFE6D2; 
    color: #000; 
    border-radius: 7px; 
    border: 2px solid #000010; 
    padding: 5px; 
    width: 500; 
}}

@media (max-width: 360px) {{
    [data-testid="stAppViewContainer"] > .main, [data-testid="stSidebar"] > div:first-child {{
        background-size: auto;
    }}
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.sidebar.image("esgfundo.png", width=250)


sinonimos = {
    "Pobreza extrema, inclusão social, desigualdade econômica, fome, vulnerabilidade, acesso à educação, emprego digno, sustentabilidade social, igualdade de oportunidades, resiliência financeira, proteção social, microfinanças, redução da pobreza, empoderamento econômico, assistência social, fornecimento de recursos, renda mínima, justiça social, desenvolvimento inclusivo, apoio comunitário.": 'Erradicação da pobreza',
    "Segurança alimentar, agricultura sustentável, desnutrição, agricultura familiar, produção de alimentos, soberania alimentar, desperdício de alimentos, seguro agrícola, acesso à água para irrigação, diversidade alimentar, infraestrutura agrícola, tecnologia agrícola, fome oculta, agricultura de subsistência, comércio justo, políticas alimentares, desenvolvimento rural, agricultura de precisão, sistemas agroflorestais, resiliência agrícola.": "Fome zero e agricultura sustentável",
    "nascimento familia, filhos mães pais famillia saude bem estar, familiar cuidado Saúde pública, prevenção de doenças, acesso à saúde, bem-estar mental, vacinação, saneamento básico, educação em saúde, atenção primária, qualidade de vida, cobertura universal de saúde, doenças transmissíveis, saúde materna, saúde infantil, tratamento médico acessível, pesquisa biomédica, infraestrutura de saúde, promoção da saúde, tecnologia médica, assistência médica preventiva, medicina personalizada.": "Saúde e bem-estar",
    "aprendizes  desenvolvimento webnar webinar aprender aprendizado Inclusão educacional, alfabetização, acesso à educação, ensino de qualidade, equidade educacional, educação inclusiva, desenvolvimento de habilidades, tecnologias educacionais, aprendizado ao longo da vida, parcerias educacionais, infraestrutura escolar, qualificação de professores, educação STEM Ciência, Tecnologia, Engenharia e Matemática, educação para a cidadania, avaliação educacional, inovação pedagógica, educação não formal, educação à distância, educação para a sustentabilidade, educação de base comunitária.": "Educação de qualidade",
    "Empoderamento feminino, igualdade salarial, participação política das mulheres, direitos reprodutivos, violência de gênero, educação de gênero, equidade de oportunidades, liderança feminina, eliminação de estereótipos de gênero, maternidade segura, saúde sexual e reprodutiva, paridade de gênero, discriminação de gênero, igualdade no local de trabalho, acesso a recursos para mulheres, direitos das mulheres, mulheres na ciência e tecnologia, empreendedorismo feminino, emprego digno para mulheres, participação igualitária em esportes.": "Igualdade de gênero",
    "Acesso à água potável, saneamento básico, gestão sustentável da água, higiene adequada, tratamento de água, abastecimento de água seguro, infraestrutura hídrica, qualidade da água, uso eficiente da água, reuso de água, saneamento rural, sistemas de esgoto, água e saneamento em situações de emergência, proteção de fontes de água, tecnologias de saneamento inovadoras, promoção da saúde por meio da água limpa, participação comunitária em projetos hídricos, educação sobre recursos hídricos, água e desenvolvimento sustentável, equidade no acesso à água e saneamento.": "Água potável e saneamento",
    "Energias renováveis, acesso à eletricidade, eficiência energética, energia sustentável, fontes de energia limpa, inovação em energia, eletrificação rural, desenvolvimento de tecnologias verdes, energia solar, energia eólica, biomassa, energia hidrelétrica, hidrogenio verde , infraestrutura energética, acesso a tecnologias de energia, democratização da energia, combate à pobreza energética, educação energética, parcerias para o acesso à energia, investimento em infraestrutura energética, transição para uma matriz energética sustentável.": "Energia limpa e acessível",
    "Emprego digno, crescimento econômico inclusivo, redução do desemprego, igualdade salarial, empreendedorismo, inovação econômica, desenvolvimento de habilidades profissionais, ambientes de trabalho seguros, erradicação do trabalho infantil, trabalho decente para todos, proteção social, emprego sustentável, formalização do trabalho, desenvolvimento de pequenas e médias empresas, bancarização, inclusão financeira, responsabilidade social corporativa, economia verde, comércio justo, redução da desigualdade de renda.": "Trabalho decente e crescimento econômico",
    "Inovação tecnológica, infraestrutura sustentável, desenvolvimento de tecnologias verdes, industrialização, acesso à tecnologia da informação, pesquisa e desenvolvimento, infraestrutura de transporte, inclusão digital, desenvolvimento de habilidades tecnológicas, desenvolvimento de infraestrutura industrial, investimento em infraestrutura, conectividade global, inovação em energias limpas, desenvolvimento de zonas industriais sustentáveis, acesso a serviços de telecomunicação, modernização da infraestrutura, fomento à inovação, desenvolvimento de parcerias público-privadas, desenvolvimento de infraestrutura rural, promoção da pesquisa aplicada.": "Indústria, inovação e infraestrutura",
    "Equidade social, distribuição de renda, inclusão social, oportunidades iguais, participação cidadã, desenvolvimento inclusivo, acesso à educação e saúde, mobilidade social, combate à discriminação, proteção dos direitos humanos, políticas de inclusão, empoderamento de comunidades marginalizadas, igualdade de oportunidades econômicas, participação política equitativa, cooperação internacional para redução de desigualdades, acesso a serviços básicos para todos, promoção da diversidade, medidas contra a discriminação de gênero, raça e orientação sexual, acesso a recursos para grupos vulneráveis, emprego e oportunidades justas para todos.": "Redução das desigualdades",
    "reclica reciclagem reusa reduzir reutilização coleta lixo papel plastico papelão vidro organico Urbanização sustentável, planejamento urbano, habitação acessível, mobilidade sustentável, infraestrutura resiliente, gestão de resíduos sólidos, desenvolvimento de áreas verdes, acesso a serviços básicos nas cidades, participação comunitária, inclusão social urbana, preservação do patrimônio cultural, redução da poluição do ar e sonora, segurança urbana, construção sustentável, eficiência energética em edificações, planejamento de uso do solo, transporte público eficiente, resiliência a desastres naturais, desenvolvimento de tecnologias urbanas sustentáveis, promoção da cultura local nas cidades.": "Cidades e comunidades sustentáveis",
    "Consumo consciente, produção sustentável, desperdício zero, ciclo de vida dos produtos, eficiência no uso de recursos, economia circular, compras éticas, redução da pegada de carbono, consumo de energia responsável, certificações ambientais, ecoeficiência, incentivo ao consumo local, reutilização de produtos, redução do uso de plástico, educação para o consumo sustentável, responsabilidade das empresas na cadeia de produção, inovação em processos sustentáveis, gestão responsável de resíduos, agricultura sustentável, redução da emissão de poluentes.": "Consumo e produção responsáveis",
    "Mitigação das mudanças climáticas, adaptação às mudanças climáticas, energias renováveis, redução das emissões de gases de efeito estufa, transição para uma economia de baixo carbono, preservação de ecossistemas cruciais para o clima, conscientização ambiental, medidas para a eficiência energética, planejamento urbano sustentável, tecnologias limpas, resiliência climática, preservação da biodiversidade, agricultura de baixa emissão de carbono, reflorestamento e conservação florestal, cooperação internacional em mudanças climáticas, regulamentação ambiental eficaz, educação sobre mudanças climáticas, financiamento para ações climáticas, monitoramento e relatórios climáticos, participação cidadã na mitigação climática.": "Ação contra a mudança global do clima",
    "agua praia praiano praias acquatica peixe  aquatica agua Conservação marinha, pesca sustentável, preservação de ecossistemas aquáticos, proteção da biodiversidade marinha, combate à poluição dos oceanos, áreas marinhas protegidas, aquicultura sustentável, monitoramento dos oceanos, redução do descarte de plásticos no mar mares praia praias, recuperação de habitats marinhos, conscientização sobre a vida marinha, combate à pesca ilegal, conservação de espécies ameaçadas aquáticas, oceanografia, ecoturismo marinho responsável, restauração de recifes de coral, gestão sustentável de recursos marinhos, ecologia marinha, inovações para a conservação marinha, cooperação internacional em questões oceânicas.praias praia": "Vida na água",
    "conserva floresta Conservação da biodiversidade, reflorestamento, combate à desertificação, proteção de ecossistemas terrestres, manejo sustentável de florestas, prevenção da extinção de espécies, conservação de habitats naturais, restauração de ecossistemas degradados, biodiversidade em áreas urbanas, proteção de áreas de importância ecológica, redução da perda de solo, combate à caça ilegal, conservação de áreas de importância cultural, monitoramento de espécies ameaçadas, educação ambiental para a vida terrestre, uso sustentável da terra, proteção contra invasões biológicas, incentivo à agricultura sustentável, medidas para a preservação de fauna e flora, desenvolvimento de tecnologias para a conservação.":"Vida terrestre",
    "Estado de direito, justiça social, paz duradoura, direitos humanos, acesso à justiça, combate à corrupção, participação cidadã, igualdade perante a lei, instituições transparentes, redução da violência, proteção de vítimas de crimes, combate ao tráfico humano, promoção da não discriminação, resolução pacífica de conflitos, construção de capacidades institucionais, desenvolvimento de sistemas judiciais eficazes, promoção da verdade e reconciliação, combate à impunidade, cooperação internacional em questões de justiça, educação para a paz.": "Paz, Justiça e instituições eficazes",
    "Cooperação internacional, parcerias público privada, desenvolvimento sustentável, engajamento global, financiamento para o desenvolvimento, compartilhamento de conhecimento, colaboração entre setores, inovação social, tecnologias para o desenvolvimento, capacitação de comunidades locais, transferência de tecnologia, desenvolvimento de capacidades institucionais, mobilização de recursos, desenvolvimento de parcerias multissetoriais, advocacia para os ODS, promoção do voluntariado, cooperação Sul-Sul, monitoramento e avaliação conjunta, participação da sociedade civil, cooperação triangular (países desenvolvidos, em desenvolvimento e instituições internacionais).": "Parcerias e meios de implementação",

}
gc_credentials = {
    "type": st.secrets['google_bigquery']['type'],
    "project_id": st.secrets['google_bigquery']['project_id'],
    "private_key_id": st.secrets['google_bigquery']['private_key_id'],
    "private_key": st.secrets['google_bigquery']['private_key'],
    "client_email": st.secrets['google_bigquery']['client_email'],
    "client_id": st.secrets['google_bigquery']['client_id'],
    "auth_uri": st.secrets['google_bigquery']['auth_uri'],
    "token_uri": st.secrets['google_bigquery']['token_uri'],
    "auth_provider_x509_cert_url": st.secrets['google_bigquery']['auth_provider_x509_cert_url'],
    "client_x509_cert_url": st.secrets['google_bigquery']['client_x509_cert_url']
}

gc = gspread.service_account_from_dict(gc_credentials)
planilha_url = st.secrets['google_sheets']['planilha_url']
dados = gc.open_by_url(planilha_url).worksheet('Respostas ao formulário 1')
colunas = dados.get_all_values()
colunas_selecionadas = ['Carimbo de data/hora', 'Aviso de privacidade de dados - Declaro estar ciente e autorizo a coleta das informações para este formulário.', 'Nome e sobrenome', 'Estado (UF) que originou a ideia:', 'Você poderia compartilhar conosco o embrião da sua ideia, mesmo que ainda não esteja totalmente estruturada?']

df = pd.DataFrame(data=dados.get_all_values(), columns=colunas[0])[colunas_selecionadas]
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df.values.flatten())

def calcular_similaridade(frase_de_entrada, df, ods_selecionadas=None):
    frase_preprocessada = frase_de_entrada.lower()

    for palavra, sinonimo in sinonimos.items():
        frase_preprocessada = frase_preprocessada.replace(palavra, sinonimo)

    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(
            df.fillna('').values.flatten())
        if tfidf_matrix.shape[1] == 0:
            return []
    except ValueError:
        return []

    similaridade_com_frase = cosine_similarity(tfidf_vectorizer.transform([frase_preprocessada]), tfidf_matrix)

    threshold = 0.00
    resultados = []

    for i, row in enumerate(df.index):
        for j, col in enumerate(df.columns):
            celula = str(df.at[row, col])
            similaridade_celula = similaridade_com_frase[0][i * len(df.columns) + j]

            if similaridade_celula > threshold and celula:
                if not ods_selecionadas or any(ods in celula for ods in ods_selecionadas):
                    resultado = {
                        "Linha": row,
                        "Coluna": col,
                        "Data": df.at[row, 'Carimbo de data/hora'],
                        "Nome": df.at[row, 'Nome e sobrenome'],
                        "UF": df.at[row, 'Estado (UF) que originou a ideia:'],
                        "Iniciativa": df.at[row, 'Você poderia compartilhar conosco o embrião da sua ideia, mesmo que ainda não esteja totalmente estruturada?'],
                        "Grau de Similaridade": f"{similaridade_celula * 100:.2f}"
                    }
                    resultados.append(resultado)

            resultados = sorted(resultados, key=lambda x: float(x['Grau de Similaridade']), reverse=True)

    return resultados

st.title("Iniciativas ESG - Similarity 🌱")
frase_de_entrada = st.text_input("Digite um tema para iniciativas (ou 'sair' para encerrar): ")

if frase_de_entrada.lower() == 'sair':
    st.stop()

resultados = calcular_similaridade(frase_de_entrada, df=df)

if frase_de_entrada:
    resultados = calcular_similaridade(frase_de_entrada, df)

    if resultados:
        st.header("Resultado:")
        linhas_filtradas = set()

        for resultado in resultados:
            linha = resultado['Linha']

            if linha not in linhas_filtradas:
                st.subheader(f" {resultado['Coluna']}: {resultado['Você poderia compartilhar conosco o embrião da sua ideia, mesmo que ainda não esteja totalmente estruturada? ']}")
                st.write(f"**Descrição:** {resultado['Carimbo de data/hora']}")
                st.write(f"**Objetivo:** {resultado['Nome e sobrenome']}")
                st.write(f"**ODS's atendidas:** {resultado['Estado (UF) que originou a ideia:']}")
                st.write(f"**Grau de Similaridade:** {resultado['Grau de Similaridade']}%")
                st.markdown("---")

                linhas_filtradas.add(linha)
    else:
        st.warning("Nenhum resultado encontrado.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por [PedroFS](https://linktr.ee/Pedrofsf)")
