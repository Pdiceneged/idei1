import pandas as pd
import gspread
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import base64

st.set_page_config(
    page_title="Pré Ideiando",
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

st.title("Pré-Ideiando 🌱")
frase_de_entrada = st.text_input("Digite um Ideia (ou 'sair' para encerrar): ")

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
                st.subheader(f" {resultado['Coluna']}: {resultado['Iniciativa']}")
                st.write(f"**Data:** {resultado['Carimbo de data/hora']}")
                st.write(f"**Nome:** {resultado['Nome e sobrenome']}")
                st.write(f"**UF:** {resultado['Estado (UF) que originou a ideia:']}")
                st.write(f"**Grau de Similaridade:** {resultado['Grau de Similaridade']}%")
                st.markdown("---")

                linhas_filtradas.add(linha)
    else:
        st.warning("Nenhum resultado encontrado.")

st.sidebar.markdown("---")
st.sidebar.markdown("Desenvolvido por [PedroFS](https://linktr.ee/Pedrofsf)")
