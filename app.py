import base64
import json
import os
import re
import unicodedata

import numpy as np
import ollama
import streamlit as st

AQUI = os.path.dirname(os.path.abspath(__file__))
MATERIAIS = os.path.join(AQUI, "materiais")
INDICE = os.path.join(AQUI, "dados", "indice.json")
ANTIGO = os.path.join(AQUI, "dados", "conhecimento.json")
IMAGENS = ["acaua.gif", "acaua.jpg", "acaua.jpeg", "acaua.png"]
MODELO = "llama3.2"
MODELO_EMB = "nomic-embed-text"
MANTER = "30m"
OPCOES = {"num_ctx": 3072, "num_predict": 400, "temperature": 0.3}
MIN_SIMILARIDADE = 0.45
VERSAO = "v4"
CHAVE_MENSAGENS = "mensagens_acaua_v4"  # chave nova para não reaproveitar mensagens antigas com fontes

st.set_page_config(page_title="Acauã", page_icon=None, layout="centered")
os.makedirs(MATERIAIS, exist_ok=True)
os.makedirs(os.path.dirname(INDICE), exist_ok=True)


def normalizar(texto):
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", texto).split())


def pede_material(pergunta):
    """Busca apenas quando a mensagem aponta para arquivos/materiais específicos."""
    p = normalizar(pergunta)
    gatilhos = (
        "material", "materiais", "apostila", "pdf", "arquivo", "documento",
        "livro digital", "segundo o texto", "segundo o material", "com base no texto",
        "com base no material", "no documento", "na apostila", "no pdf",
        "nos arquivos", "na unidade", "aula da faculdade", "uniasselvi",
        "desafio profissional", "resuma o arquivo", "resuma o pdf", "resuma a apostila",
        "o que diz o arquivo", "o que diz o material", "procure nos materiais",
        "pesquise nos arquivos", "consulte a apostila"
    )
    return any(g in p for g in gatilhos)


def parece_nao_saber(resposta):
    r = normalizar(resposta)
    sinais = (
        "nao sei", "nao tenho certeza", "nao consigo confirmar", "nao tenho informacao",
        "nao encontrei informacao", "nao posso afirmar", "preciso consultar"
    )
    return any(s in r for s in sinais)


def limpar_fontes(texto):
    """Remove linhas de fontes/citações automáticas, se o modelo tentar incluí-las."""
    linhas = str(texto or "").splitlines()
    limpas = []
    for linha in linhas:
        n = normalizar(linha.strip())
        if re.match(r"^(fontes?|sources?)\s*[:：-]", n):
            continue
        if n in {"fontes", "sources", "referencias", "referencias utilizadas"}:
            continue
        limpas.append(linha)
    return "\n".join(limpas).strip()


def ler_arquivo(caminho):
    if caminho.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(caminho).pages)
    with open(caminho, encoding="utf-8", errors="ignore") as f:
        return f.read()


def dividir(texto, tamanho=700, sobreposicao=100):
    texto = " ".join(texto.split())
    pedacos, i = [], 0
    while i < len(texto):
        pedacos.append(texto[i:i + tamanho])
        i += tamanho - sobreposicao
    return pedacos


def embutir(textos, tipo):
    entrada = [f"{tipo}: {t}" for t in textos]
    return ollama.embed(model=MODELO_EMB, input=entrada, keep_alive=MANTER)["embeddings"]


def listar_arquivos():
    achados = {}
    for raiz, _, nomes in os.walk(MATERIAIS):
        for nome in nomes:
            if nome.lower().endswith((".pdf", ".txt", ".md")):
                caminho = os.path.join(raiz, nome)
                achados[os.path.relpath(caminho, MATERIAIS)] = caminho
    return achados


def assinatura():
    return [(nome, os.path.getmtime(c), os.path.getsize(c))
            for nome, c in sorted(listar_arquivos().items())]


def carregar_antigo():
    if not os.path.exists(ANTIGO):
        return []
    try:
        with open(ANTIGO, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


@st.cache_resource(show_spinner=False)
def carregar_base(assin):
    antigo = {}
    if os.path.exists(INDICE):
        try:
            with open(INDICE, encoding="utf-8") as f:
                antigo = json.load(f)
        except Exception:
            antigo = {}

    novo, arquivos = {}, listar_arquivos()
    for nome, mtime, tamanho in assin:
        marca = f"{VERSAO}-{mtime}-{tamanho}"
        if nome in antigo and antigo[nome].get("marca") == marca:
            novo[nome] = antigo[nome]
            continue
        try:
            pedacos = dividir(ler_arquivo(arquivos[nome]))
            if not pedacos:
                continue
            embs = []
            for i in range(0, len(pedacos), 16):
                embs += embutir(pedacos[i:i + 16], "search_document")
            novo[nome] = {"marca": marca, "pedacos": pedacos, "embs": embs}
        except Exception:
            continue

    try:
        with open(INDICE, "w", encoding="utf-8") as f:
            json.dump(novo, f, ensure_ascii=False)
    except Exception:
        pass

    fontes, textos, vetores = [], [], []
    for nome, dados in novo.items():
        for trecho, vetor in zip(dados["pedacos"], dados["embs"]):
            fontes.append(nome)
            textos.append(trecho)
            vetores.append(vetor)
    for item in carregar_antigo():
        try:
            fontes.append(item.get("fonte", "antigo"))
            textos.append(item["texto"])
            vetores.append(item["emb"])
        except Exception:
            continue

    if not vetores:
        return [], [], None
    dimensao = len(vetores[0])
    pares = [(f, t, v) for f, t, v in zip(fontes, textos, vetores) if len(v) == dimensao]
    fontes = [p[0] for p in pares]
    textos = [p[1] for p in pares]
    matriz = np.array([p[2] for p in pares], dtype=np.float32)
    matriz /= np.linalg.norm(matriz, axis=1, keepdims=True) + 1e-9
    return fontes, textos, matriz


def buscar(pergunta, fontes, textos, matriz, k=4):
    if matriz is None:
        return []
    vetor = np.array(embutir([pergunta], "search_query")[0], dtype=np.float32)
    if matriz.shape[1] != vetor.shape[0]:
        return []
    vetor /= np.linalg.norm(vetor) + 1e-9
    similaridades = matriz @ vetor
    ordem = np.argsort(similaridades)[::-1][:k]
    return [(fontes[i], textos[i]) for i in ordem if similaridades[i] >= MIN_SIMILARIDADE]


def ave_html():
    try:
        caminho = next((os.path.join(AQUI, n) for n in IMAGENS if os.path.exists(os.path.join(AQUI, n))), None)
        if not caminho:
            return ""
        with open(caminho, "rb") as f:
            dados = f.read()
        ext = caminho.rsplit(".", 1)[-1].lower()
        mime = {"gif": "image/gif", "png": "image/png"}.get(ext, "image/jpeg")
        src = "data:" + mime + ";base64," + base64.b64encode(dados).decode()
        return f'<div class="ave"><img src="{src}" alt="Acauã"></div>'
    except Exception:
        return ""


def modelos_instalados():
    try:
        resp = ollama.list()
        itens = resp["models"] if isinstance(resp, dict) else resp.models
        nomes = []
        for item in itens:
            nomes.append(item["model"] if isinstance(item, dict) else (getattr(item, "model", None) or item.name))
        return nomes, None
    except Exception as e:
        return [], str(e)


def tem(nome, instalados):
    return nome in instalados or f"{nome}:latest" in instalados


def gerar_resposta(pergunta, historico, contexto=""):
    if contexto:
        sistema = (
            "Você é a Acauã, uma IA amigável. Responda em português, com clareza. "
            "Para perguntas sobre os arquivos, use os trechos encontrados como fonte principal. "
            "Não mostre fontes, referências, nomes de arquivos nem uma seção chamada Fontes. "
            "Se os trechos não responderem, diga isso de forma simples e complemente com conhecimento geral quando souber.\n\n"
            f"TRECHOS DOS MATERIAIS:\n{contexto}"
        )
    else:
        sistema = (
            "Você é a Acauã, uma inteligência artificial amigável, natural e autossuficiente. "
            "Converse e responda em português claro. Use seu conhecimento geral sem consultar arquivos. "
            "Não acrescente fontes, referências, citações ou uma seção chamada Fontes. "
            "Se não souber algo, admita com honestidade, sem inventar."
        )
    mensagens = [{"role": "system", "content": sistema}] + historico
    resp = ollama.chat(model=MODELO, messages=mensagens, stream=False,
                       keep_alive=MANTER, options=OPCOES)
    return limpar_fontes(resp["message"]["content"])


st.markdown("""
<style>
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stBottom"], [data-testid="stBottom"] > div { background:#000 !important; }
#MainMenu, footer, header { visibility:hidden; }
.stApp p, .stApp span, .stApp li, .stMarkdown, [data-testid="stCaptionContainer"] { color:#fff !important; }
.ave { display:flex; justify-content:center; padding:8px 0 12px; }
.ave img { width:280px; max-width:80%; height:auto; display:block; }
[data-testid="stChatMessage"] { background:#111 !important; border:1px solid #2a2a2a; border-radius:16px; padding:10px 14px; margin:6px 0; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div { background:#fff !important; border-radius:30px; }
[data-testid="stChatInput"] textarea { background:#fff !important; color:#000 !important; -webkit-text-fill-color:#000 !important; caret-color:#000 !important; }
[data-testid="stChatInput"] textarea::placeholder { color:#666 !important; -webkit-text-fill-color:#666 !important; }
[data-testid="stChatInput"] button { background:#000 !important; }
[data-testid="stChatInput"] button svg { fill:#fff !important; color:#fff !important; }
</style>
""", unsafe_allow_html=True)
st.markdown(ave_html(), unsafe_allow_html=True)

instalados, erro_ollama = modelos_instalados()
tem_chat = tem(MODELO, instalados)
tem_emb = tem(MODELO_EMB, instalados)
if erro_ollama:
    st.error("O Ollama não está aberto. Abra o aplicativo Ollama e recarregue esta página.")
elif not tem_chat:
    st.error(f"Falta o modelo. No terminal, rode: ollama pull {MODELO}")
elif not tem_emb:
    st.warning(f"Falta o modelo de busca. No terminal, rode: ollama pull {MODELO_EMB}")

# Chave nova limpa o histórico visual antigo que poderia conter a barra de fontes.
if CHAVE_MENSAGENS not in st.session_state:
    st.session_state[CHAVE_MENSAGENS] = []
mensagens = st.session_state[CHAVE_MENSAGENS]
for item in mensagens:
    with st.chat_message(item["role"]):
        st.markdown(limpar_fontes(item["content"]))

pergunta = st.chat_input("Pergunte algo à Acauã...")
if pergunta:
    mensagens.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    historico = mensagens[-7:]
    contexto = ""
    achados = []
    deve_buscar = pede_material(pergunta)

    try:
        # Só abre/consulta a base quando a pergunta cita materiais ou quando a IA admite não saber.
        resposta = ""
        if not deve_buscar:
            resposta = gerar_resposta(pergunta, historico)
            deve_buscar = parece_nao_saber(resposta)

        if deve_buscar and tem_emb and not erro_ollama:
            fontes, textos, matriz = carregar_base(assinatura())
            if matriz is not None:
                achados = buscar(pergunta, fontes, textos, matriz)
            if achados:
                contexto = "\n\n".join(trecho for _, trecho in achados)
                resposta = gerar_resposta(pergunta, historico, contexto)
            elif not resposta:
                resposta = gerar_resposta(pergunta, historico)

        resposta = limpar_fontes(resposta)
        with st.chat_message("assistant"):
            st.markdown(resposta or "Não consegui gerar uma resposta. Tente de novo.")
        mensagens.append({"role": "assistant", "content": resposta or "Não consegui gerar uma resposta. Tente de novo."})
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Erro ao responder: {e}")
