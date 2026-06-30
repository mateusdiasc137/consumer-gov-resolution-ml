from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from string import Template

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


# ============================================================
# Configuração geral
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURES_ESPERADAS = [
    "Canal de Origem",
    "Região",
    "UF",
    "Sexo",
    "Faixa Etária",
    "Ano Abertura",
    "Mês Abertura",
    "Segmento de Mercado",
    "Área",
    "Assunto",
    "Grupo Problema",
    "Problema",
    "Como Comprou Contratou",
    "Procurou Empresa",
]

st.set_page_config(
    page_title="Consumidor.gov.br | Resolução de Reclamações",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Identidade visual (paleta, tipografia, CSS)
# ============================================================
#
# Paleta pensada para um painel de dados públicos: tons de azul/petróleo
# (confiança, institucional) com um verde-azulado de destaque para métricas
# positivas e um terracota usado com moderação em alertas. As mesmas cores
# alimentam o CSS injetado abaixo e os gráficos Plotly (PLOTLY_COLORWAY),
# para que tudo tenha a mesma cara.

CORES = {
    "primaria": "#0E3A53",
    "primaria_clara": "#1C5C82",
    "acento": "#129E8C",
    "acento_suave": "#DCF1EC",
    "meio_suave": "#FCEFC7",
    "alerta": "#D97A3D",
    "alerta_suave": "#FBE9DD",
    "fundo": "#F7F9FB",
    "cartao": "#FFFFFF",
    "texto": "#16242E",
    "texto_suave": "#51677A",
    "borda": "#E2E8EE",
}

PLOTLY_COLORWAY = [
    CORES["primaria"],
    CORES["acento"],
    CORES["alerta"],
    CORES["primaria_clara"],
    "#8A4A1E",
    CORES["texto_suave"],
]

_CSS_TEMPLATE = Template(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --cg-primary: $primaria;
    --cg-primary-light: $primaria_clara;
    --cg-accent: $acento;
    --cg-accent-soft: $acento_suave;
    --cg-mid-soft: $meio_suave;
    --cg-warn: $alerta;
    --cg-warn-soft: $alerta_suave;
    --cg-bg: $fundo;
    --cg-card: $cartao;
    --cg-text: $texto;
    --cg-text-soft: $texto_suave;
    --cg-border: $borda;
}

html, body, .stApp {
    font-family: 'Inter', -apple-system, "Segoe UI", sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Sora', -apple-system, "Segoe UI", sans-serif !important;
    color: var(--cg-text) !important;
    letter-spacing: -0.01em;
}

[data-testid="stAppViewContainer"] {
    background: var(--cg-bg);
}

section[data-testid="stSidebar"] {
    background: var(--cg-card);
    border-right: 1px solid var(--cg-border);
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: .4rem .55rem;
    border-radius: 10px;
    width: 100%;
    transition: background .15s ease;
}

section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: var(--cg-accent-soft);
}

[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button {
    font-weight: 600;
}

[data-testid="stMetric"] {
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-radius: 14px;
    padding: .8rem 1rem;
}

/* ---------- marca / cabeçalho da barra lateral ---------- */
.cg-brand {
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: .2rem 0 1rem 0;
}
.cg-brand-icon { font-size: 1.7rem; line-height: 1; }
.cg-brand-title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    color: var(--cg-text);
    line-height: 1.2;
}
.cg-brand-subtitle { font-size: .74rem; color: var(--cg-text-soft); }

/* ---------- selos de status ---------- */
.cg-pill {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    padding: .3rem .7rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
    margin: .15rem .25rem .15rem 0;
}
.cg-pill--block { display: flex; width: 100%; margin: .15rem 0; }
.cg-pill--ok   { background: var(--cg-accent-soft); color: #0B5C50; }
.cg-pill--err  { background: #FBE3DD; color: #9B3B22; }
.cg-pill--warn { background: var(--cg-warn-soft); color: #8A4A1E; }
.cg-pill--info { background: #E7EEF4; color: var(--cg-primary); }

/* ---------- cartões kpi ---------- */
.cg-kpi {
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-left: 3px solid var(--cg-primary);
    border-radius: 12px;
    padding: .85rem 1rem;
    height: 100%;
}
.cg-kpi--accent { border-left-color: var(--cg-accent); }
.cg-kpi--warn   { border-left-color: var(--cg-warn); }
.cg-kpi-label {
    display: block;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .05em;
    font-weight: 700;
    color: var(--cg-text-soft);
    margin-bottom: .2rem;
}
.cg-kpi-value {
    display: block;
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.45rem;
    color: var(--cg-text);
    line-height: 1.2;
}
.cg-kpi-sub { display: block; font-size: .78rem; color: var(--cg-text-soft); margin-top: .15rem; }

/* ---------- texto eyebrow acima de seções ---------- */
.cg-eyebrow {
    font-size: .75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--cg-accent);
    margin: .2rem 0;
}

/* ---------- hero da página inicial ---------- */
.cg-hero {
    background: linear-gradient(135deg, var(--cg-primary) 0%, #163F58 55%, var(--cg-primary-light) 100%);
    border-radius: 18px;
    padding: 2rem 2.2rem;
    color: #fff;
    margin-bottom: 1.2rem;
}
.cg-hero h1 { color: #fff !important; margin-bottom: .5rem; font-size: 1.75rem !important; }
.cg-hero p { color: #DCEAF0; font-size: 1.02rem; max-width: 46rem; margin-bottom: 0; }

/* ---------- passos numerados ---------- */
.cg-step {
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.cg-step-num {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: var(--cg-accent);
}
.cg-step-title { font-weight: 700; margin: .3rem 0 .25rem 0; color: var(--cg-text); }
.cg-step-desc { font-size: .85rem; color: var(--cg-text-soft); }

/* ---------- rodapé da barra lateral ---------- */
.cg-footer-note { color: var(--cg-text-soft); font-size: .75rem; margin-top: 1.2rem; line-height: 1.4; }
</style>
"""
)


def injetar_estilo():
    st.markdown(_CSS_TEMPLATE.substitute(CORES), unsafe_allow_html=True)


injetar_estilo()


# ============================================================
# Pequenos componentes de interface reutilizáveis
# ============================================================

def kpi(coluna, rotulo, valor, complemento=None, tom="primary"):
    """Renderiza um cartão de indicador (KPI) dentro de uma coluna do Streamlit."""
    classe_tom = {"primary": "", "accent": " cg-kpi--accent", "warn": " cg-kpi--warn"}.get(tom, "")
    complemento_html = f'<span class="cg-kpi-sub">{complemento}</span>' if complemento else ""

    coluna.markdown(
        f"""
        <div class="cg-kpi{classe_tom}">
            <span class="cg-kpi-label">{rotulo}</span>
            <span class="cg-kpi-value">{valor}</span>
            {complemento_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill_html(texto, tom="info", bloco=False):
    """Gera o HTML de um selo (badge) colorido de status."""
    classes = {"ok": "cg-pill--ok", "err": "cg-pill--err", "warn": "cg-pill--warn", "info": "cg-pill--info"}
    extra = " cg-pill--block" if bloco else ""
    return f'<span class="cg-pill {classes.get(tom, "cg-pill--info")}{extra}">{texto}</span>'


def mostrar_detalhe_tecnico(detalhe):
    """Mostra um texto técnico (mensagem de exceção, resposta de erro da API etc.) dentro de um expander discreto."""
    if detalhe:
        with st.expander("Detalhes técnicos"):
            st.code(str(detalhe), language="text")


# ============================================================
# Funções auxiliares (carregamento, formatação)
# ============================================================

def normalizar_nome(nome):
    texto = str(nome).strip().lower()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.replace("-", "_").replace(" ", "_")
    return texto


def encontrar_coluna(df, candidatos):
    if df is None or df.empty:
        return None

    mapa = {normalizar_nome(c): c for c in df.columns}

    for candidato in candidatos:
        chave = normalizar_nome(candidato)
        if chave in mapa:
            return mapa[chave]

    return None


def fmt_int(valor):
    return f"{int(valor):,}".replace(",", ".")


def fmt_pct(valor):
    if valor is None or pd.isna(valor):
        return "N/A"
    return f"{valor:.1%}".replace(".", ",")


def existe(caminho: Path):
    return caminho.exists()


def ler_csv(caminho: Path):
    if not existe(caminho):
        return pd.DataFrame()

    try:
        return pd.read_csv(caminho)
    except UnicodeDecodeError:
        return pd.read_csv(caminho, encoding="latin1")


def opcoes(dados, coluna):
    if coluna not in dados.columns:
        return ["Não informado"]

    valores = (
        dados[coluna]
        .fillna("Não informado")
        .astype(str)
        .str.strip()
        .replace("", "Não informado")
        .unique()
        .tolist()
    )

    valores = sorted(valores)

    if "Não informado" not in valores:
        valores.insert(0, "Não informado")

    return valores


# ============================================================
# Gráficos (Plotly), com o mesmo tema visual do restante do painel
# ============================================================

def aplicar_tema_plotly(fig):
    fig.update_layout(
        template="plotly_white",
        colorway=PLOTLY_COLORWAY,
        separators=",.",
        font=dict(family="Inter, sans-serif", color=CORES["texto"], size=13),
        title_font=dict(family="Sora, sans-serif", size=16, color=CORES["texto"]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    return fig


def grafico_barra(df, coluna_modelo, coluna_metrica, titulo, menor_melhor=False, mapa_cores=None):
    if df.empty or coluna_modelo is None or coluna_metrica is None:
        st.info("Dados insuficientes para gerar o gráfico.")
        return

    temp = df[[coluna_modelo, coluna_metrica]].copy()
    temp[coluna_metrica] = pd.to_numeric(temp[coluna_metrica], errors="coerce")
    temp = temp.dropna()

    if temp.empty:
        st.info("Dados insuficientes para gerar o gráfico.")
        return

    temp = temp.sort_values(coluna_metrica, ascending=menor_melhor)

    fig = px.bar(
        temp,
        x=coluna_metrica,
        y=coluna_modelo,
        orientation="h",
        text=coluna_metrica,
        title=titulo,
        color=coluna_modelo,
        color_discrete_map=mapa_cores,
        color_discrete_sequence=PLOTLY_COLORWAY,
    )

    aplicar_tema_plotly(fig)

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        yaxis_title="",
        xaxis_title="Valor",
        height=420,
        margin=dict(l=20, r=30, t=60, b=20),
        showlegend=False,
    )

    st.plotly_chart(fig, width="stretch")


def grafico_contagem(dados, coluna, titulo, top_n=10):
    if dados.empty or coluna not in dados.columns:
        st.info(f"Coluna `{coluna}` não encontrada.")
        return

    temp = (
        dados[coluna]
        .fillna("Não informado")
        .astype(str)
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    temp.columns = [coluna, "Quantidade"]
    temp = temp.sort_values("Quantidade")

    fig = px.bar(
        temp,
        x="Quantidade",
        y=coluna,
        orientation="h",
        text="Quantidade",
        title=titulo,
        color_discrete_sequence=[CORES["primaria"]],
    )

    aplicar_tema_plotly(fig)

    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        yaxis_title="",
        xaxis_title="Quantidade",
        height=max(380, 42 * len(temp)),
        margin=dict(l=20, r=30, t=60, b=20),
    )

    st.plotly_chart(fig, width="stretch")


def grafico_gauge_probabilidade(probabilidade, media_geral):
    """Velocímetro com a probabilidade estimada, comparada à média histórica geral."""
    valor = probabilidade * 100
    referencia = media_geral * 100

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=valor,
            number={"suffix": "%", "font": {"size": 42, "family": "Sora, sans-serif", "color": CORES["texto"]}},
            delta={
                "reference": referencia,
                "relative": False,
                "suffix": " p.p. vs. média geral",
                "increasing": {"color": CORES["acento"]},
                "decreasing": {"color": CORES["alerta"]},
            },
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"color": CORES["texto_suave"], "size": 11}},
                "bar": {"color": CORES["primaria"], "thickness": 0.3},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 33], "color": CORES["alerta_suave"]},
                    {"range": [33, 66], "color": CORES["meio_suave"]},
                    {"range": [66, 100], "color": CORES["acento_suave"]},
                ],
                "threshold": {
                    "line": {"color": CORES["texto"], "width": 3},
                    "thickness": 0.8,
                    "value": referencia,
                },
            },
            title={"text": "Probabilidade estimada de resolução", "font": {"size": 13, "color": CORES["texto_suave"]}},
        )
    )

    aplicar_tema_plotly(fig)
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=70, b=10))

    return fig


# ============================================================
# Carregamento dos artefatos
# ============================================================

@st.cache_resource
def carregar_modelo():
    caminho = ROOT / "models" / "modelo_final.joblib"

    if not existe(caminho):
        return None

    return joblib.load(caminho)


@st.cache_resource
def carregar_features():
    caminho = ROOT / "models" / "features.joblib"

    if not existe(caminho):
        return FEATURES_ESPERADAS

    features = joblib.load(caminho)

    if isinstance(features, dict):
        for chave in ["features", "columns", "colunas", "feature_names"]:
            if chave in features:
                features = features[chave]
                break

    try:
        features = list(features)
    except TypeError:
        features = FEATURES_ESPERADAS

    if len(features) == 0:
        features = FEATURES_ESPERADAS

    return [str(f) for f in features]


@st.cache_data
def carregar_parquet(caminho):
    if not existe(caminho):
        return pd.DataFrame()

    return pd.read_parquet(caminho)


@st.cache_data
def carregar_dados():
    train = carregar_parquet(ROOT / "data" / "modelagem" / "train.parquet")
    valid = carregar_parquet(ROOT / "data" / "modelagem" / "valid.parquet")
    test = carregar_parquet(ROOT / "data" / "modelagem" / "test.parquet")

    partes = []

    if not train.empty:
        temp = train.copy()
        temp["Conjunto"] = "Treino"
        partes.append(temp)

    if not valid.empty:
        temp = valid.copy()
        temp["Conjunto"] = "Validação"
        partes.append(temp)

    if not test.empty:
        temp = test.copy()
        temp["Conjunto"] = "Teste"
        partes.append(temp)

    if partes:
        dados = pd.concat(partes, ignore_index=True)
    else:
        dados = pd.DataFrame()

    return train, valid, test, dados


@st.cache_data
def carregar_resultados():
    validacao = ler_csv(ROOT / "results" / "resultados_validacao.csv")
    teste = ler_csv(ROOT / "results" / "resultados_teste.csv")
    tuning = ler_csv(ROOT / "results" / "tuning_validacao.csv")
    return validacao, teste, tuning


@st.cache_data
def carregar_interpretacao():
    pasta = ROOT / "results" / "interpretacao"

    return {
        "reg_aumentam": ler_csv(pasta / "regressao_top_aumentam_probabilidade.csv"),
        "reg_reduzem": ler_csv(pasta / "regressao_top_reduzem_probabilidade.csv"),
        "reg_variaveis": ler_csv(pasta / "regressao_importancia_por_variavel.csv"),
        "xgb_features": ler_csv(pasta / "xgboost_top_features.csv"),
        "xgb_variaveis": ler_csv(pasta / "xgboost_importancia_por_variavel.csv"),
    }


modelo = carregar_modelo()
features = carregar_features()
train, valid, test, dados = carregar_dados()
resultados_validacao, resultados_teste, tuning = carregar_resultados()
interpretacao = carregar_interpretacao()

target_col = encontrar_coluna(train, ["target_resolvida", "target", "resolvida"])


# ============================================================
# Integração com a Gemini API (com fallback entre modelos)
# ============================================================
#
# Endpoint e formato de payload seguem a documentação oficial do endpoint
# REST `generateContent` (https://ai.google.dev/api/generate-content):
#   POST {base}/models/{model}:generateContent
#   headers: {"x-goog-api-key": API_KEY}
#   body:    {"contents": [...], "generationConfig": {...}}
#
# A lista abaixo usa modelos Gemini atualmente disponíveis (revise em
# https://ai.google.dev/gemini-api/docs/models de tempos em tempos, pois a
# Google descontinua modelos antigos periodicamente).

GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_TIMEOUT_SEGUNDOS = 45
GEMINI_MAX_OUTPUT_TOKENS = 1024

GEMINI_MODELS_FALLBACK = [
    ("gemini-3.5-flash", "Gemini 3.5 Flash"),
    ("gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite"),
    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
]


@dataclass
class GeminiIndisponivelError(Exception):
    """Erro amigável para qualquer falha de comunicação com a Gemini API."""

    categoria: str
    mensagem_usuario: str
    detalhe_tecnico: str = ""

    def __str__(self):
        return self.mensagem_usuario


def obter_gemini_api_key():
    chave = None

    try:
        chave = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        chave = None

    if not chave:
        chave = os.getenv("GEMINI_API_KEY")

    return chave


def extrair_mensagem_erro_google(resposta):
    """Tenta extrair uma mensagem de erro legível do corpo de resposta padrão da Google."""
    try:
        corpo = resposta.json()
        erro = corpo.get("error", {})
        mensagem = erro.get("message")
        status = erro.get("status")

        if mensagem and status:
            return f"{status}: {mensagem}"

        if mensagem:
            return mensagem
    except Exception:
        pass

    return resposta.text[:300] if resposta.text else f"HTTP {resposta.status_code}"


def extrair_texto_gemini(resposta_json):
    """Extrai o texto gerado a partir do JSON de resposta do endpoint generateContent."""
    if not isinstance(resposta_json, dict):
        return "", None

    motivo_bloqueio = (resposta_json.get("promptFeedback") or {}).get("blockReason")

    candidatos = resposta_json.get("candidates") or []

    if not candidatos:
        return "", motivo_bloqueio

    candidato = candidatos[0] or {}
    motivo_finalizacao = candidato.get("finishReason")

    partes = (candidato.get("content") or {}).get("parts") or []
    texto = "".join(p.get("text", "") for p in partes if isinstance(p, dict)).strip()

    if not texto and motivo_finalizacao == "SAFETY":
        return "", "conteúdo bloqueado pelos filtros de segurança do modelo"

    return texto, motivo_bloqueio


def chamar_gemini_com_fallback(prompt: str, max_tokens: int = GEMINI_MAX_OUTPUT_TOKENS):
    """
    Envia `prompt` ao primeiro modelo de GEMINI_MODELS_FALLBACK que responder
    com sucesso e retorna (texto_gerado, nome_do_modelo_usado).

    Se todos os modelos falharem, levanta GeminiIndisponivelError com uma
    mensagem amigável (mensagem_usuario) e o detalhe técnico de cada
    tentativa (detalhe_tecnico), para a interface decidir o que exibir.
    """
    api_key = obter_gemini_api_key()

    if not api_key:
        raise GeminiIndisponivelError(
            categoria="sem_chave",
            mensagem_usuario=(
                "A chave da Gemini API não foi configurada neste ambiente, então o "
                "assistente de IA está indisponível no momento."
            ),
            detalhe_tecnico="GEMINI_API_KEY não encontrada em st.secrets nem nas variáveis de ambiente.",
        )

    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    tentativas = []

    for model_id, model_label in GEMINI_MODELS_FALLBACK:
        url = f"{GEMINI_API_BASE_URL}/models/{model_id}:generateContent"

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": max_tokens,
            },
        }

        try:
            resposta = requests.post(url, headers=headers, json=payload, timeout=GEMINI_TIMEOUT_SEGUNDOS)
        except requests.exceptions.Timeout:
            tentativas.append((model_label, "o tempo de resposta foi esgotado"))
            continue
        except requests.exceptions.ConnectionError:
            tentativas.append((model_label, "falha de conexão com a Gemini API"))
            continue
        except requests.exceptions.RequestException as erro:
            tentativas.append((model_label, str(erro)))
            continue

        if resposta.status_code == 200:
            texto, motivo = extrair_texto_gemini(resposta.json())

            if texto:
                return texto, model_label

            tentativas.append((model_label, motivo or "a resposta chegou vazia"))
            continue

        if resposta.status_code in (401, 403):
            raise GeminiIndisponivelError(
                categoria="autenticacao",
                mensagem_usuario="A chave da Gemini API foi rejeitada (erro de autenticação/autorização).",
                detalhe_tecnico=f"{model_label}: {extrair_mensagem_erro_google(resposta)}",
            )

        if resposta.status_code == 429:
            tentativas.append((model_label, "limite de requisições da API foi excedido (HTTP 429)"))
            continue

        if resposta.status_code == 404:
            tentativas.append((model_label, "modelo não encontrado nesta versão da API (HTTP 404)"))
            continue

        tentativas.append((model_label, extrair_mensagem_erro_google(resposta)))

    raise GeminiIndisponivelError(
        categoria="todos_falharam",
        mensagem_usuario="Não foi possível obter resposta de nenhum dos modelos Gemini configurados.",
        detalhe_tecnico=" | ".join(f"{nome}: {motivo}" for nome, motivo in tentativas),
    )


def exibir_erro_gemini(erro):
    """Mostra um aviso amigável para falhas da Gemini API, com o detalhe técnico disponível em um expander."""
    if isinstance(erro, GeminiIndisponivelError):
        icones = {"sem_chave": "🔑", "autenticacao": "🔒", "todos_falharam": "📡"}
        st.warning(f"{icones.get(erro.categoria, '⚠️')}  {erro.mensagem_usuario}")
        mostrar_detalhe_tecnico(erro.detalhe_tecnico)
    else:
        st.warning("📡  Não foi possível conectar à Gemini API agora. Tente novamente em alguns instantes.")
        mostrar_detalhe_tecnico(str(erro))


# ============================================================
# Explicações (template fixo + camada opcional de IA)
# ============================================================

def explicacao_template(probabilidade, media_geral, media_grupo, entrada):
    grupo = entrada["Grupo Problema"].iloc[0]
    segmento = entrada["Segmento de Mercado"].iloc[0]
    problema = entrada["Problema"].iloc[0]

    if probabilidade > media_geral + 0.03:
        comp_geral = "acima"
    elif probabilidade < media_geral - 0.03:
        comp_geral = "abaixo"
    else:
        comp_geral = "próxima"

    if probabilidade > media_grupo + 0.03:
        comp_grupo = "acima"
    elif probabilidade < media_grupo - 0.03:
        comp_grupo = "abaixo"
    else:
        comp_grupo = "próxima"

    return f"""
O modelo estimou uma probabilidade de resolução de **{fmt_pct(probabilidade)}**.

Essa estimativa está **{comp_geral} da média geral histórica**, que é de **{fmt_pct(media_geral)}**, e **{comp_grupo} da média histórica do grupo de problema selecionado**, que é de **{fmt_pct(media_grupo)}**.

A reclamação simulada pertence ao grupo **{grupo}**, ao segmento **{segmento}** e envolve o problema **{problema}**.

Essa probabilidade não garante o resultado de uma reclamação individual. Ela deve ser interpretada apenas como uma tendência estatística baseada em dados históricos.
"""


def explicacao_llm_gemini(probabilidade, media_geral, media_grupo, entrada):
    prompt = f"""
Você é um assistente que explica resultados de aprendizado de máquina em linguagem simples.

Um modelo supervisionado estimou a probabilidade de uma reclamação do Consumidor.gov.br ser avaliada como resolvida.

Resultado calculado pelo modelo:
- Probabilidade estimada: {fmt_pct(probabilidade)}
- Média geral histórica: {fmt_pct(media_geral)}
- Média do grupo de problema: {fmt_pct(media_grupo)}

Características da reclamação:
- Área: {entrada["Área"].iloc[0]}
- Segmento de Mercado: {entrada["Segmento de Mercado"].iloc[0]}
- Assunto: {entrada["Assunto"].iloc[0]}
- Grupo Problema: {entrada["Grupo Problema"].iloc[0]}
- Problema: {entrada["Problema"].iloc[0]}
- UF: {entrada["UF"].iloc[0]}
- Procurou Empresa: {entrada["Procurou Empresa"].iloc[0]}

Regras:
- Não altere a probabilidade.
- Não diga que a reclamação será resolvida com certeza.
- Não dê conselho jurídico.
- Não invente informações.
- Use no máximo 120 palavras.
- Explique que é uma estimativa estatística baseada em dados históricos.
""".strip()

    return chamar_gemini_com_fallback(prompt)


def responder_llm(pergunta):
    contexto = """
Projeto: estimativa de resolução de reclamações no Consumidor.gov.br.

Modelos avaliados:
- Regressão Logística
- Árvore de Decisão
- XGBoost
- SVM Linear calibrado

Resultados no teste de 2026:
- XGBoost teve melhor F1 macro: 0,640.
- XGBoost teve melhor ROC-AUC: 0,690.
- Regressão Logística teve melhor Brier Score: 0,214.
- Regressão Logística teve melhor Log Loss: 0,619.
- A Regressão Logística foi escolhida para o painel porque o objetivo é estimar probabilidades.
- A probabilidade estimada não é garantia de resolução.
""".strip()

    prompt = f"""
Responda com base apenas no contexto.

Contexto:
{contexto}

Pergunta:
{pergunta}

Regras:
- Responda em português.
- Use linguagem simples.
- Seja breve.
- Não dê conselho jurídico.
""".strip()

    return chamar_gemini_com_fallback(prompt)


# ============================================================
# Barra lateral
# ============================================================

PAGINAS = [
    "Início",
    "Simulador",
    "Visão geral dos dados",
    "Resultados dos modelos",
    "Interpretação",
    "Assistente LLM",
    "Sobre e limitações",
]

ICONES_PAGINA = {
    "Início": "🏠",
    "Simulador": "🎯",
    "Visão geral dos dados": "📊",
    "Resultados dos modelos": "🧪",
    "Interpretação": "🔎",
    "Assistente LLM": "🤖",
    "Sobre e limitações": "ℹ️",
}


def rotulo_pagina(p):
    return f"{ICONES_PAGINA.get(p, '')}  {p}"


def ir_para(pagina_destino):
    st.session_state["pagina_atual"] = pagina_destino


st.sidebar.markdown(
    """
    <div class="cg-brand">
        <div class="cg-brand-icon">🧾</div>
        <div>
            <div class="cg-brand-title">Consumidor.gov.br</div>
            <div class="cg-brand-subtitle">Estimativa de resolução de reclamações</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

pagina = st.sidebar.radio(
    "Navegação",
    PAGINAS,
    format_func=rotulo_pagina,
    key="pagina_atual",
    label_visibility="collapsed",
)

modo_detalhado = st.sidebar.toggle(
    "Modo detalhado",
    value=False,
    help="Exibe tabelas e métricas técnicas adicionais, voltadas a quem quer se aprofundar nos dados.",
)

st.sidebar.divider()

if modelo is None:
    st.sidebar.markdown(pill_html("❌ Modelo não carregado", "err", bloco=True), unsafe_allow_html=True)
else:
    st.sidebar.markdown(pill_html("✅ Modelo carregado", "ok", bloco=True), unsafe_allow_html=True)

if dados.empty:
    st.sidebar.markdown(pill_html("⚠️ Dados não carregados", "warn", bloco=True), unsafe_allow_html=True)
else:
    st.sidebar.markdown(
        pill_html(f"📚 {fmt_int(len(dados))} registros carregados", "info", bloco=True),
        unsafe_allow_html=True,
    )

st.sidebar.markdown(
    '<p class="cg-footer-note">As probabilidades exibidas são estimativas estatísticas e não '
    "constituem garantia de resultado para casos individuais.</p>",
    unsafe_allow_html=True,
)


# ============================================================
# Página: Início
# ============================================================

if pagina == "Início":
    st.markdown(
        """
        <div class="cg-hero">
            <h1>Estimativa de Resolução de Reclamações no Consumidor.gov.br</h1>
            <p>
                Explore os dados de reclamações registradas na plataforma Consumidor.gov.br e veja,
                com base em um modelo estatístico treinado sobre dados históricos, qual a chance
                estimada de uma reclamação com determinadas características ser avaliada como
                <strong>resolvida</strong> pelo próprio consumidor.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Melhor F1 macro", "0,640", "XGBoost", tom="accent")
    kpi(c2, "Melhor ROC-AUC", "0,690", "XGBoost", tom="accent")
    kpi(c3, "Melhor Brier Score", "0,214", "Regressão Logística", tom="primary")
    kpi(c4, "Melhor Log Loss", "0,619", "Regressão Logística", tom="primary")

    st.write("")

    col_esq, col_dir = st.columns([3, 2])

    with col_esq:
        st.markdown('<div class="cg-eyebrow">Como usar este painel</div>', unsafe_allow_html=True)
        st.markdown("#### Três passos simples")

        passos = [
            ("1", "Explore os dados", "Veja como as reclamações se distribuem por segmento, UF, grupo de problema e ano em “Visão geral dos dados”."),
            ("2", "Simule um caso", "Informe as características de uma reclamação no “Simulador” e veja a probabilidade estimada de resolução."),
            ("3", "Entenda o resultado", "Leia a explicação automática ou peça uma explicação em linguagem simples ao assistente de IA."),
        ]

        cols_passos = st.columns(3)
        for col, (num, titulo_passo, desc) in zip(cols_passos, passos):
            col.markdown(
                f"""
                <div class="cg-step">
                    <div class="cg-step-num">{num}</div>
                    <div class="cg-step-title">{titulo_passo}</div>
                    <div class="cg-step-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        b1, b2 = st.columns(2)
        with b1:
            st.button(
                "🎯  Ir para o Simulador",
                type="primary",
                width="stretch",
                on_click=ir_para,
                args=("Simulador",),
            )
        with b2:
            st.button(
                "📊  Ver visão geral dos dados",
                width="stretch",
                on_click=ir_para,
                args=("Visão geral dos dados",),
            )

    with col_dir:
        st.markdown('<div class="cg-eyebrow">Resumo do projeto</div>', unsafe_allow_html=True)
        st.markdown("#### O que este painel é (e não é)")
        st.write(
            "A tarefa foi formulada como classificação binária: a classe positiva representa "
            "reclamações avaliadas como **resolvidas** pelo consumidor."
        )
        st.success("✅ Mostra análises e probabilidades calculadas por um modelo estatístico já treinado.")
        st.info("ℹ️ O XGBoost foi o melhor classificador; a Regressão Logística foi escolhida para o simulador por ter as melhores métricas probabilísticas.")
        st.warning("⚠️ Não substitui órgãos de defesa do consumidor nem aconselhamento jurídico — veja “Sobre e limitações”.")


# ============================================================
# Página: Simulador
# ============================================================

elif pagina == "Simulador":
    st.title("🎯 Simulador de probabilidade de resolução")

    if modelo is None:
        st.error("Arquivo `models/modelo_final.joblib` não encontrado.")
        st.stop()

    if dados.empty:
        st.error("Arquivos em `data/modelagem/` não encontrados.")
        st.stop()

    st.write(
        "Preencha as características de uma reclamação simulada para ver a probabilidade "
        "estimada de ela ser avaliada como **resolvida** pelo consumidor."
    )

    with st.form("form_simulador"):
        st.markdown('<div class="cg-eyebrow">Quem é o consumidor</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sexo = st.selectbox("Sexo", opcoes(dados, "Sexo"))
        with col2:
            faixa = st.selectbox("Faixa Etária", opcoes(dados, "Faixa Etária"))
        with col3:
            regiao = st.selectbox("Região", opcoes(dados, "Região"))
        with col4:
            uf = st.selectbox("UF", opcoes(dados, "UF"))

        st.markdown(
            '<div class="cg-eyebrow" style="margin-top:.6rem;">Como a reclamação foi registrada</div>',
            unsafe_allow_html=True,
        )
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            canal = st.selectbox("Canal de Origem", opcoes(dados, "Canal de Origem"))
        with col6:
            como = st.selectbox("Como Comprou Contratou", opcoes(dados, "Como Comprou Contratou"))
        with col7:
            ano = st.selectbox("Ano Abertura", opcoes(dados, "Ano Abertura"))
        with col8:
            mes = st.selectbox("Mês Abertura", opcoes(dados, "Mês Abertura"))

        st.markdown(
            '<div class="cg-eyebrow" style="margin-top:.6rem;">Sobre o problema</div>',
            unsafe_allow_html=True,
        )
        col9, col10, col11 = st.columns(3)
        with col9:
            segmento = st.selectbox("Segmento de Mercado", opcoes(dados, "Segmento de Mercado"))
        with col10:
            area = st.selectbox("Área", opcoes(dados, "Área"))
        with col11:
            assunto = st.selectbox("Assunto", opcoes(dados, "Assunto"))

        col12, col13, col14 = st.columns(3)
        with col12:
            grupo = st.selectbox("Grupo Problema", opcoes(dados, "Grupo Problema"))
        with col13:
            problema = st.selectbox("Problema", opcoes(dados, "Problema"))
        with col14:
            procurou = st.selectbox("Procurou Empresa", opcoes(dados, "Procurou Empresa"))

        st.write("")
        usar_llm = st.checkbox("✨ Gerar explicação em linguagem simples com IA (Gemini)", value=False)
        enviar = st.form_submit_button("Estimar probabilidade", type="primary")

    if enviar:
        entrada = pd.DataFrame(
            [
                {
                    "Canal de Origem": canal,
                    "Região": regiao,
                    "UF": uf,
                    "Sexo": sexo,
                    "Faixa Etária": faixa,
                    "Ano Abertura": str(ano),
                    "Mês Abertura": str(mes),
                    "Segmento de Mercado": segmento,
                    "Área": area,
                    "Assunto": assunto,
                    "Grupo Problema": grupo,
                    "Problema": problema,
                    "Como Comprou Contratou": como,
                    "Procurou Empresa": procurou,
                }
            ]
        )

        entrada = entrada.fillna("Não informado").astype(str)
        entrada_modelo = entrada.reindex(columns=features, fill_value="Não informado")

        try:
            probabilidade = float(modelo.predict_proba(entrada_modelo)[:, 1][0])
        except Exception as erro:
            st.error(
                "Não foi possível calcular a probabilidade para esta combinação de "
                "características. Isso costuma acontecer quando as features do modelo não "
                "correspondem às colunas da entrada."
            )
            mostrar_detalhe_tecnico(erro)
            st.stop()

        if target_col and target_col in train.columns:
            media_geral = float(train[target_col].mean())

            if "Grupo Problema" in train.columns:
                medias_grupo = train.groupby("Grupo Problema")[target_col].mean().to_dict()
                media_grupo = float(medias_grupo.get(grupo, media_geral))
            else:
                media_grupo = media_geral
        else:
            media_geral = 0.0
            media_grupo = 0.0

        st.divider()

        col_gauge, col_kpis = st.columns([3, 2])

        with col_gauge:
            st.plotly_chart(grafico_gauge_probabilidade(probabilidade, media_geral), width="stretch")

        with col_kpis:
            st.write("")
            st.write("")
            k1, k2 = st.columns(2)
            kpi(k1, "Média geral histórica", fmt_pct(media_geral), "Todas as reclamações", tom="primary")
            kpi(k2, "Média do grupo selecionado", fmt_pct(media_grupo), grupo, tom="primary")
            st.write("")
            st.caption(f"Segmento: **{segmento}**  ·  UF: **{uf}**  ·  Procurou a empresa: **{procurou}**")

        st.warning(
            "⚠️ Esta probabilidade é uma estimativa estatística baseada em dados históricos. "
            "Ela não garante o resultado de uma reclamação individual."
        )

        st.markdown("#### Explicação")

        texto = explicacao_template(probabilidade, media_geral, media_grupo, entrada)
        fonte = "📐 Explicação automática (modelo fixo, sem uso de IA)"

        if usar_llm:
            with st.spinner("Gerando explicação em linguagem simples com a Gemini API..."):
                try:
                    texto, modelo_usado = explicacao_llm_gemini(probabilidade, media_geral, media_grupo, entrada)
                    fonte = f"🤖 Explicação gerada por IA ({modelo_usado})"
                except Exception as erro:
                    exibir_erro_gemini(erro)
                    st.caption("Mostrando, em vez disso, a explicação automática abaixo.")

        with st.container(border=True):
            st.caption(fonte)
            st.markdown(texto)


# ============================================================
# Página: Visão geral dos dados
# ============================================================

elif pagina == "Visão geral dos dados":
    st.title("📊 Visão geral dos dados")

    if dados.empty:
        st.error("Dados de modelagem não encontrados.")
        st.stop()

    st.write(
        "Estes números resumem a base usada para treinar e avaliar os modelos: quantos "
        "registros existem em cada conjunto e qual fração das reclamações foi avaliada como "
        "resolvida pelo consumidor."
    )

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Treino", fmt_int(len(train)), tom="primary")
    kpi(c2, "Validação", fmt_int(len(valid)), tom="primary")
    kpi(c3, "Teste", fmt_int(len(test)), tom="primary")

    if target_col:
        st.write("")
        c1, c2, c3 = st.columns(3)
        kpi(c1, "Taxa de resolução — treino", fmt_pct(train[target_col].mean()), tom="accent")
        kpi(c2, "Taxa de resolução — validação", fmt_pct(valid[target_col].mean()), tom="accent")
        kpi(c3, "Taxa de resolução — teste", fmt_pct(test[target_col].mean()), tom="accent")

    st.write("")
    st.markdown('<div class="cg-eyebrow">Como as reclamações se distribuem</div>', unsafe_allow_html=True)

    aba1, aba2, aba3, aba4 = st.tabs(["Segmentos", "Grupos", "UF", "Ano"])

    with aba1:
        st.caption("Segmentos de mercado com mais reclamações registradas na base.")
        grafico_contagem(dados, "Segmento de Mercado", "Principais segmentos de mercado", 12)

    with aba2:
        st.caption("Tipos de problema mais frequentes entre as reclamações.")
        grafico_contagem(dados, "Grupo Problema", "Principais grupos de problema", 12)

    with aba3:
        st.caption("Unidades federativas com mais reclamações registradas.")
        grafico_contagem(dados, "UF", "Distribuição por UF", 15)

    with aba4:
        st.caption("Quantidade de reclamações por ano de abertura.")
        grafico_contagem(dados, "Ano Abertura", "Distribuição por ano", 15)


# ============================================================
# Página: Resultados dos modelos
# ============================================================

elif pagina == "Resultados dos modelos":
    st.title("🧪 Resultados dos modelos")

    if resultados_teste.empty:
        st.error("Arquivo `results/resultados_teste.csv` não encontrado.")
        st.stop()

    col_modelo = encontrar_coluna(resultados_teste, ["modelo", "model"])
    col_f1 = encontrar_coluna(resultados_teste, ["f1_macro", "F1 macro", "f1"])
    col_auc = encontrar_coluna(resultados_teste, ["roc_auc", "ROC-AUC"])
    col_brier = encontrar_coluna(resultados_teste, ["brier_score", "Brier Score", "brier"])
    col_log = encontrar_coluna(resultados_teste, ["log_loss", "Log Loss", "logloss"])

    st.info(
        "📌 **Resumo:** o **XGBoost** teve o melhor desempenho de classificação (F1 macro e "
        "ROC-AUC), mas a **Regressão Logística** teve as melhores métricas probabilísticas "
        "(Brier Score e Log Loss) — por isso ela foi a escolhida para o Simulador, cujo "
        "objetivo é estimar uma probabilidade bem calibrada, e não apenas acertar a classe."
    )

    mapa_cores_modelos = None
    if col_modelo:
        nomes_modelos = sorted(resultados_teste[col_modelo].dropna().astype(str).unique())
        mapa_cores_modelos = {
            nome: PLOTLY_COLORWAY[i % len(PLOTLY_COLORWAY)] for i, nome in enumerate(nomes_modelos)
        }

    aba_teste, aba_validacao = st.tabs(["Teste (2026)", "Validação"])

    with aba_teste:
        st.caption(
            "Para F1 macro e ROC-AUC, valores maiores são melhores. Para Brier Score e Log "
            "Loss, valores menores são melhores."
        )

        cols = [c for c in [col_modelo, col_f1, col_auc, col_brier, col_log] if c]
        tabela = resultados_teste[cols].copy()

        tabela = tabela.rename(
            columns={
                col_modelo: "Modelo",
                col_f1: "F1 macro",
                col_auc: "ROC-AUC",
                col_brier: "Brier Score",
                col_log: "Log Loss",
            }
        )

        st.dataframe(tabela, width="stretch", hide_index=True)

        sub1, sub2 = st.tabs(["Gráficos interativos", "Gráficos salvos"])

        with sub1:
            grafico_barra(resultados_teste, col_modelo, col_f1, "Comparação do F1 macro", mapa_cores=mapa_cores_modelos)
            grafico_barra(resultados_teste, col_modelo, col_auc, "Comparação da ROC-AUC", mapa_cores=mapa_cores_modelos)
            grafico_barra(
                resultados_teste, col_modelo, col_brier, "Comparação do Brier Score",
                menor_melhor=True, mapa_cores=mapa_cores_modelos,
            )
            grafico_barra(
                resultados_teste, col_modelo, col_log, "Comparação do Log Loss",
                menor_melhor=True, mapa_cores=mapa_cores_modelos,
            )

        with sub2:
            pasta = ROOT / "results" / "graficos"

            arquivos = [
                "f1_macro_teste.png",
                "roc_auc_teste.png",
                "brier_score_teste.png",
                "log_loss_teste.png",
                "matriz_confusao_xgboost.png",
                "matriz_confusao_regressao_logistica.png",
                "curva_calibracao_modelo_final.png",
            ]

            algum_encontrado = False
            for nome in arquivos:
                caminho = pasta / nome
                if caminho.exists():
                    algum_encontrado = True
                    st.image(str(caminho), caption=nome, width="stretch")

            if not algum_encontrado:
                st.info("Nenhuma imagem encontrada em `results/graficos/`.")

    with aba_validacao:
        if resultados_validacao.empty:
            st.info("Arquivo `results/resultados_validacao.csv` não encontrado.")
        else:
            st.caption("As mesmas métricas, calculadas sobre o conjunto de validação.")

            col_modelo_v = encontrar_coluna(resultados_validacao, ["modelo", "model"])
            col_f1_v = encontrar_coluna(resultados_validacao, ["f1_macro", "F1 macro", "f1"])
            col_auc_v = encontrar_coluna(resultados_validacao, ["roc_auc", "ROC-AUC"])
            col_brier_v = encontrar_coluna(resultados_validacao, ["brier_score", "Brier Score", "brier"])
            col_log_v = encontrar_coluna(resultados_validacao, ["log_loss", "Log Loss", "logloss"])

            cols_v = [c for c in [col_modelo_v, col_f1_v, col_auc_v, col_brier_v, col_log_v] if c]
            tabela_v = resultados_validacao[cols_v].copy()

            tabela_v = tabela_v.rename(
                columns={
                    col_modelo_v: "Modelo",
                    col_f1_v: "F1 macro",
                    col_auc_v: "ROC-AUC",
                    col_brier_v: "Brier Score",
                    col_log_v: "Log Loss",
                }
            )

            st.dataframe(tabela_v, width="stretch", hide_index=True)

            grafico_barra(resultados_validacao, col_modelo_v, col_f1_v, "F1 macro (validação)", mapa_cores=mapa_cores_modelos)
            grafico_barra(resultados_validacao, col_modelo_v, col_auc_v, "ROC-AUC (validação)", mapa_cores=mapa_cores_modelos)

    if modo_detalhado and not tuning.empty:
        with st.expander("🔧 Resultados de tuning de hiperparâmetros (avançado)", expanded=False):
            st.dataframe(tuning, width="stretch", hide_index=True)


# ============================================================
# Página: Interpretação
# ============================================================

elif pagina == "Interpretação":
    st.title("🔎 Interpretação das variáveis")

    st.write(
        "As importâncias apresentadas indicam **associações estatísticas** aprendidas pelos "
        "modelos a partir dos dados históricos. Elas não devem ser interpretadas como "
        "relações de causa e efeito."
    )

    reg_var = interpretacao["reg_variaveis"]
    xgb_var = interpretacao["xgb_variaveis"]
    reg_aum = interpretacao["reg_aumentam"]
    reg_red = interpretacao["reg_reduzem"]
    xgb_feat = interpretacao["xgb_features"]

    aba1, aba2, aba3 = st.tabs(["Importância por variável", "Categorias", "Features XGBoost"])

    with aba1:
        st.caption(
            "Quanto maior a importância de uma variável, mais ela influencia, em média, as "
            "previsões do modelo — em qualquer direção."
        )
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("##### Regressão Logística")
            if reg_var.empty:
                st.info("Arquivo de interpretação não encontrado.")
            else:
                st.dataframe(reg_var, width="stretch", hide_index=True)

        with c2:
            st.markdown("##### XGBoost")
            if xgb_var.empty:
                st.info("Arquivo de interpretação não encontrado.")
            else:
                st.dataframe(xgb_var, width="stretch", hide_index=True)

    with aba2:
        st.caption(
            "Categorias específicas que, segundo a Regressão Logística, mais aumentam ou "
            "reduzem a probabilidade estimada de resolução."
        )
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("##### ⬆️ Aumentam a probabilidade")
            if reg_aum.empty:
                st.info("Arquivo não encontrado.")
            else:
                st.dataframe(reg_aum, width="stretch", hide_index=True)

        with c2:
            st.markdown("##### ⬇️ Reduzem a probabilidade")
            if reg_red.empty:
                st.info("Arquivo não encontrado.")
            else:
                st.dataframe(reg_red, width="stretch", hide_index=True)

    with aba3:
        st.caption("Importância das variáveis (em formato codificado) segundo o modelo XGBoost.")
        if xgb_feat.empty:
            st.info("Arquivo não encontrado.")
        else:
            st.dataframe(xgb_feat, width="stretch", hide_index=True)


# ============================================================
# Página: Assistente LLM
# ============================================================

elif pagina == "Assistente LLM":
    st.title("🤖 Assistente de IA")

    st.write(
        "Esta página usa a Gemini API para responder, em linguagem simples, perguntas sobre o "
        "projeto e os resultados já calculados. A IA **não** calcula probabilidades, não "
        "altera resultados e não substitui o modelo supervisionado treinado."
    )

    with st.expander("⚙️ Configuração da Gemini API", expanded=modo_detalhado):
        if obter_gemini_api_key():
            st.markdown(pill_html("🔑 Uma chave de API foi encontrada neste ambiente.", "ok"), unsafe_allow_html=True)
        else:
            st.markdown(pill_html("🔑 Nenhuma chave de API foi encontrada neste ambiente.", "warn"), unsafe_allow_html=True)

        st.write("Configure a chave de uma das formas abaixo, a partir da raiz do projeto:")

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Opção 1 — variável de ambiente")
            st.code('export GEMINI_API_KEY="sua_chave_aqui"\nstreamlit run app/streamlit_app.py', language="bash")
        with col_b:
            st.caption("Opção 2 — `.streamlit/secrets.toml`")
            st.code('GEMINI_API_KEY = "sua_chave_aqui"', language="toml")

        st.write("Ordem de fallback entre modelos Gemini:")
        for i, (model_id, label) in enumerate(GEMINI_MODELS_FALLBACK, start=1):
            st.caption(f"{i}. {label}  ·  `{model_id}`")

        if st.button("🔧 Testar conexão com a Gemini API"):
            with st.spinner("Testando conexão..."):
                try:
                    _, modelo_usado = chamar_gemini_com_fallback(
                        "Responda apenas com a palavra: ok", max_tokens=10
                    )
                    st.success(f"✅ Conexão funcionando — resposta recebida de **{modelo_usado}**.")
                except Exception as erro:
                    exibir_erro_gemini(erro)

    st.markdown('<div class="cg-eyebrow">Perguntas frequentes</div>', unsafe_allow_html=True)

    pergunta = st.selectbox(
        "Escolha uma pergunta",
        [
            "O que significa F1 macro?",
            "Por que o XGBoost foi o melhor classificador?",
            "Por que a Regressão Logística foi escolhida para o painel?",
            "O que significa Brier Score?",
            "O que significa Log Loss?",
            "Por que a acurácia não foi a métrica principal?",
            "Quais são as limitações do modelo?",
        ],
    )

    pergunta_livre = st.text_input(
        "Ou digite sua própria pergunta sobre o projeto (opcional)",
        placeholder="Ex.: como os dados de 2026 foram divididos entre treino e teste?",
    )

    pergunta_final = pergunta_livre.strip() or pergunta

    if st.button("Responder com Gemini", type="primary"):
        with st.spinner("Gerando resposta com a Gemini API..."):
            try:
                resposta, modelo_usado = responder_llm(pergunta_final)
                with st.container(border=True):
                    st.caption(f"🤖 Resposta gerada por {modelo_usado}")
                    st.markdown(resposta)
            except Exception as erro:
                exibir_erro_gemini(erro)


# ============================================================
# Página: Sobre e limitações
# ============================================================

elif pagina == "Sobre e limitações":
    st.title("ℹ️ Sobre e limitações")

    st.subheader("📌 O que o painel faz")
    st.write(
        "O painel permite explorar os dados processados, visualizar os resultados dos "
        "modelos e simular a probabilidade estimada de resolução de uma reclamação."
    )

    st.subheader("🚫 O que o painel não faz")
    st.write(
        "O painel não substitui órgãos de defesa do consumidor, aconselhamento jurídico ou "
        "indicadores oficiais do Consumidor.gov.br."
    )

    st.subheader("🤖 Uso da Gemini API")
    st.write(
        "A Gemini API é usada apenas para gerar explicações textuais em linguagem acessível. "
        "Ela não calcula a probabilidade, não altera os resultados e não substitui o modelo "
        "supervisionado treinado."
    )

    st.subheader("⚖️ Cuidados éticos")
    st.write(
        "As probabilidades estimadas são associações estatísticas baseadas em dados "
        "históricos. Elas não garantem o resultado de reclamações individuais."
    )

    st.subheader("🗂️ Arquivos esperados")

    arquivos = [
        "models/modelo_final.joblib",
        "models/features.joblib",
        "data/modelagem/train.parquet",
        "data/modelagem/valid.parquet",
        "data/modelagem/test.parquet",
        "results/resultados_validacao.csv",
        "results/resultados_teste.csv",
    ]

    for arq in arquivos:
        if (ROOT / arq).exists():
            st.markdown(pill_html(f"✅ {arq}", "ok"), unsafe_allow_html=True)
        else:
            st.markdown(pill_html(f"❌ {arq}", "err"), unsafe_allow_html=True)
