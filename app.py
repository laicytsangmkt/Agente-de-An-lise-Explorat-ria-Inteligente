import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import json
import base64
from datetime import datetime

# --- 1. DEFINIÇÃO DO PREFIXO (INSTRUÇÕES PARA O AGENTE) ---
PREFIXO_AGENTE_MELHORADO = """
Você é um agente de análise de dados especialista em Python e Pandas, projetado para ser extremamente metódico e claro.

REGRAS DE OURO PARA O SEU RACIOCÍNIO:
1.  **Pense Passo a Passo:** Antes de escrever qualquer código, sempre explique seu plano de ação em etapas simples. Ex: "Para responder a isso, primeiro vou verificar os tipos de dados das colunas. Em segundo lugar, vou calcular a média da coluna 'idade'. Por fim, vou apresentar o resultado."
2.  **Divida Perguntas Complexas:** Se a pergunta do usuário for ampla (ex: "analise os dados"), divida-a em partes menores e execute uma de cada vez. Informe ao usuário o que você está fazendo. Ex: "Essa é uma pergunta ampla. Vou começar com uma descrição geral dos dados (estatísticas descritivas)."
3.  **Peça Esclarecimentos:** Se uma pergunta for ambígua (ex: "mostre as vendas"), peça ao usuário para esclarecer. Ex: "Para analisar as 'vendas', você gostaria de ver a soma total, a média, ou a tendência ao longo do tempo (diária, mensal)?" Não presuma.
4.  **Código Simples e Focado:** Gere o código Python mais simples e direto possível para cada etapa. Evite criar códigos muito longos ou complexos em uma única etapa.
5.  **Verificação Inicial é Obrigatória:** Para a primeira pergunta do usuário, sua primeira ação DEVE SER SEMPRE inspecionar o dataframe com `df.info()` e `df.head()` para entender a estrutura, colunas, tipos de dados e valores ausentes. Isso é crucial para todas as análises futuras.

Agora, comece a interagir com o usuário sobre o dataframe fornecido. Você tem uma ferramenta para executar código Python.
"""

# --- SISTEMA DE MEMÓRIA (NOVO) ---
class MemoriaAnalise:
    """Gerencia a memória de conclusões e insights do agente."""
    
    def __init__(self):
        self.conclusoes = []
    
    def adicionar_conclusao(self, pergunta, resposta):
        """Adiciona uma nova conclusão à memória."""
        # Gera conclusão automática
        conclusao_gerada = self._gerar_conclusao_automatica(pergunta, resposta)
        
        entrada = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pergunta": pergunta,
            "conclusao": conclusao_gerada
        }
        self.conclusoes.append(entrada)
    
    def _gerar_conclusao_automatica(self, pergunta, resposta):
        """Gera uma conclusão automática baseada na pergunta e resposta."""
        import re
        
        # Extrai números relevantes
        numeros = re.findall(r'\b\d+\.?\d*\b', resposta)
        
        # Detecta palavras-chave
        palavras_chave = {
            'média': 'calculou média',
            'soma': 'calculou soma',
            'total': 'calculou total',
            'distribuição': 'analisou distribuição',
            'correlação': 'analisou correlação',
            'valores ausentes': 'identificou valores ausentes',
            'missing': 'identificou valores ausentes',
            'mínimo': 'identificou valor mínimo',
            'máximo': 'identificou valor máximo',
        }
        
        resposta_lower = resposta.lower()
        tipo_analise = "Análise realizada"
        
        for palavra, descricao in palavras_chave.items():
            if palavra in resposta_lower or palavra in pergunta.lower():
                tipo_analise = descricao
                break
        
        if numeros:
            return f"{tipo_analise}: {pergunta[:60]}... | Valores: {', '.join(numeros[:3])}"
        else:
            return f"{tipo_analise}: {pergunta[:80]}..."
    
    def obter_resumo_conclusoes(self):
        """Retorna um resumo de todas as conclusões."""
        if not self.conclusoes:
            return "🤔 **Nenhuma análise registrada ainda.**\n\nFaça perguntas sobre seus dados e eu vou memorizar as descobertas!"
        
        resumo = f"📊 **MEMÓRIA DO AGENTE - {len(self.conclusoes)} Análises Realizadas**\n\n"
        
        for i, entrada in enumerate(self.conclusoes, 1):
            resumo += f"**{i}. {entrada['pergunta'][:60]}{'...' if len(entrada['pergunta']) > 60 else ''}**\n"
            resumo += f"   💡 {entrada['conclusao']}\n"
            resumo += f"   🕐 {entrada['timestamp']}\n\n"
        
        return resumo
    
    def serializar(self):
        """Converte a memória para string base64."""
        return base64.b64encode(json.dumps(self.conclusoes).encode()).decode()
    
    @staticmethod
    def deserializar(string_codificada):
        """Reconstrói a memória de uma string base64."""
        if not string_codificada:
            return MemoriaAnalise()
        try:
            memoria = MemoriaAnalise()
            memoria.conclusoes = json.loads(base64.b64decode(string_codificada.encode()).decode())
            return memoria
        except:
            return MemoriaAnalise()

# --- Funções para codificar e decodificar o histórico do chat ---
def serializar_chat(mensagens):
    """Converte a lista de mensagens em uma string base64 segura para URL."""
    if not mensagens:
        return ""
    mensagens_sem_figura = [
        {"role": m["role"], "content": m["content"]} for m in mensagens
    ]
    return base64.b64encode(json.dumps(mensagens_sem_figura).encode()).decode()

def deserializar_chat(string_codificada):
    """Converte a string da URL de volta para uma lista de mensagens."""
    if not string_codificada:
        return []
    try:
        mensagens_decodificadas = json.loads(base64.b64decode(string_codificada.encode()).decode())
        for m in mensagens_decodificadas:
            m['figure'] = None
        return mensagens_decodificadas
    except:
        return []

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Agente de Análise de CSV",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente Autônomo para Análise de Dados em CSV")
st.write("Esta aplicação utiliza um agente de IA para responder perguntas sobre arquivos CSV.")

def carregar_e_processar_csv(arquivo_csv):
    try:
        df = pd.read_csv(arquivo_csv)
        return df
    except UnicodeDecodeError:
        st.warning("Falha na decodificação UTF-8. Tentando com 'latin1'.")
        arquivo_csv.seek(0)
        df = pd.read_csv(arquivo_csv, encoding='latin1')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo CSV: {e}")
        return None

# --- Inicialização do Estado da Sessão ---
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'memoria' not in st.session_state:
    st.session_state.memoria = MemoriaAnalise()

# --- Inicialização do chat a partir da URL ---
query_params = st.query_params.to_dict()
if "messages" not in st.session_state:
    try:
        chat_param = st.query_params.get("chat", "")
        st.session_state.messages = deserializar_chat(chat_param)
        
        # Restaura a memória da URL
        memoria_param = st.query_params.get("memoria", "")
        st.session_state.memoria = MemoriaAnalise.deserializar(memoria_param)
    except:
        st.session_state.messages = []
        st.session_state.memoria = MemoriaAnalise()

# --- Configuração da API Key ---
try:
    st.session_state.google_api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    st.sidebar.warning("A chave da API do Google não foi encontrada nos segredos. Por favor, insira-a abaixo.")
    api_key_input = st.sidebar.text_input("Chave da API do Google", type="password")
    if api_key_input:
        st.session_state.google_api_key = api_key_input
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.sidebar.success("API Key configurada!")

# --- Barra Lateral ---
with st.sidebar:
    st.header("Upload do Arquivo")
    arquivo_csv = st.file_uploader("Selecione um arquivo CSV", type=["csv"])

    if arquivo_csv is not None and st.session_state.get('uploaded_file_name') != arquivo_csv.name:
        st.session_state.df = carregar_e_processar_csv(arquivo_csv)
        
        if st.session_state.df is not None:
            st.success(f"Arquivo '{arquivo_csv.name}' carregado!")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
            
            st.session_state.uploaded_file_name = arquivo_csv.name
            st.session_state.agent = None
            st.session_state.messages = []
            st.session_state.memoria = MemoriaAnalise()
            st.query_params.clear()
            st.rerun()
    
    # --- NOVA SEÇÃO: Visualização da Memória ---
    if st.session_state.memoria.conclusoes:
        st.divider()
        st.header("🧠 Memória do Agente")
        
        num_conclusoes = len(st.session_state.memoria.conclusoes)
        st.metric("Análises Registradas", num_conclusoes)
        
        with st.expander(f"📋 Ver Todas ({num_conclusoes})", expanded=False):
            st.markdown(st.session_state.memoria.obter_resumo_conclusoes())
        
        if st.button("🗑️ Limpar Memória", use_container_width=True):
            st.session_state.memoria = MemoriaAnalise()
            st.query_params["memoria"] = ""
            st.success("Memória limpa!")
            st.rerun()

# --- Lógica Principal da Aplicação ---
if st.session_state.google_api_key and st.session_state.df is not None:
    
    if st.session_state.agent is None:
        st.info("Inicializando o agente de IA com novas instruções...")
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                convert_system_message_to_human=True,
                api_version="v1"
            )
            st.session_state.agent = create_pandas_dataframe_agent(
                llm=llm,
                df=st.session_state.df,
                agent_type='tool-calling',
                prefix=PREFIXO_AGENTE_MELHORADO,
                verbose=True,
                handle_parsing_errors=True,
                agent_executor_kwargs={"handle_parsing_errors": True},
                allow_dangerous_code=True
            )
            st.success("Agente pronto para conversar!")
        except Exception as e:
            st.error(f"Erro ao criar o agente: {e}")
            st.stop()

    st.header("Converse com seus Dados")

    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Olá! Sou seu assistente de análise de dados. O que você gostaria de saber sobre este arquivo?",
            "figure": None
        })

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "figure" in message and message["figure"] is not None:
                st.pyplot(message["figure"])

    if prompt := st.chat_input("Qual a distribuição da variável 'idade'?"):
        
        # Detecta perguntas sobre memória
        palavras_chave_memoria = ["conclusões", "conclusoes", "insights", "descobrimos", 
                                   "aprendemos", "resumo", "análises anteriores", "memória"]
        pergunta_sobre_memoria = any(palavra in prompt.lower() for palavra in palavras_chave_memoria)
        
        if pergunta_sobre_memoria and st.session_state.memoria.conclusoes:
            st.session_state.messages.append({"role": "user", "content": prompt, "figure": None})
            resposta_memoria = st.session_state.memoria.obter_resumo_conclusoes()
            st.session_state.messages.append({"role": "assistant", "content": resposta_memoria, "figure": None})
            
            st.query_params["chat"] = serializar_chat(st.session_state.messages)
            st.query_params["memoria"] = st.session_state.memoria.serializar()
            st.rerun()
        
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.query_params["chat"] = serializar_chat(st.session_state.messages)

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("O agente está pensando..."):
                    try:
                        plt.close('all')
                        chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
                        response = st.session_state.agent.invoke({"input": prompt, "chat_history": chat_history})
                        output_text = response["output"]
                        fig = plt.gcf()
                        has_plot = any(ax.has_data() for ax in fig.get_axes()) if fig else False

                        if has_plot:
                            st.pyplot(fig)
                            st.session_state.messages.append({"role": "assistant", "content": output_text, "figure": fig})
                        else:
                            plt.close(fig)
                            st.markdown(output_text)
                            st.session_state.messages.append({"role": "assistant", "content": output_text, "figure": None})
                        
                        # ADICIONA À MEMÓRIA (silenciosamente)
                        st.session_state.memoria.adicionar_conclusao(prompt, output_text)

                    except Exception as e:
                        error_message = f"Ocorreu um erro: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message, "figure": None})
            
            # Atualiza URL com chat e memória
            st.query_params["chat"] = serializar_chat(st.session_state.messages)
            st.query_params["memoria"] = st.session_state.memoria.serializar()

else:
    st.info("Por favor, configure a API Key e faça o upload de um arquivo CSV na barra lateral para começar.")