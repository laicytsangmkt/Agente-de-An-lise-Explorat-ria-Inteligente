import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import json
import base64
from datetime import datetime

# --- PREFIXO DO AGENTE ---
PREFIXO_AGENTE_MELHORADO = """
Você é um agente de análise de dados especialista em Python e Pandas, projetado para ser extremamente metódico e claro.

REGRAS DE OURO PARA O SEU RACIOCÍNIO:
1.  **Pense Passo a Passo:** Antes de escrever qualquer código, sempre explique seu plano de ação em etapas simples.
2.  **Divida Perguntas Complexas:** Se a pergunta do usuário for ampla, divida-a em partes menores e execute uma de cada vez.
3.  **Peça Esclarecimentos:** Se uma pergunta for ambígua, peça ao usuário para esclarecer.
4.  **Código Simples e Focado:** Gere o código Python mais simples e direto possível para cada etapa.
5.  **Verificação Inicial é Obrigatória:** Para a primeira pergunta, inspecione o dataframe com `df.info()` e `df.head()`.

Agora, comece a interagir com o usuário sobre o dataframe fornecido.
"""

# --- SISTEMA DE MEMÓRIA ---
class MemoriaAnalise:
    """Gerencia a memória de conclusões e insights do agente."""
    
    def __init__(self):
        self.conclusoes = []
        self.metadados_dataset = {}
    
    def adicionar_conclusao(self, pergunta, resposta, conclusao_gerada):
        """Adiciona uma nova conclusão à memória."""
        entrada = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pergunta": pergunta,
            "resposta_completa": resposta[:500],  # Primeiros 500 chars
            "conclusao": conclusao_gerada
        }
        self.conclusoes.append(entrada)
    
    def gerar_conclusao_automatica(self, pergunta, resposta):
        """Gera uma conclusão automática baseada na pergunta e resposta."""
        # Extrai informações numéricas e palavras-chave
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
            'desvio': 'calculou desvio padrão',
            'mediana': 'calculou mediana',
            'contagem': 'fez contagem',
            'agrupamento': 'realizou agrupamento',
            'filtro': 'aplicou filtro',
        }
        
        conclusao_parts = []
        resposta_lower = resposta.lower()
        
        # Detecta tipo de análise
        for palavra, descricao in palavras_chave.items():
            if palavra in resposta_lower or palavra in pergunta.lower():
                conclusao_parts.append(descricao)
                break
        
        # Extrai números relevantes (valores entre 0-9 com decimais)
        import re
        numeros = re.findall(r'\b\d+\.?\d*\b', resposta)
        if numeros:
            conclusao_parts.append(f"valores encontrados: {', '.join(numeros[:3])}")
        
        if conclusao_parts:
            return f"Análise: {pergunta[:50]}... - {' | '.join(conclusao_parts)}"
        else:
            return f"Análise realizada sobre: {pergunta[:80]}..."
    
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
    
    def obter_contexto_para_agente(self):
        """Retorna as conclusões em formato para o agente usar como contexto."""
        if not self.conclusoes:
            return ""
        
        contexto = "\n\n--- ANÁLISES ANTERIORES NESTA SESSÃO ---\n"
        for entrada in self.conclusoes[-3:]:  # Últimas 3 análises
            contexto += f"• Pergunta: {entrada['pergunta']}\n"
            contexto += f"  Conclusão: {entrada['conclusao']}\n"
        contexto += "--- FIM DO HISTÓRICO ---\n\n"
        return contexto
    
    def serializar(self):
        """Converte a memória para string base64."""
        dados = {
            "conclusoes": self.conclusoes,
            "metadados": self.metadados_dataset
        }
        return base64.b64encode(json.dumps(dados).encode()).decode()
    
    @staticmethod
    def deserializar(string_codificada):
        """Reconstrói a memória de uma string base64."""
        if not string_codificada:
            return MemoriaAnalise()
        try:
            dados = json.loads(base64.b64decode(string_codificada.encode()).decode())
            memoria = MemoriaAnalise()
            memoria.conclusoes = dados.get("conclusoes", [])
            memoria.metadados_dataset = dados.get("metadados", {})
            return memoria
        except:
            return MemoriaAnalise()

# --- Funções de serialização do chat ---
def serializar_chat(mensagens):
    if not mensagens:
        return ""
    mensagens_sem_figura = [
        {"role": m["role"], "content": m["content"]} for m in mensagens
    ]
    return base64.b64encode(json.dumps(mensagens_sem_figura).encode()).decode()

def deserializar_chat(string_codificada):
    if not string_codificada:
        return []
    try:
        mensagens_decodificadas = json.loads(base64.b64decode(string_codificada.encode()).decode())
        for m in mensagens_decodificadas:
            m['figure'] = None
        return mensagens_decodificadas
    except:
        return []

# --- Configuração da Página ---
st.set_page_config(
    page_title="Agente de Análise de CSV",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente Autônomo para Análise de Dados em CSV")
st.write("Esta aplicação utiliza um agente de IA com memória persistente para análise de dados.")

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

# Inicialização do chat e memória a partir da URL
query_params = st.query_params.to_dict()
if "messages" not in st.session_state:
    try:
        chat_param = st.query_params.get("chat", "")
        st.session_state.messages = deserializar_chat(chat_param)
        
        # Restaura a memória da URL também
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
    st.sidebar.warning("A chave da API do Google não foi encontrada nos segredos.")
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
    
    # --- SEÇÃO: Visualização da Memória ---
    st.divider()
    st.header("🧠 Memória do Agente")
    
    num_conclusoes = len(st.session_state.memoria.conclusoes)
    st.metric("Análises Memoradas", num_conclusoes)
    
    if num_conclusoes > 0:
        with st.expander(f"📋 Ver Todas as Análises ({num_conclusoes})", expanded=False):
            st.markdown(st.session_state.memoria.obter_resumo_conclusoes())
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Limpar Memória", use_container_width=True):
                st.session_state.memoria = MemoriaAnalise()
                st.query_params["memoria"] = ""
                st.success("Memória limpa!")
                st.rerun()
        
        with col2:
            if st.button("📥 Exportar Memória", use_container_width=True):
                dados_export = json.dumps(st.session_state.memoria.conclusoes, indent=2, ensure_ascii=False)
                st.download_button(
                    label="💾 Download JSON",
                    data=dados_export,
                    file_name=f"memoria_analise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    else:
        st.info("Faça análises para começar a construir a memória!")

# --- Lógica Principal ---
if st.session_state.google_api_key and st.session_state.df is not None:
    
    if st.session_state.agent is None:
        st.info("Inicializando o agente de IA com sistema de memória...")
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0,
                convert_system_message_to_human=True
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
            st.success("✅ Agente pronto para conversar!")
        except Exception as e:
            st.error(f"Erro ao criar o agente: {e}")
            st.stop()

    st.header("💬 Converse com seus Dados")

    if not st.session_state.messages:
        mensagem_inicial = "Olá! Sou seu assistente de análise de dados com **memória persistente**. \n\n"
        mensagem_inicial += "🧠 Vou memorizar todas as análises que fizermos juntos!\n\n"
        mensagem_inicial += "Você pode me perguntar a qualquer momento: *'Quais conclusões tiramos até agora?'*"
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": mensagem_inicial,
            "figure": None
        })

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "figure" in message and message["figure"] is not None:
                st.pyplot(message["figure"])

    if prompt := st.chat_input("Faça sua pergunta sobre os dados..."):
        
        # Detecta se o usuário está perguntando sobre conclusões anteriores
        palavras_chave_memoria = ["conclusões", "conclusoes", "insights", "descobrimos", "aprendemos", 
                                   "resumo", "análises anteriores", "o que já fizemos", "memória", "memoria"]
        pergunta_sobre_memoria = any(palavra in prompt.lower() for palavra in palavras_chave_memoria)
        
        if pergunta_sobre_memoria and st.session_state.memoria.conclusoes:
            # Responde diretamente com o resumo da memória
            st.session_state.messages.append({"role": "user", "content": prompt, "figure": None})
            resposta_memoria = st.session_state.memoria.obter_resumo_conclusoes()
            st.session_state.messages.append({"role": "assistant", "content": resposta_memoria, "figure": None})
            
            st.query_params["chat"] = serializar_chat(st.session_state.messages)
            st.query_params["memoria"] = st.session_state.memoria.serializar()
            st.rerun()
        
        else:
            # Análise normal com o agente
            st.session_state.messages.append({"role": "user", "content": prompt, "figure": None})
            st.query_params["chat"] = serializar_chat(st.session_state.messages)

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤔 O agente está pensando..."):
                    try:
                        plt.close('all')
                        
                        # Adiciona contexto de conclusões anteriores
                        contexto = st.session_state.memoria.obter_contexto_para_agente()
                        prompt_com_contexto = contexto + prompt if contexto else prompt
                        
                        chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                        response = st.session_state.agent.invoke({
                            "input": prompt_com_contexto, 
                            "chat_history": chat_history
                        })
                        output_text = response["output"]
                        
                        # FORÇA a geração de conclusão automática
                        conclusao_gerada = st.session_state.memoria.gerar_conclusao_automatica(prompt, output_text)
                        st.session_state.memoria.adicionar_conclusao(prompt, output_text, conclusao_gerada)
                        
                        fig = plt.gcf()
                        has_plot = any(ax.has_data() for ax in fig.get_axes()) if fig.get_axes() else False

                        if has_plot:
                            st.pyplot(fig)
                            st.session_state.messages.append({"role": "assistant", "content": output_text, "figure": fig})
                        else:
                            plt.close(fig)
                            st.markdown(output_text)
                            st.session_state.messages.append({"role": "assistant", "content": output_text, "figure": None})
                        
                        # Mostra toast de nova conclusão adicionada
                        st.toast(f"💾 Análise memorizada: {conclusao_gerada[:50]}...", icon="🧠")

                    except Exception as e:
                        error_message = f"❌ Ocorreu um erro: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message, "figure": None})
            
            # Atualiza URL com chat e memória
            st.query_params["chat"] = serializar_chat(st.session_state.messages)
            st.query_params["memoria"] = st.session_state.memoria.serializar()

else:
    st.info("👈 Por favor, configure a API Key e faça o upload de um arquivo CSV na barra lateral para começar.")