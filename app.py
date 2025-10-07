import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import json # ### MUDANÇA ###: Importar para serialização
import base64 # ### MUDANÇA ###: Importar para codificação

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

# --- ### MUDANÇA ###: Funções para codificar e decodificar o histórico do chat ---
def serializar_chat(mensagens):
    """Converte a lista de mensagens em uma string base64 segura para URL."""
    if not mensagens:
        return ""
    # Nota: Figuras não podem ser serializadas, então as removemos para a URL.
    mensagens_para_serializar = []
    for m in mensagens:
        msg_copy = {"role": m["role"], "content": m["content"]}
        if "figure" in m and m["figure"] is not None:
            # Salva a figura em um buffer e codifica para base64
            import io
            import matplotlib.pyplot as plt
            buf = io.BytesIO()
            m["figure"].savefig(buf, format="png")
            msg_copy["figure_base64"] = base64.b64encode(buf.getvalue()).decode()
            plt.close(m["figure"]) # Libera a memória da figura
        mensagens_para_serializar.append(msg_copy)
    return base64.b64encode(json.dumps(mensagens_para_serializar).encode()).decode()

def deserializar_chat(string_codificada):
    """Converte a string da URL de volta para uma lista de mensagens."""
    if not string_codificada:
        return []
    try:
        mensagens_decodificadas = json.loads(base64.b64decode(string_codificada.encode()).decode())
        for m in mensagens_decodificadas:
            m["figure"] = None # Garante que o objeto Figure não seja armazenado
            # O campo figure_base64 será usado para exibição, se presente
        return mensagens_decodificadas
    except:
        # Se a URL estiver corrompida, retorna um chat vazio
        return []

# --- Configuração da Página Streamlit ---
st.set_page_config(
    page_title="Agente de Análise de CSV",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agente Autônomo para Análise de Dados em CSV")
st.write("Esta aplicação utiliza um agente de IA para responder perguntas sobre arquivos CSV.")

# ... (O resto das suas funções e configurações permanece o mesmo) ...
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

# --- ### MUDANÇA ###: Inicialização do chat a partir da URL ---
# Pega o estado do chat da URL ao carregar a página
query_params = st.query_params.to_dict()
if "messages" not in st.session_state:
    # Usamos st.query_params para ler, como recomendado pela nova API do Streamlit.
    try:
        # O .get() em st.query_params retorna um valor simples, não uma lista.
        chat_param = st.query_params.get("chat", "")
        st.session_state.messages = deserializar_chat(chat_param)
    except:
        # Em caso de erro na URL, apenas começamos com uma lista vazia.
        st.session_state.messages = []

# ... (Lógica da API Key e Upload do Arquivo permanecem os mesmos) ...
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

# --- ### CORREÇÃO DEFINITIVA NA BARRA LATERAL ### ---
with st.sidebar:
    st.header("Upload do Arquivo")
    arquivo_csv = st.file_uploader("Selecione um arquivo CSV", type=["csv"])

    # Verificamos se um arquivo foi carregado E se ele é diferente do que já está na sessão.
    # Isso garante que este bloco só execute UMA VEZ por upload.
    if arquivo_csv is not None and st.session_state.get('uploaded_file_name') != arquivo_csv.name:
        st.session_state.df = carregar_e_processar_csv(arquivo_csv)
        
        if st.session_state.df is not None:
            st.success(f"Arquivo '{arquivo_csv.name}' carregado!")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
            
            # Armazena o nome do novo arquivo para evitar re-execução
            st.session_state.uploaded_file_name = arquivo_csv.name
            
            # Agora, aqui é o lugar certo para resetar o agente e o chat
            st.session_state.agent = None
            st.session_state.messages = []
            st.query_params.clear()
            
            # Força um novo carregamento da página para garantir que tudo está limpo
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
            if "figure_base64" in message and message["figure_base64"] is not None:
                st.image(base64.b64decode(message["figure_base64"]), use_container_width=True)

            elif "figure" in message and message["figure"] is not None:
                st.pyplot(message["figure"])

    if prompt := st.chat_input("Qual a distribuição da variável 'idade'?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # ### MUDANÇA ###: Atualiza a URL após a pergunta do usuário
        st.query_params["chat"] = serializar_chat(st.session_state.messages)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("O agente está pensando..."):
                try:
                    plt.close('all')
                    chat_history = []
                    for m in st.session_state.messages:
                        msg_copy = {"role": m["role"], "content": m["content"]}
                        # Remove figure e figure_base64 para o histórico do agente
                        if "figure" in msg_copy: del msg_copy["figure"]
                        if "figure_base64" in msg_copy: del msg_copy["figure_base64"]
                        chat_history.append(msg_copy)

                    response = st.session_state.agent.invoke({"input": prompt, "chat_history": chat_history})
                    output_text = response["output"]
                    fig = plt.gcf()
                    has_plot = any(ax.has_data() for ax in fig.get_axes()) if fig else False

                    if has_plot:
                        # Salva a figura em um buffer e codifica para base64
                        import io
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png")
                        fig_base64 = base64.b64encode(buf.getvalue()).decode()
                        st.pyplot(fig) # Exibe a figura
                        st.session_state.messages.append({"role": "assistant", "content": output_text, "figure": None, "figure_base64": fig_base64})
                        plt.close(fig) # Libera a memória da figura
                    else:
                        plt.close(fig)
                        st.markdown(output_text)
                        st.session_state.messages.append({"role": "assistant", "content": output_text, "figure": None})

                except Exception as e:
                    error_message = f"Ocorreu um erro: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message, "figure": None})
        
        # ### MUDANÇA ###: Atualiza a URL novamente após a resposta do assistente
        st.query_params["chat"] = serializar_chat(st.session_state.messages)

else:
    st.info("Por favor, configure a API Key e faça o upload de um arquivo CSV na barra lateral para começar.")

