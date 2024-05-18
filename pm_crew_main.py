import sys
import time
import re
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_community.chat_models import ChatOpenAI
from decouple import AutoConfig
from langchain_community.tools import DuckDuckGoSearchRun

config = AutoConfig(".")

OPENAI_API_KEY = config("OPENAI_API_KEY")
OPENAI_MODEL = config("OPENAI_MODEL")

llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    temperature=0,
)

duckduckgo_search = DuckDuckGoSearchRun()


def create_crewai_setup(project_description: str):
    # Define Agents
    project_manager = Agent(
        role="Gerente de Projeto",
        goal=f"""Liderar a equipe do projeto. Garantir que o projeto seja concluído dentro do prazo, orçamento e escopo definidos. Garantir a entrega bem-sucedida do projeto {project_description}, atendendo aos requisitos e expectativas dos stakeholders.""",
        backstory="""Profissional com ampla experiência em gerenciamento de projetos complexos, conhecimento profundo do PMBOK7 e metodologias ágeis.""",
        verbose=True,
        allow_delegation=True,
        tools=[duckduckgo_search],
        llm=llm,
    )

    methodology_consultant = Agent(
        role="Consultor de Metodologia",
        goal="""Definir e adaptar metodologias de gerenciamento de projetos, incluindo práticas do PMBOK7 e ágeis. Criar um framework híbrido que combine as melhores práticas tradicionais e ágeis para o projeto.""",
        backstory="""Especialista em metodologias de gerenciamento de projetos com experiência na implementação de frameworks ágeis e tradicionais.""",
        verbose=True,
        # tools=[duckduckgo_search],
        llm=llm,
    )

    business_analyst = Agent(
        role="Analista de Negócios",
        goal="""Coletar e documentar os requisitos do projeto, garantindo que as necessidades dos stakeholders sejam atendidas. Identificar e alinhar os requisitos do projeto com os objetivos estratégicos da organização.""",
        backstory="""Profissional com forte experiência em análise de negócios e documentação de requisitos, preferencialmente com certificação em análise de negócios.""",
        verbose=True,
        # tools=[duckduckgo_search],
        llm=llm,
    )

    change_management_specialist = Agent(
        role="Especialista em Gestão de Mudança",
        goal="""Gerenciar a transição para as novas práticas e entregas do projeto, garantindo a adesão e adaptação da equipe às mudanças.  Facilitar a implementação e aceitação das mudanças resultantes do projeto, minimizando resistência e promovendo o engajamento.""",
        backstory="""Experiência em gestão de mudanças organizacionais, com histórico de conduzir mudanças bem-sucedidas em ambientes corporativos.""",
        verbose=True,
        # tools=[duckduckgo_search],
        llm=llm,
    )

    quality_analyst = Agent(
        role="Analista de Qualidade",
        goal="""Garantir que os processos e entregas do projeto atendam aos padrões de qualidade estabelecidos. Implementar e monitorar processos de qualidade contínua durante o projeto.""",
        backstory="""Profissional com experiência em controle de qualidade e garantia de qualidade, familiarizado com normas ISO e outras certificações relevantes.""",
        verbose=True,
        # tools=[duckduckgo_search],
        llm=llm,
    )

    resource_manager = Agent(
        role="Gerente de Recurso",
        goal="""Gerenciar a alocação e utilização de recursos humanos e materiais no projeto. Garantir que os recursos estejam disponíveis e sejam utilizados eficientemente durante o projeto.""",
        backstory="""Experiência em gestão de recursos, com habilidades em planejamento e alocação de equipe.""",
        verbose=True,
        # tools=[duckduckgo_search],
        llm=llm,
    )

    tools_and_systems_administrator = Agent(
        role="Administrador de Ferramentas e Sistemas",
        goal="""Configurar e manter as ferramentas de gerenciamento de projetos que serão utilizadas no projeto. Garantir que as ferramentas tecnológicas suportem eficientemente os processos do projeto.""",
        backstory="""Experiência técnica em administração de sistemas e ferramentas de gerenciamento de projetos.""",
        verbose=True,
        # tools=[duckduckgo_search],
        llm=llm,
    )

    communication_specialist = Agent(
        role="Especialista em Comunicação",
        goal="""Desenvolver e implementar estratégias de comunicação para o projeto. Garantir que todas as partes interessadas estejam informadas e alinhadas durante o projeto.""",
        backstory="""Experiência em comunicação corporativa e gestão de stakeholders.""",
        verbose=True,
        # tools=[duckduckgo_search],
        llm=llm,
    )

    financial_analyst = Agent(
        role="Analista Financeiro",
        goal="Monitorar o orçamento do projeto e garantir o controle financeiro. Assegurar que o projeto seja concluído dentro do orçamento previsto e que todos os recursos financeiros sejam utilizados de forma eficiente.",
        backstory="""Experiência em gestão financeira de projetos, com formação em contabilidade ou finanças.""",
        verbose=True,
        # tools=[duckduckgo_search],
        llm=llm,
    )
    
    project_documents = [
        "Termo de Abertura do Projeto",
        "Registro das Partes Interessadas",
        "Plano de Gerenciamento do Projeto",
        "Plano de Escopo",
        "Plano de Cronograma",
        "Plano de Qualidade",
        "Plano de Recursos",
        "Plano de Comunicação",
        "Plano de Riscos",
        "Plano de Aquisições",
        "Plano de Engajamento das Partes Interessadas",
    ]
    
    documents = ', '.join([document.lower() for document in project_documents])

    tasks = []
    tasks.append(
        Task(
            description=f"Elaborar os documentos ({documents}) do projeto com descrição: {project_description}",
            expected_output=f"Documentos ({documents}) de acordo com PMNOK7",
            agent=project_manager,
        )
    )

    # Create and Run the Crew
    project_crew = Crew(
        agents=[
            project_manager,
            methodology_consultant,
            business_analyst,
            #change_management_specialist,
            quality_analyst,
            resource_manager,
            tools_and_systems_administrator,
            communication_specialist,
            financial_analyst,
        ],
        tasks=tasks,
        verbose=2,
        process=Process.sequential,
    )

    crew_result = project_crew.kickoff()
    return crew_result


# display the console processing on streamlit UI
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r"\x1B\[[0-9;]*[mK]", "", data)

        # Check if the data contains 'task' information
        task_match_object = re.search(
            r"\"task\"\s*:\s*\"(.*?)\"", cleaned_data, re.IGNORECASE
        )
        task_match_input = re.search(
            r"task\s*:\s*([^\n]*)", cleaned_data, re.IGNORECASE
        )
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.toast(":robot_face: " + task_value)

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown("".join(self.buffer), unsafe_allow_html=True)
            self.buffer = []


# Streamlit interface
def run_crewai_app():
    st.title("Equipe de IA para abertura de projetos")

    project_description = st.text_input(
        "Insira um a descrição de um projeto para gerar os documentos deste projeto."
    )

    if st.button("Analisar"):
        # Placeholder for stopwatch
        stopwatch_placeholder = st.empty()

        # Start the stopwatch
        start_time = time.time()
        with st.expander("Processando!"):
            sys.stdout = StreamToExpander(st)
            with st.spinner("Gerando resultados"):
                crew_result = create_crewai_setup(project_description)

        # Stop the stopwatch
        end_time = time.time()
        total_time = end_time - start_time
        stopwatch_placeholder.text(f"Tempo total decorrido: {total_time:.2f} segundos")

        st.header("Resultados:")
        st.markdown(crew_result)


if __name__ == "__main__":
    run_crewai_app()
