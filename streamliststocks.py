"""
Executando o notebook em um script Python
- pip install notebook
- pip install jupyter
- pip install streamlit
- usando no terminal jupyter nbconvert --to script crewai-stocks.ipynb

Para executar a aplicação: streamlit run crewai-stocks.py
"""


#import das libs

import json
import os
from datetime import datetime

#importando o yahoo finance
import yfinance as yf

#importando a CrewIA
from crewai import Agent, Task, Crew, Process

#importando tools
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st


"""
Criando Yahoo Finance Tool
- puxando ações da Apple iniciando em 08/08/2023 e terminando em 2024
- transformando em uma função padrão para utilizar como uma ferramente de agente de IA
"""
#função que agrupa os preços de uma ação (recebe um ticket como parametro para buscar no yahoo finance)
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock

#transformando em uma ferramenta
yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stocks prices for {ticket} from the last year about a specific strocks company from Yahoo Finance API",
    func = lambda ticket: fetch_stock_price(ticket)
)

#importando OpenIA LLM - GPT 3.5 turbão
#chave da api - COLOCAR SUA KEY CRIADA NA OPEN
os.environ['OPENAI_API_KEY'] = ""
llm = ChatOpenAI(model="gpt-3.5-turbo")

#Criando o primeiro agente
stockPriceAnalyst = Agent(
    role = "Senior stock price Analyst",
    #obj
    goal = "Find the {ticket} stock price ande analyses trends",
    backstory = """ You're a highty experience in analyzing the price of an specific stock
                and make predictions about its future price.""",
    #ver todo passo a passo
    verbose = True,
    llm = llm,
    #máximo de interações
    max_iter = 5,
    memory = True,
    tools = (yahoo_finance_tool),
    allow_delegation = False
)

#Criando uma tarefa para o primeiro agente executar
getStockPrice = Task(
    description = "Analyze the strock {ticket} price history and create a trnd analyses of up, down or sideways",
    expected_output = """ Specify the current trend stock price - up, down or sideways.
                        eg. stock = 'AAPL, price UP' """,
    agent = stockPriceAnalyst
)

#Importando a Tool de search
search_tool = DuckDuckGoSearchResults(
    backend = 'news', 
    num_results = 10
)

#Criando segundo agente - Analise de Notícias
newsAnalyst = Agent(
    role = "Stock news Analyst",
    goal = """ Create a short sumary of the market news related to the stock {ticket} company. 
                Specify the current trend - up, down or sideway with the news context. For each
                request stock asset, specify a number between 0 and 100, where 0 is extreme fear
                and 100 is extreme greed.""",
    backstory = """ You're highly experienced in analyzing the market trends and news and have tracked assest for more then 10 years.
                    You're also master level analyts in the tradicional markets and have deep understanding of human psychology.
                    You understand news, theirs tittles and information, but you look at those with a health dose of skepticism. 
                    You consider also the source of the news articles. """,
    verbose = True,
    llm = llm,
    max_iter = 10,
    memory = True,
    tools = [search_tool],
    allow_delegation = False
) 

#Criando a tarefa do segundo agente
get_news = Task(
    description = """ Take the stock and always include BTC to it (if not request).
                        Use the search tool to search each one individually. 
                        The current date is {datetime.now()}.
                        Compose the results into a helpfull report """,
    expected_output = """ A summary of the overall market and one sentence summary for each request asset. 
                            Include a fear/greed score for each asset based on the news. Use format:
                            <STOCK ASSET>
                            <SUMMARY BASED ON NEWS>
                            <TREND PREDICTION>
                            <FEAR/GREED SCORE> """,
    agent = newsAnalyst
)

# Criando o terceiro agente - Escrever/Fazer a Análise de fato
stockAnalystWrite = Agent(
    role = "Senior stock Analyst Writer",
    goal = """ "Analyze the trends price and news and write an insighfull compelling and informative 
                3 paragraph long newsletter based on the stock report and price trend. """,
    backstory = """ You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories
                    and narratives that resonate with wider audiences. 

                    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. 
                    You're able to hold multiple opinions when analyzing anything. """,
    verbose = True,
    llm = llm,
    max_iter = 5,
    memory = True,
    allow_delegation = False
)

#Criando a tarefa do terceiro agente
writeAnalyses = Task(
    description = """ Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} company
                        that is brief and highlights the most important points.
                        Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
                        Include the previous analyses of stock trend and news summary. """,
    expected_output = """ "An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain:
                            - 3 bullets executive summary 
                            - Introduction 
                            - set the overall picture and spike up the interest
                            - main part provides the meat of the analysis including the news summary and fead/greed scores
                            - summary 
                            - key facts and concrete future trend prediction - up, down or sideways. """,
    agent = stockAnalystWrite,
    #pra essa tarefa, precisamos de dois contextos (primeira e segunda tarefas juntas)
    context = [getStockPrice, get_news]
)

#Criando o grupo de agentes de IA - usando método crew
crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWrite],
    tasks = [getStockPrice, get_news, writeAnalyses],
    verbose = 2,
    
    #Process - se vai ser sequencial (uma atras da outra) ou hierarquico (organizaar as tarefas de forma hierarquica - estrutura de comando)
    process = Process.hierarchical,
    
    full_output = True,
    share_crew = False,
    manager_llm = llm,
    max_iter = 15
)

#Executando o Crew
results= crew.kickoff(inputs={'ticket': 'AAPL'})

#Printando as finals outputs
results['final_output']

#Printando Resultados - streamlit
#Construindo uma app web construindo um sidebar
with st.sidebar:
    st.header('Enter the Stock to Research')

    with st.form(key='research_form'):
        #input (usuário)
        topic = st.text_input("Select the ticket")
        #botão de submit - "buscar"
        submit_button = st.form_submit_button(label = "Run Research")
#Se usuario usar o botão ou "Enter"
if submit_button:
    if not topic:
        st.error("Please fill the ticket field")
    #Se existir o topic - retorna os valores
    else:
        results= crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results of research:")
        st.write(results['final_output'])