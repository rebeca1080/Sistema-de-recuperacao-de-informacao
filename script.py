
import torch

from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA

 #Verifica dispositivo disponível
device = 0 if torch.cuda.is_available() else -1

print("Dispositivo:", "GPU" if device == 0 else "CPU")

# Criar a base de conhecimento
texto = """
A inteligência artificial é uma área da computação
que desenvolve sistemas capazes de realizar tarefas
que normalmente exigem inteligência humana.
No mercado financeiro, a inteligência artificial
pode auxiliar investidores na análise de dados,
identificação de tendências e avaliação de riscos.
As redes neurais artificiais são modelos
computacionais inspirados no funcionamento
do cérebro humano.
O aprendizado de máquina permite que sistemas
identifiquem padrões nos dados e melhorem
seu desempenho por meio da experiência.
Os embeddings são representações numéricas
de textos que permitem comparar o significado
de diferentes informações.
"""

# Dividi o texto em chunks
divisor = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

documentos = divisor.create_documents([texto])

print(f"\nQuantidade de chunks: {len(documentos)}")

# Gerar embeddings localmente
print("\nCarregando modelo de embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Criar banco de dados vetorial FAISS
print("\nCriando banco de dados FAISS...")

banco_vetorial = FAISS.from_documents(
    documentos,
    embeddings
)

#Salvar o banco vetorial
banco_vetorial.save_local("banco_faiss")

print("Banco vetorial criado e salvo com sucesso!")

# Configura o modelo de linguagem local
print("\nCarregando modelo de linguagem...")

gerador = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=device,
    max_new_tokens=150
)

llm = HuggingFacePipeline(
    pipeline=gerador
)

# Configurar o RetrievalQA
retriever = banco_vetorial.as_retriever(
    search_kwargs={"k": 3}
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

print("\nSistema de recuperação iniciado!")

# Receber perguntas do usuário
while True:

    pergunta = input(
        "\nDigite sua pergunta (ou 'sair' para encerrar): "
    )

    if pergunta.lower() == "sair":
        print("Encerrando o sistema...")
        break

    resultado = qa.invoke({"query": pergunta})

    print("\n--- RESPOSTA DO SISTEMA ---")
    print(resultado["result"])

    print("\n--- TRECHOS RECUPERADOS ---")

    for documento in resultado["source_documents"]:
        print("-", documento.page_content)
