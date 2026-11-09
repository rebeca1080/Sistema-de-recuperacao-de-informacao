
# Sistema de Recuperação de Informações

# Sobre o projeto

Este projeto tem o objetivo desenvolver um sistema de recuperação de informações utilizando Inteligência Artificial, embeddings e banco de dados vetorial.

O sistema utiliza LangChain e FAISS para armazenar e recuperar informações relevantes, permitindo responder perguntas com base em um conteúdo previamente fornecido.

# Tecnologias utilizadas

- Python
- LangChain
- FAISS
- Hugging Face
- Sentence Transformers
- PyTorch
- FLAN-T5

## Instalação

Instale as dependências com:

```bash
pip install -r requirements.txt
```

## Como executar

Execute o comando:

```bash
python script.py
```

Na primeira execução, os modelos serão baixados
automaticamente.

## Exemplos de perguntas

- o que é inteligência artificial?
- O que são redes neurais?
- Como a IA auxilia o mercado financeiro?

## Funcionamento

1. O texto é dividido em pequenos trechos (chunks).
2. São gerados embeddings para cada trecho.
3. Os embeddings são armazenados no FAISS.
4. O RetrievalQA realiza a recuperação das informações.
5. O modelo de linguagem gera uma resposta com base
   nos trechos recuperados.

## Autora

Atividade acadêmica desenvolvida para a disciplina de Tópicos I
