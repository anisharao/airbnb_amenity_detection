import os
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from operator import itemgetter


os.environ["OPENAI_API_KEY"] = 'OPENAI KEY HERE'

with open('Airbnb_questions.txt') as query_file:
    questions = query_file.read()

template = """Act like a human assistant for an Airbnb like application that helps users with their queries.
You are provided with two information sources - 1. user information in our application. 2. A Knowledge document
Important:
- Always respond to the latest message
- If the user query is about the property or amenities, Use the provided factual data (user information) to answer him.
- If the user query cannot be answered with the the provided user information, then answer it using the knowledge document.
- If you cannot find any answer from the provided User Information and Knowledge document, respond 'Sorry, I do not have information about that, try something else'

User Information:
{user_info}

Knowledge document:
{questions}

Customer conversation: 
{conv_history}

Return only the response for the user.
"""


def answer_question(conv_history, user_info=None):
    QA_CHAIN_PROMPT = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    chain = ({"questions": itemgetter('questions'),
              "conv_history": itemgetter('conv_history'),
              "user_info": itemgetter('user_info')}
             | QA_CHAIN_PROMPT
             | llm
             | StrOutputParser())

    return chain.invoke({'questions': questions,
                         'conv_history': conv_history,
                         'user_info': user_info})


if __name__ == '__main__':
    samp_user_info = str({'property1': {'amenities': {'oven': 1, 'chairs': 2}},
                          'property2': {'amenities': {'oven': 2, 'chairs': 4}}})
    resp = answer_question(
        "How can I add inventory into application", user_info='')
    print(resp)
