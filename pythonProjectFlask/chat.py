import json
import os
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from operator import itemgetter
import pymysql

os.environ["OPENAI_API_KEY"] = 'Open AI API Key Here'

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

connection = pymysql.connect(host="34.70.162.68", user="anisharao", passwd="airbnb", db="airbnb-298")


def get_inventory(airbnb_id=None):
    try:
        # Create a cursor object
        cursor = connection.cursor()

        if airbnb_id:
            # Execute a query to get the table names
            cursor.execute(f"select * from airbnb_detection where airbnb_id = {airbnb_id}")
        else:
            cursor.execute("select * from airbnb_detection")

        # Fetch all the tables
        records = cursor.fetchall()
        json_records = []
        for record in records:
            json_records.append({'airbnb_id': record[0],
                                 'amenity_name': record[1],
                                 'amenity_count': record[2],
                                 'CATEGORY': record[3]})
        json_output = json.dumps(json_records, indent=2)
        return json_output

    finally:
        # Close the cursor
        cursor.close()


def answer_question(conv_history, user_info=None):
    if not user_info:
        user_info = get_inventory()

    print(user_info)

    QA_CHAIN_PROMPT = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)

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
    print(get_inventory(1))
    samp_user_info = str({'property1': {'amenities': {'oven': 1, 'chairs': 2}},
                          'property2': {'amenities': {'oven': 2, 'chairs': 4}}})
    resp = answer_question(
        "How can I add inventory into application", user_info='')
    print(resp)
