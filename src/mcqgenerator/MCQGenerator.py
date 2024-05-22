import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

load_dotenv()

key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key=key, model_name="gpt-3.5-turbo", temperature=0.7)

template = """
Text: {text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQS
### RESPONSE_JSON
{response_json}
"""
quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template)

quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

template2 = """
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at par with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""
quiz_evaluation_prompt = PromptTemplate(input_variables=["subject", "quiz"], template=template2)

review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

generate_evaluate_chain = SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                          output_variables=["quiz", "review"], verbose=True)

def format_quiz_response(quiz_response):
    formatted_quiz = []
    for question in quiz_response:
        formatted_question = {
            "question": question["question"],
            "options": [opt for opt in question["options"].values()],
            "correctAnswer": question["correct"]
        }
        formatted_quiz.append(formatted_question)
    return formatted_quiz

def generate_mcqs(text, number, subject, tone, response_json):
    result = generate_evaluate_chain({
        "text": text,
        "number": number,
        "subject": subject,
        "tone": tone,
        "response_json": response_json
    })
    quiz_response = result["quiz"]
    formatted_quiz = format_quiz_response(quiz_response)
    return formatted_quiz

def save_quiz_to_json(formatted_quiz, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(formatted_quiz, f, ensure_ascii=False, indent=4)

# Example usage:
if __name__ == "__main__":
    text = "Your text here"
    number = 10
    subject = "History"
    tone = "simple"
    response_json = "Your response JSON template here"

    formatted_quiz = generate_mcqs(text, number, subject, tone, response_json)
    save_quiz_to_json(formatted_quiz, 'quiz_data.json')
