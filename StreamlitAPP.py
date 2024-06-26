import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
import streamlit as st
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.MCQGenerator import generate_evaluate_chain
from src.mcqgenerator.logger import logging
from langchain.globals import set_llm_cache, get_llm_cache

# Loading JSON file
with open('C:/Users/karan/OneDrive/Documents/Code/Gen_AI_Projects/mcqgen/Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

st.title("MCQs Creator Application with Langchain")

with st.form("user_inputs"):
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF or txt file")
    # Input fields
    mcq_count = st.number_input("No. of MCQs", min_value=3, max_value=50)
    # Subject
    subject = st.text_input("Insert Subject", max_chars=20)
    # Quiz Tone
    tone = st.text_input("Complexity level of Questions", max_chars=20, placeholder="Simple")
    # Add Button
    button = st.form_submit_button("Create MCQs")

    # Check if the button is clicked and all fields have input
    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading..."):
            try:
                text = read_file(uploaded_file)
                # Count tokens and cost of API call
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain({
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "response_json": json.dumps(RESPONSE_JSON)
                    })
                #st.write(response)

            except Exception as e:
                traceback.print_exception(type(e), e, e.__traceback__)
                st.error("Error")

            else:
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost: {cb.total_cost}")
                if isinstance(response, dict):
                    # Extract the quiz data from the response
                    quiz = response.get("quiz", None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data:
                            df = pd.DataFrame(table_data, columns=["MCQ", "Choices", "Correct"])
                            df.index = df.index + 1
                            st.table(df)
                            # Display the review in a text box as well
                            st.text_area(label="Review", value=response["review"])
                            # Convert DataFrame to JSON
                            json_data = df.to_json(orient="records", indent=4)
                            # Save JSON to a local file
                            json_file_path = 'C:/Users/karan/OneDrive/Documents/Code/Gen_AI_Projects/mcq-quiz-main/public/mcq_data.json'
                            with open(json_file_path, 'w') as json_file:
                                json_file.write(json_data)
                            st.success(f"Data saved to {json_file_path}")
                        else:
                            st.error("Error in the table data")
                else:
                    st.write(response)
