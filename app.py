import joblib
import gradio as gr
import pandas as pd
import numpy as np
import re
import os

# Load saved model files
model = joblib.load("salary_model.pkl")
encoder = joblib.load("salary_encoder.pkl")
model_columns = joblib.load("model_columns.pkl")


# Calculate job title seniority
def calculate_title_rank(title):
    t = title.lower()

    if re.search(r"principal|distinguished|director|head|vp|vice president|chief|fellow", t):
        return 4
    if re.search(r"\blead\b|\bstaff\b|manager", t):
        return 3
    if re.search(r"senior|\bsr\b", t):
        return 2
    if re.search(r"intern|trainee|junior|\bjr\b|entry|graduate|associate", t):
        return 0

    return 1

# Connect user inputs to the trained model
def predict_salary(
    job_title, job_category, country, job_level,
    experience, education, remote, num_skills,
    machine_learning, python, sql, excel,
    aws, power_bi, spark, tableau
):

    if not job_title or not job_category or not country or not job_level or not education:
        raise gr.Error("Please complete all job information.")

    if experience is None or experience < 0:
        raise gr.Error("Please enter valid years of experience.")

    if num_skills is None or num_skills < 0:
        raise gr.Error("Please enter a valid number of skills.")

    # Categorical inputs
    categorical_input = pd.DataFrame([{
        "country": country,
        "job_level": job_level,
        "search_keyword": job_category,
        "degree_required": education,
        "experience_group": "6+" if experience > 5 else "0-5"
    }])

    # Encode categorical features
    encoded = encoder.transform(categorical_input)

    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out()
    )

    # Numerical and binary features
    numeric_df = pd.DataFrame([{
        "experience_required": experience,
        "num_skills": num_skills,
        "is_remote": int(remote),
        "title_rank": calculate_title_rank(job_title),
        "skill_machine_learning": int(machine_learning),
        "skill_python": int(python),
        "skill_sql": int(sql),
        "skill_excel": int(excel),
        "skill_aws": int(aws),
        "skill_power_bi": int(power_bi),
        "skill_spark": int(spark),
        "skill_tableau": int(tableau)
    }])

    # Combine and arrange the exact 26 model features
    input_data = pd.concat([numeric_df, encoded_df], axis=1)
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Predict salary
    prediction = model.predict(input_data)[0]

    return f"Predicted Annual Salary: ${prediction:,.2f} USD"


# Build Gradio interface
with gr.Blocks(title="TalentPulse Analytics") as demo:

    gr.Markdown("# TalentPulse Analytics")
    gr.Markdown("### Global Salary Prediction")

    with gr.Row():
        job_title = gr.Textbox(label="Full Job Title",
                               placeholder="e.g. Senior Data Scientist")
        job_category = gr.Dropdown(
            ["business analyst", "data analyst", "data engineer",
             "data scientist", "machine learning engineer"],
            label="Job Category"
        )

    with gr.Row():
        country = gr.Dropdown(
            ["Australia", "Canada", "India", "UK", "USA"],
            label="Country"
        )
        job_level = gr.Dropdown(
            ["Junior", "Mid", "Senior", "Lead"],
            label="Job Level"
        )

    with gr.Row():
        experience = gr.Number(label="Years of Experience", value=3)
        education = gr.Dropdown(
            ["Bachelor", "Master", "Not specified"],
            label="Education"
        )

         remote = gr.Radio([0, 1], value=0, label="Remote Work (0 = No, 1 = Yes)")
    num_skills = gr.Number(label="Total Number of Skills", value=3)

    gr.Markdown("### Select Your Skills")

    with gr.Row():
        machine_learning = gr.Checkbox(label="Machine Learning")
        python = gr.Checkbox(label="Python")
        sql = gr.Checkbox(label="SQL")
        excel = gr.Checkbox(label="Excel")

    with gr.Row():
        aws = gr.Checkbox(label="AWS")
        power_bi = gr.Checkbox(label="Power BI")
        spark = gr.Checkbox(label="Spark")
        tableau = gr.Checkbox(label="Tableau")

    button = gr.Button("Predict Salary", variant="primary")
    result = gr.Textbox(label="Salary Prediction")

    button.click(
        predict_salary,
        inputs=[
            job_title, job_category, country, job_level,
            experience, education, remote, num_skills,
            machine_learning, python, sql, excel,
            aws, power_bi, spark, tableau
        ],
        outputs=result
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 10000))
)
#demo.launch()