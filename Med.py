from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 🔍 Use a general-purpose LLM like Falcon, BLOOM, or DistilGPT2
diagnosis_pipeline = pipeline("text-generation", model="distilgpt2", max_length=200)

# ⛓️ Wrap it in LangChain
llm = HuggingFacePipeline(pipeline=diagnosis_pipeline)

# 🧠 Prompt Template
prompt = PromptTemplate(
    input_variables=["symptoms"],
    template="""
You are an experienced medical assistant. Based on the following symptoms provided by a patient, suggest possible medical conditions they might be experiencing and recommend the next steps (tests, lifestyle changes, or whether to consult a specialist).

Symptoms: {symptoms}

Diagnosis and Advice:
"""
)

# 🩺 Create the chain
diagnosis_chain = LLMChain(llm=llm, prompt=prompt)

# 🔡 Input symptoms
user_symptoms = "fever, sore throat, runny nose, headache, muscle aches"

# 🔬 Run diagnosis
result = diagnosis_chain.run(symptoms=user_symptoms)

# 📤 Output
print("🧠 AI Diagnosis Assistant:")
print(result)
