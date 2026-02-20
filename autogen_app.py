import os
import pandas as pd
import streamlit as st
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from langchain_openai import ChatOpenAI

# -------------------------------
# Setup OpenAI API Key
# -------------------------------
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=os.environ["OPENAI_API_KEY"]
)

system_prompt = """
You are an agentic AI system designed to support maternal health by enabling early detection of pregnancy complications,
personalized care pathways, and remote monitoring. Follow ethical guardrails: provide decision support, not final medical decisions.
"""
# === Pipeline Class ===
class MaternalHealthPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # Step 1: Preprocessing
    def preprocess_data(self):
        df = self.df.dropna()
        df.columns = [c.strip().lower() for c in df.columns]
        self.df = df
        return df

    # Step 2: Risk Stratification
    def stratify_risk(self):
        def stratify(row):
            if row['systolicbp'] >= 140 or row['diastolicbp'] >= 90 or row['bs'] >= 11:
                return "high risk"
            elif row['systolicbp'] >= 120 or row['bs'] >= 8:
                return "mid risk"
            else:
                return "low risk"
        self.df['predicted_risk'] = self.df.apply(stratify, axis=1)
        return self.df

    # Step 3: Care Planning
    def generate_care_plans(self):
        def care_plan(row):
            plan = {
                "risk_category": row['predicted_risk'],
                "monitoring_frequency": "weekly" if row['predicted_risk']=="high risk" else "monthly",
                "interventions": [],
                "alerts": []
            }
            if row['predicted_risk'] == "high risk":
                plan["interventions"].append("Schedule specialist consultation")
                plan["alerts"].append("Notify clinician immediately")
            elif row['predicted_risk'] == "mid risk":
                plan["interventions"].append("Recommend lifestyle adjustments (diet, exercise)")
            else:
                plan["interventions"].append("Routine monitoring")
            return plan
        self.df['care_plan'] = self.df.apply(care_plan, axis=1)
        return self.df

    # Step 4: Alerts
    def generate_alerts(self):
        alerts = []
        for _, row in self.df.iterrows():
            if row['predicted_risk'] == "high risk":
                alerts.append(f"ALERT: Patient age {row['age']} requires immediate clinician attention.")
        return alerts
    # === Step 5: Remote Monitoring (stub for IoT integration) ===
    def remote_monitoring(self):
        # Placeholder for wearable/device integration
        return "Remote monitoring pipeline ready for IoT data ingestion."

    # Run Full Pipeline
    def run_pipeline(self):
        self.preprocess_data()
        self.stratify_risk()
        self.generate_care_plans()
        alerts = self.generate_alerts()

        results_json = self.df[['age','systolicbp','diastolicbp','bs','predicted_risk','care_plan']].to_json(orient="records")
        tabular_summary = self.df[['age','systolicbp','diastolicbp','bs','predicted_risk']]

        return {
            "json_output": results_json,
            "tabular_summary": tabular_summary,
            "alerts": alerts,
            "remote_monitoring_status": remote_status,
            "system_prompt": system_prompt
        }

# === Streamlit UI ===
st.title("🤰 Maternal Health Risk Detection - Agentic AI Pipeline")

uploaded_file = st.file_uploader("Upload Maternal Health CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head(10))

    pipeline = MaternalHealthPipeline(df)
    results = pipeline.run_pipeline()

    st.subheader("📈 Risk Stratification Summary")
    st.dataframe(results["tabular_summary"].head(20))

    st.subheader("📑 JSON Output")
    st.json(results["json_output"])

    st.subheader("🚨 Alerts")
    for alert in results["alerts"]:
        st.warning(alert)
else:
    st.info("Please upload the maternal health dataset CSV to begin analysis.")





