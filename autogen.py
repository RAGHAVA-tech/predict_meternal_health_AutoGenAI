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
class MaternalHealthPipeline:
    def __init__(self, data_file: str):
        self.data_file = data_file
        # Use the data_file parameter and assume it's a CSV based on example usage
        # If it's an Excel file, change pd.read_csv to pd.read_excel accordingly
        self.df = pd.read_csv(io.BytesIO(uploaded['Maternal Health Risk Data Set (1).csv']))


        # Define agents
        self.DataPreprocessor = AssistantAgent(
            name="DataPreprocessor",
            system_message="Clean and normalize patient data."
        )
        self.RiskStratifier = AssistantAgent(
            name="RiskStratifier",
            system_message="Detect at-risk pregnancies using statistical and ML models."
        )
        self.CarePlanner = AssistantAgent(
            name="CarePlanner",
            system_message="Generate personalized care pathways based on risk profile."
        )
        self.AlertManager = AssistantAgent(
            name="AlertManager",
            system_message="Guide timely interventions and trigger clinician alerts."
        )
        self.RemoteMonitor = AssistantAgent(
            name="RemoteMonitor",
            system_message="Continuously track patient vitals and symptoms."
        )

        # Group chat setup
        agents = [self.DataPreprocessor, self.RiskStratifier, self.CarePlanner,
                  self.AlertManager, self.RemoteMonitor]
        self.groupchat = GroupChat(agents=agents, messages=[])
        self.manager = GroupChatManager(groupchat=self.groupchat)

        self.user = UserProxyAgent(
            name="Clinician",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
        )

    # === Step 1: Preprocessing ===
    def preprocess_data(self):
        df = self.df.copy()
        df = df.dropna()
        df.columns = [c.strip().lower() for c in df.columns]
        self.df = df
        return df

    # === Step 2: Risk Stratification ===
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

    # === Step 3: Care Planning ===
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

    # === Step 4: Alerts ===
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

    # === Run Full Pipeline ===
    def run_pipeline(self):
        print("Step 1: Preprocessing...")
        self.preprocess_data()

        print("Step 2: Risk Stratification...")
        self.stratify_risk()

        print("Step 3: Care Planning...")
        self.generate_care_plans()

        print("Step 4: Alerts...")
        alerts = self.generate_alerts()

        print("Step 5: Remote Monitoring...")
        remote_status = self.remote_monitoring()

        # Outputs
        results_json = self.df[['age','systolicbp','diastolicbp','bs','predicted_risk','care_plan']].to_json(orient="records")
        tabular_summary = self.df[['age','systolicbp','diastolicbp','bs','predicted_risk']].head(10)

        return {
            "json_output": results_json,
            "tabular_summary": tabular_summary,
            "alerts": alerts,
            "remote_monitoring_status": remote_status,
            "system_prompt": system_prompt

        }
