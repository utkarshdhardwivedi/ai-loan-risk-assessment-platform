import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Loan Risk Assessment", layout="wide")
st.title("🏦 AI Loan Risk Assessment Platform")

tab1, tab2, tab3 = st.tabs(["🆕 New Application", "📋 Applications", "💬 Ask the Assistant"])

with tab1:
    st.subheader("Applicant Financial Profile")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30)
        annual_income = st.number_input("Annual Income ($)", 0.0, 2_000_000.0, 60000.0, step=1000.0)
        employment_years = st.number_input("Years Employed", 0.0, 60.0, 4.0)
        loan_amount = st.number_input("Requested Loan Amount ($)", 0.0, 2_000_000.0, 20000.0, step=500.0)
    with col2:
        credit_history_years = st.number_input("Credit History (years)", 0.0, 50.0, 5.0)
        home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        loan_intent = st.selectbox(
            "Loan Intent",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
        prior_default = st.selectbox("Prior Default on File", ["N", "Y"])

    loan_percent_income = loan_amount / annual_income if annual_income > 0 else 0.0
    st.caption(f"Computed loan-to-income ratio: {loan_percent_income:.1%}")

    st.markdown("**Optional: Upload a payslip / bank statement (PDF)** to cross-check reported income.")
    uploaded_pdf = st.file_uploader("Supporting document", type=["pdf"])

    if st.button("Run Risk Assessment", type="primary"):
        applicant_payload = {
            "age": age,
            "annual_income": annual_income,
            "employment_years": employment_years,
            "loan_amount": loan_amount,
            "credit_history_years": credit_history_years,
            "home_ownership": home_ownership,
            "loan_intent": loan_intent,
            "prior_default": prior_default,
        }

        extracted_income = None
        if uploaded_pdf is not None:
            with st.spinner("Parsing document..."):
                files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                doc_resp = requests.post(f"{API_URL}/assess/document", files=files)
                if doc_resp.ok:
                    doc_result = doc_resp.json()
                    extracted_income = doc_result.get("extracted_income")
                    st.info(
                        f"Document-extracted annual income estimate: "
                        f"{extracted_income if extracted_income else 'not detected'} "
                        f"(method: {doc_result.get('method')})"
                    )

        with st.spinner("Scoring application..."):
            params = {}
            if extracted_income is not None:
                params["extracted_income"] = extracted_income
            resp = requests.post(f"{API_URL}/assess/full", json=applicant_payload, params=params)

        if resp.ok:
            result = resp.json()
            decision_color = {"APPROVE": "green", "REVIEW": "orange", "REJECT": "red"}[result["decision"]]

            st.markdown(f"### Decision: :{decision_color}[{result['decision']}]")
            st.metric("Estimated Default Risk", f"{result['risk_probability']:.1%}", result["risk_band"])
            st.write(result["narrative"])

            st.markdown("**Top contributing factors:**")
            factors_df = pd.DataFrame(result["top_factors"])
            st.dataframe(factors_df, use_container_width=True)

            if result["fraud_flags"]:
                st.warning("⚠️ Fraud/anomaly flags:\n" + "\n".join(f"- {f}" for f in result["fraud_flags"]))

            st.caption(f"Application ID: {result['application_id']} (saved to database)")
        else:
            st.error(f"Assessment failed: {resp.text}")

with tab2:
    st.subheader("All Applications")
    if st.button("Refresh"):
        st.rerun()

    resp = requests.get(f"{API_URL}/applications")
    if resp.ok:
        records = resp.json()
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No applications submitted yet.")
    else:
        st.error("Could not load applications. Is the API running?")

with tab3:
    st.subheader("Ask the Assistant About an Application")
    app_id = st.number_input("Application ID", min_value=1, step=1)
    question = st.text_area(
        "Question", placeholder="e.g. Why was this application flagged as high risk?"
    )
    if st.button("Ask"):
        resp = requests.post(
            f"{API_URL}/assistant/ask", json={"application_id": int(app_id), "question": question}
        )
        if resp.ok:
            st.write(resp.json()["answer"])
        else:
            st.error(f"Could not get an answer: {resp.text}")
