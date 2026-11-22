import os
import streamlit as st
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="AI Candidate Screening",
    page_icon="👥",
    layout="wide",
)

st.title("AI Candidate Screening")
st.caption("LangGraph-powered candidate screening workflow")

st.markdown(
    """
    Submit a candidate's details and application/resume text. The LangGraph
    workflow will classify experience, assess the required skills, and route
    the application to the appropriate next step.
    """
)


# -----------------------------
# LangGraph state
# -----------------------------
class State(TypedDict, total=False):
    application: str
    job_role: str
    experience_level: str
    skill_match: str
    response: str


def get_api_key():
    """Read the OpenAI key from Streamlit secrets or environment variables."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY")


def build_workflow(api_key: str):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    def categorize_experience(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            """
            Based on the following job application, categorize the candidate
            as exactly one of: Entry-level, Mid-level, or Senior-level.

            Return only the category.

            Job application:
            {application}
            """
        )
        result = (prompt | llm).invoke(
            {"application": state["application"]}
        ).content

        experience_level = str(result).strip()
        if "senior" in experience_level.lower():
            experience_level = "Senior-level"
        elif "mid" in experience_level.lower():
            experience_level = "Mid-level"
        else:
            experience_level = "Entry-level"

        return {"experience_level": experience_level}

    def assess_skillset(state: State) -> State:
        prompt = ChatPromptTemplate.from_template(
            """
            Assess whether the candidate's application matches the skills
            required for the following job role: {job_role}.

            Respond with exactly one of these values:
            Match
            No Match

            Job application:
            {application}
            """
        )
        result = (prompt | llm).invoke(
            {
                "application": state["application"],
                "job_role": state["job_role"],
            }
        ).content

        skill_match = str(result).strip()
        skill_match = (
            "Match"
            if "no match" not in skill_match.lower()
            and "match" in skill_match.lower()
            else "No Match"
        )

        return {"skill_match": skill_match}

    def schedule_hr_interview(state: State) -> State:
        return {
            "response": "Candidate has been shortlisted for an HR interview."
        }

    def escalate_to_recruiter(state: State) -> State:
        return {
            "response": (
                "Candidate has senior-level experience but doesn't match "
                "the required job skills. Escalated to the recruiter."
            )
        }

    def reject_application(state: State) -> State:
        return {
            "response": "Candidate doesn't meet the job requirements and has been rejected."
        }

    def route_app(state: State) -> str:
        if state["skill_match"] == "Match":
            return "schedule_hr_interview"
        elif state["experience_level"] == "Senior-level":
            return "escalate_to_recruiter"
        return "reject_application"

    workflow = StateGraph(State)

    workflow.add_node("categorize_experience", categorize_experience)
    workflow.add_node("assess_skillset", assess_skillset)
    workflow.add_node("schedule_hr_interview", schedule_hr_interview)
    workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
    workflow.add_node("reject_application", reject_application)

    workflow.add_edge(START, "categorize_experience")
    workflow.add_edge("categorize_experience", "assess_skillset")

    workflow.add_conditional_edges(
        "assess_skillset",
        route_app,
        {
            "schedule_hr_interview": "schedule_hr_interview",
            "escalate_to_recruiter": "escalate_to_recruiter",
            "reject_application": "reject_application",
        },
    )

    workflow.add_edge("schedule_hr_interview", END)
    workflow.add_edge("escalate_to_recruiter", END)
    workflow.add_edge("reject_application", END)

    return workflow.compile()


# -----------------------------
# Candidate input form
# -----------------------------
with st.form("candidate_form"):
    col1, col2 = st.columns(2)

    with col1:
        candidate_name = st.text_input("Candidate name", placeholder="Rahul")
        job_role = st.text_input(
            "Job role",
            value="Python Developer",
        )

    with col2:
        experience_years = st.number_input(
            "Experience (years)",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
        )

    application_text = st.text_area(
        "Candidate application / resume",
        height=220,
        placeholder=(
            "Paste the candidate's resume, skills, work experience, "
            "projects and other relevant details here..."
        ),
    )

    submitted = st.form_submit_button(
        "Screen candidate",
        type="primary",
        use_container_width=True,
    )


# -----------------------------
# Run workflow and show result
# -----------------------------
if submitted:
    if not candidate_name.strip():
        st.error("Please enter the candidate name.")
    elif not job_role.strip():
        st.error("Please enter the job role.")
    elif not application_text.strip():
        st.error("Please paste the candidate's application or resume.")
    else:
        api_key = get_api_key()

        if not api_key:
            st.error(
                "OpenAI API key not found. Add OPENAI_API_KEY to your "
                "environment variables or Streamlit secrets."
            )
        else:
            application = f"""
Candidate name: {candidate_name}
Job role: {job_role}
Total experience: {experience_years} years

Candidate application:
{application_text}
"""

            try:
                with st.spinner("Running candidate screening workflow..."):
                    app = build_workflow(api_key)
                    result = app.invoke(
                        {
                            "application": application,
                            "job_role": job_role,
                        }
                    )

                st.divider()
                st.subheader("Screening result")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Experience level",
                        result.get("experience_level", "Not available"),
                    )
                with col2:
                    st.metric(
                        "Skill match",
                        result.get("skill_match", "Not available"),
                    )

                response = result.get("response", "No response generated.")

                if "shortlisted" in response.lower():
                    st.success(response)
                elif "escalated" in response.lower():
                    st.warning(response)
                else:
                    st.error(response)

                with st.expander("View workflow details"):
                    st.write("**Candidate:**", candidate_name)
                    st.write("**Job role:**", job_role)
                    st.write("**Experience provided:**", f"{experience_years} years")
                    st.write("**Experience classification:**", result.get("experience_level"))
                    st.write("**Skill assessment:**", result.get("skill_match"))
                    st.write("**Final action:**", response)

            except Exception as exc:
                st.error(f"Unable to complete candidate screening: {exc}")


# -----------------------------
# Workflow overview
# -----------------------------
with st.expander("View LangGraph workflow"):
    st.graphviz_chart(
        """
        digraph {
            rankdir=TB;
            node [shape=box, style="rounded,filled", fillcolor="#EEF2FF", color="#818CF8"];
            start [label="START", shape=oval];
            categorize [label="categorize_experience"];
            assess [label="assess_skillset"];
            schedule [label="schedule_hr_interview"];
            escalate [label="escalate_to_recruiter"];
            reject [label="reject_application"];
            end1 [label="END", shape=oval];
            end2 [label="END", shape=oval];
            end3 [label="END", shape=oval];

            start -> categorize -> assess;
            assess -> schedule [label="Match"];
            assess -> escalate [label="Senior-level + No Match"];
            assess -> reject [label="Otherwise"];
            schedule -> end1;
            escalate -> end2;
            reject -> end3;
        }
        """,
        use_container_width=True,
    )
