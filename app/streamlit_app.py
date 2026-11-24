
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ============================================================
# PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recommendation import weighted_recommend
from src.preprocessing import transform_profile


st.set_page_config(
    page_title="DormMatch | Find Your Roommate",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================
# ============================================================
st.markdown("""
<style>
/* ---------- Base ---------- */
html, body, [class*="css"] {
    font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
}

.stApp {
    background-color: #FAFAF8;
}

.main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.stApp p, .stApp label, .stApp h1, .stApp h2,
.stApp h3, .stApp h4, .stApp span {
    color: #2A2A2A !important;
}

/* ---------- Header ---------- */
.main-title {
    font-size: 40px;
    font-weight: 700;
    color: #1F3A5F !important;
    text-align: center;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}

.subtitle {
    font-size: 15px;
    color: #6B6B6B !important;
    text-align: center;
    margin-bottom: 28px;
}

/* ---------- Section headings ---------- */
.section-title {
    font-size: 20px;
    font-weight: 600;
    color: #1F3A5F !important;
    margin-top: 18px;
    margin-bottom: 12px;
}

/* ---------- Form box ---------- */
[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E3E3DE;
    border-radius: 10px;
    padding: 28px 26px 20px 26px;
}

/* Subheads inside form */
[data-testid="stForm"] h3 {
    font-size: 16px;
    font-weight: 600;
    color: #1F3A5F !important;
    margin-top: 10px;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid #EEEEE9;
}

/* ---------- Labels ---------- */
.stApp label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #4A4A4A !important;
}

/* ---------- Inputs ---------- */
.stTextInput input,
.stNumberInput input {
    background-color: #FFFFFF !important;
    color: #2A2A2A !important;
    border-radius: 6px !important;
    border: 1px solid #D8D8D2 !important;
}

.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #1F3A5F !important;
    box-shadow: none !important;
}

/* ---------- Selectboxes ---------- */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #2A2A2A !important;
    border-radius: 6px !important;
    border: 1px solid #D8D8D2 !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"] {
    background-color: #FFFFFF !important;
    border-radius: 6px !important;
    border: 1px solid #E3E3DE !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
}

/* ---------- Slider ---------- */
[data-testid="stSlider"] [role="slider"] {
    background-color: #1F3A5F !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 0 0 1px #1F3A5F !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background: #1F3A5F !important;
}

[data-testid="stThumbValue"] {
    color: #1F3A5F !important;
    font-weight: 600 !important;
}

/* ---------- Info box ---------- */
[data-testid="stInfo"] {
    background: #F5F7F9 !important;
    border: 1px solid #E3E3DE !important;
    border-left: 3px solid #1F3A5F !important;
    border-radius: 6px !important;
    color: #3A3A3A !important;
}

[data-testid="stInfo"] p {
    color: #3A3A3A !important;
    font-size: 13.5px !important;
}

/* ---------- Submit button ---------- */
.stFormSubmitButton button {
    background-color: #1F3A5F !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    height: 48px;
    font-size: 15px;
    font-weight: 600;
}

.stFormSubmitButton button:hover {
    background-color: #162C48 !important;
    color: #FFFFFF !important;
}

/* ---------- Match cards ---------- */
.match-card {
    background: #FFFFFF;
    border: 1px solid #E3E3DE;
    border-radius: 10px;
    padding: 18px 20px;
    margin-top: 16px;
    margin-bottom: 12px;
}

.match-card h3 {
    font-size: 18px;
    font-weight: 600;
    color: #1F3A5F !important;
    margin: 0 0 4px 0;
}

.match-score {
    color: #1F3A5F !important;
    font-size: 24px;
    font-weight: 700;
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E3E3DE;
    border-radius: 8px;
    padding: 12px 14px !important;
}

[data-testid="stMetricValue"] {
    color: #1F3A5F !important;
    font-weight: 600 !important;
}

[data-testid="stMetricLabel"] {
    color: #6B6B6B !important;
    font-size: 12px !important;
}

/* ---------- Progress ---------- */
[data-testid="stProgress"] > div > div > div {
    background: #1F3A5F !important;
}

[data-testid="stProgress"] > div > div {
    background: #EEEEE9 !important;
}

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E3E3DE !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    color: #1F3A5F !important;
}

/* ---------- Success alert ---------- */
.stSuccess {
    background: #F0F7F2 !important;
    color: #2D5A3D !important;
    border: 1px solid #C8DFD0 !important;
    border-left: 3px solid #4A8B5F !important;
    border-radius: 6px !important;
}

.stSuccess p { color: #2D5A3D !important; }

/* ---------- Chart container ---------- */
[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid #E3E3DE;
    border-radius: 10px;
    padding: 8px;
}

/* ---------- Divider ---------- */
hr {
    border: none;
    border-top: 1px solid #E3E3DE;
    margin: 22px 0;
}

/* ---------- Caption ---------- */
.stApp [data-testid="stCaptionContainer"] p {
    color: #8A8A85 !important;
    font-size: 12px !important;
}

/* ---------- Hide chrome ---------- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA AND EXISTING PREPROCESSING OBJECTS
# ============================================================

@st.cache_resource
def load_artifacts():

    cleaned = pd.read_csv(
        ROOT / "data/dormmatch_cleaned.csv"
    )

    raw = pd.read_csv(
        ROOT / "data/dormmatch_raw.csv"
    )

    assignments = pd.read_csv(
        ROOT / "models/cluster_assignments.csv"
    )

    preprocessing = joblib.load(
        ROOT / "models/preprocessing.joblib"
    )

    reference = joblib.load(
        ROOT / "models/reference_feature_matrix.joblib"
    )

    # Keep one raw profile record per student for
    # guest and music frequency.
    raw_profile = raw[
        ["student_id", "guest_frequency", "music_frequency"]
    ].drop_duplicates(
        "student_id",
        keep="first"
    )

    # Build candidate student pool.
    candidates = (
        cleaned
        .merge(
            raw_profile,
            on="student_id",
            how="left",
            validate="one_to_one"
        )
        .merge(
            assignments,
            on="student_id",
            how="left",
            validate="one_to_one"
        )
    )

    if len(candidates) != len(reference):
        raise ValueError(
            "Candidate records and reference feature matrix "
            "have different row counts."
        )

    if len(candidates) != len(assignments):
        raise ValueError(
            "Candidate records and cluster assignments "
            "have different row counts."
        )

    if not candidates["student_id"].is_unique:
        raise ValueError("Duplicate student IDs in candidate pool.")

    return {
        "cleaned": cleaned,
        "raw": raw,
        "candidates": candidates,
        "reference": reference,
        "preprocessing": preprocessing,
    }


try:
    artifacts = load_artifacts()
    candidates = artifacts["candidates"]

except Exception as e:
    st.error(f"Unable to load DormMatch data or models: {e}")
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_food(value):

    value = str(value).strip().lower()

    mapping = {
        "vegetarian": "Vegetarian",
        "non-vegetarian": "Non-vegetarian",
        "non vegetarian": "Non-vegetarian",
        "eggetarian": "Eggetarian",
        "vegan": "Vegan"
    }

    return mapping.get(value, value.title())


def normalize_environment(value):

    mapping = {
        "quiet": "Quiet",
        "social": "Social",
        "balanced": "Balanced"
    }

    return mapping.get(str(value).lower(), "Balanced")


def circular_difference(t1, t2):

    difference = abs(float(t1) - float(t2))

    return min(difference, 24 - difference)


def get_reasons(query, candidate):

    reasons = []

    # Sleep compatibility
    sleep_diff = circular_difference(
        query["sleep_time"],
        candidate["sleep_time"]
    )

    if sleep_diff <= 1.5:
        reasons.append("Similar sleeping schedules")

    # Wake compatibility
    wake_diff = circular_difference(
        query["wake_time"],
        candidate["wake_time"]
    )

    if wake_diff <= 1.5:
        reasons.append("Similar waking schedules")

    # Study compatibility
    study_diff = abs(
        float(query["study_hours"]) -
        float(candidate["study_hours"])
    )

    if study_diff <= 2:
        reasons.append("Similar study routines")

    # Cleanliness
    clean_diff = abs(
        float(query["cleanliness_level"]) -
        float(candidate["cleanliness_level"])
    )

    if clean_diff <= 1:
        reasons.append("Similar cleanliness preferences")

    # Noise
    noise_diff = abs(
        float(query["noise_tolerance"]) -
        float(candidate["noise_tolerance"])
    )

    if noise_diff <= 1:
        reasons.append("Similar noise preferences")

    # Environment
    if (
        str(query["preferred_environment"]).lower()
        == str(candidate["preferred_environment"]).lower()
    ):
        reasons.append("Same preferred room environment")

    # Food
    if (
        normalize_food(query["food_habit"])
        == normalize_food(candidate["food_habit"])
    ):
        reasons.append("Same food preference")

    # AC
    if (
        str(query["ac_preference"]).lower()
        == str(candidate["ac_preference"]).lower()
    ):
        reasons.append("Same AC preference")

    if not reasons:
        reasons.append(
            "Ranked highly by overall lifestyle compatibility"
        )

    return reasons


def get_score(row):

    if "overall" in row.index:
        return float(row["overall"])

    if "compatibility_score" in row.index:
        return float(row["compatibility_score"])

    raise ValueError(
        "Recommendation output must contain an overall "
        "or compatibility_score column."
    )


def show_compatibility_breakdown(row):

    # Display actual breakdown returned by recommendation module.
    possible_scores = {
        "Sleep": ["sleep", "sleep_score", "sleep_compatibility"],
        "Study": ["study", "study_score", "study_compatibility"],
        "Cleanliness": [
            "cleanliness",
            "cleanliness_score",
            "cleanliness_compatibility"
        ],
        "Noise": ["noise", "noise_score", "noise_compatibility"],
        "Social": ["social", "social_score", "social_compatibility"],
        "Guest": ["guest", "guest_score", "guest_compatibility"],
        "Music": ["music", "music_score", "music_compatibility"],
    }

    found = {}

    for label, names in possible_scores.items():

        for name in names:

            if name in row.index:

                found[label] = float(row[name])
                break

    if found:

        st.markdown("**Compatibility breakdown**")

        cols = st.columns(min(len(found), 4))

        for i, (label, value) in enumerate(found.items()):

            with cols[i % len(cols)]:
                st.metric(
                    label,
                    f"{np.clip(value, 0, 100):.0f}%"
                )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏠 DormMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Find a roommate who matches your lifestyle, habits and preferences.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")

st.markdown(
    '<div class="section-title">👤 Tell us about yourself</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter your lifestyle preferences below. "
    "DormMatch will compare your profile against existing students "
    "and find your most compatible roommates."
)


# ============================================================
# STUDENT PROFILE FORM
# ============================================================

with st.form("roommate_profile_form"):

    st.subheader("Basic Information")

    c1, c2, c3 = st.columns(3)

    with c1:

        age = st.number_input(
            "Age",
            min_value=17,
            max_value=30,
            value=20
        )

        year = st.selectbox(
            "Year of study",
            [1, 2, 3, 4],
            index=2
        )

    with c2:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"]
        )

        branch = st.selectbox(
            "Branch",
            [
                "Computer Science",
                "Electronics and Communication",
                "Electrical Engineering",
                "Mechanical Engineering",
                "Civil Engineering",
                "Biotechnology",
                "Other"
            ]
        )

    with c3:

        budget = st.number_input(
            "Monthly room budget",
            min_value=2500,
            max_value=50000,
            value=6500,
            step=500
        )

        room_type = st.selectbox(
            "Preferred room type",
            ["Single", "Double", "Triple"]
        )

    st.markdown("---")

    st.subheader("Sleep and Study Habits")

    c1, c2, c3 = st.columns(3)

    with c1:

        sleep_time = st.slider(
            "Sleep time (24-hour format)",
            0.0,
            23.75,
            23.0,
            0.25
        )

    with c2:

        wake_time = st.slider(
            "Wake time (24-hour format)",
            0.0,
            23.75,
            7.0,
            0.25
        )

    with c3:

        sleep_duration = st.slider(
            "Sleep duration (hours)",
            1.0,
            18.0,
            8.0,
            0.25
        )

    study_hours = st.slider(
        "Daily study hours",
        0.0,
        18.0,
        4.0,
        0.5
    )

    st.markdown("---")

    st.subheader("Lifestyle Preferences")

    c1, c2, c3 = st.columns(3)

    with c1:

        cleanliness = st.slider(
            "Cleanliness level",
            1,
            5,
            3
        )

        noise = st.slider(
            "Noise tolerance",
            1,
            5,
            3
        )

        social = st.slider(
            "Social activity",
            1,
            5,
            3
        )

    with c2:

        introvert = st.slider(
            "Introversion / Extroversion",
            1,
            5,
            3
        )

        sharing = st.slider(
            "Sharing preference",
            1,
            5,
            3
        )

        interaction = st.slider(
            "Roommate interaction preference",
            1,
            5,
            3
        )

    with c3:

        guest = st.selectbox(
            "Guest frequency",
            ["Rarely", "Sometimes", "Often", "Very often"]
        )

        music = st.selectbox(
            "Music frequency",
            ["Never", "Rarely", "Sometimes", "Often"]
        )

        weekend = st.selectbox(
            "Weekend activity",
            [
                "Gaming",
                "Movies",
                "Relaxing",
                "Outings",
                "Sports",
                "Study"
            ]
        )

    st.markdown("---")

    st.subheader("Room and Personal Preferences")

    c1, c2, c3 = st.columns(3)

    with c1:

        food = st.selectbox(
            "Food habit",
            [
                "Vegetarian",
                "Non-vegetarian",
                "Eggetarian",
                "Vegan"
            ]
        )

        smoking = st.selectbox(
            "Smoking preference",
            [
                "No smoking",
                "No preference",
                "Prefer smoke-free"
            ]
        )

    with c2:

        ac = st.selectbox(
            "AC preference",
            ["Yes", "No", "No preference"]
        )

        environment = st.selectbox(
            "Preferred environment",
            ["Quiet", "Balanced", "Social"]
        )

    with c3:

        st.info(
            "Your profile is used only for matching. "
            "It is not added to the existing student dataset."
        )

    st.markdown("---")

    st.subheader("Ready to find your roommate?")

    st.caption(
        "DormMatch automatically uses weighted lifestyle "
        "compatibility to rank your matches."
    )

    submitted = st.form_submit_button(
        "🔍 Find My Roommate",
        use_container_width=True
    )


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

if submitted:

    profile = {
        "student_id": "NEW_STUDENT",
        "age": age,
        "gender": gender,
        "year_of_study": year,
        "branch": branch,

        "sleep_time": sleep_time,
        "wake_time": wake_time,
        "sleep_duration": sleep_duration,
        "study_hours": study_hours,

        "cleanliness_level": cleanliness,
        "noise_tolerance": noise,
        "social_activity": social,
        "introvert_extrovert": introvert,
        "sharing_preference": sharing,

        "guest_frequency": guest.lower(),
        "music_frequency": music.lower(),
        "weekend_activity": weekend,

        "food_habit": normalize_food(food),
        "smoking_preference": smoking,
        "ac_preference": ac,

        "preferred_room_type": room_type,
        "budget": budget,
        "preferred_environment": normalize_environment(environment),
        "roommate_interaction": interaction,
    }

    st.session_state["student_profile"] = profile

    with st.spinner("Finding your most compatible roommates..."):

        try:

            # Transform the new student using the saved
            # training-time preprocessing objects.
            query_vector = transform_profile(
                profile,
                artifacts["preprocessing"]["feature_columns"],
                artifacts["preprocessing"]["scaler"],
                artifacts["preprocessing"]["continuous_columns"]
            )

            query_vector = np.asarray(query_vector)

            query_vector = query_vector.reshape(1, -1)

            # Weighted compatibility is the automatic algorithm.
            # No algorithm selection is exposed to the student.
            result = weighted_recommend(
                pd.Series(profile),
                candidates,
                top_k=5,
                constraints={
                    "room_type": "Any",
                    "ac_preference": "Any",
                    "food_preference": "Any"
                }
            )

            if result is None or len(result) == 0:
                st.session_state.pop("recommendations", None)
                st.warning(
                    "No compatible students were found. "
                    "Please review your profile."
                )
                st.stop()

            result = result.sort_values(
                "overall",
                ascending=False
            ).head(5).copy()

            st.session_state["recommendations"] = result
            st.session_state["query_profile"] = profile

        except Exception as e:

            st.error(
                f"Unable to generate recommendations: {str(e)}"
            )

            st.stop()


# ============================================================
# DISPLAY RECOMMENDATIONS
# ============================================================

if "recommendations" in st.session_state:

    result = st.session_state["recommendations"]

    query = st.session_state["query_profile"]

    st.markdown("---")

    st.markdown(
        '<div class="section-title">'
        '🎯 Your Most Compatible Roommates'
        '</div>',
        unsafe_allow_html=True
    )

    st.success(
        f"We found {len(result)} potential roommate matches for you!"
    )

    for rank, (_, row) in enumerate(result.iterrows(), start=1):

        student_id = row["student_id"]

        matches = candidates[
            candidates["student_id"] == student_id
        ]

        if matches.empty:
            continue

        candidate = matches.iloc[0]

        score = np.clip(get_score(row), 0, 100)

        st.markdown(
            f"""
            <div class="match-card">
                <h3>#{rank} &nbsp; {student_id}</h3>
                <div class="match-score">
                    {score:.1f}% Compatibility
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(float(score / 100))

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Sleep time",
            f"{float(candidate['sleep_time']):.2f}"
        )

        c2.metric(
            "Study hours",
            f"{float(candidate['study_hours']):.1f} hrs"
        )

        c3.metric(
            "Cleanliness",
            f"{candidate['cleanliness_level']}/5"
        )

        c4.metric(
            "Noise tolerance",
            f"{candidate['noise_tolerance']}/5"
        )

        # Actual compatibility breakdown from the model.
        show_compatibility_breakdown(row)

        st.markdown("**Why this student matches you**")

        reasons = get_reasons(query, candidate)

        for reason in reasons:
            st.write(f"✓ {reason}")

        with st.expander(
            f"View {student_id}'s lifestyle profile"
        ):

            details = {
                "Sleep time": candidate["sleep_time"],
                "Wake time": candidate["wake_time"],
                "Sleep duration": candidate["sleep_duration"],
                "Study hours": candidate["study_hours"],
                "Cleanliness": candidate["cleanliness_level"],
                "Noise tolerance": candidate["noise_tolerance"],
                "Social activity": candidate["social_activity"],
                "Guest frequency": candidate["guest_frequency"],
                "Music frequency": candidate["music_frequency"],
                "Food habit": candidate["food_habit"],
                "Room type": candidate["preferred_room_type"],
                "Environment": candidate["preferred_environment"],
                "Budget": candidate["budget"],
            }

            st.dataframe(
                pd.DataFrame(
                    details.items(),
                    columns=["Attribute", "Value"]
                ),
                hide_index=True,
                use_container_width=True
            )

        st.markdown("---")

    # Compatibility comparison chart.
    chart_data = result[
        ["student_id", "overall"]
    ].copy()

    chart_data["overall"] = chart_data["overall"].clip(0, 100)

    fig = px.bar(
        chart_data,
        x="student_id",
        y="overall",
        color="overall",
        color_continuous_scale="Blues",
        title="Top 5 Roommate Compatibility Scores",
        labels={
            "student_id": "Student",
            "overall": "Compatibility (%)"
        }
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#26354A"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Compatibility scores are model-generated estimates "
        "based on synthetic student lifestyle data. They do not "
        "represent measured real-world roommate satisfaction."
    )

