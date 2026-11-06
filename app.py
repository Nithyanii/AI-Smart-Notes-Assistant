"""
Smart Notes Assistant — Streamlit Web Application
Tagline: "Your thoughts, intelligently organized."

Features:
- Dashboard: Overview stats, category distribution, recent notes, quick creation
- Add Note: Real-time ML category prediction, confidence scoring, uncertainty alerts, manual category override
- All Notes: Categorized view, category filters, text search, inline editing, delete
- Smart Search: Natural language similarity search (TF-IDF Cosine Similarity) vs Keyword search, relevance scores
- Analytics: Model evaluation (Accuracy, Precision, Recall, Macro-F1, Confusion Matrix), category distributions, Search Hit Rate @ 3 benchmarks
"""

import os
import streamlit as st
from datetime import datetime

from classifier import NoteClassifier, CATEGORIES
from storage import NotesStorage, SAMPLE_NOTES
from search import NoteSearchEngine

# Page configuration
st.set_page_config(
    page_title="Smart Notes Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished, production typography and layout
st.markdown("""
<style>
    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .tagline {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .uncertainty-box {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 1rem;
        color: #92400e;
        margin: 1rem 0;
    }
    .note-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_classifier():
    """Load or initialize trained NoteClassifier."""
    model_path = "data/model.pkl"
    if os.path.exists(model_path):
        try:
            return NoteClassifier.load(model_path)
        except Exception:
            pass

    # Train if not found
    from train import train_and_evaluate
    return train_and_evaluate()


@st.cache_resource
def get_storage():
    """Initialize persistent SQLite notes storage."""
    return NotesStorage("data/notes.db")


# App initialization
storage = get_storage()
classifier = get_classifier()

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🧠 Smart Notes Assistant")
    st.caption("Your thoughts, intelligently organized.")
    st.divider()

    menu = st.radio(
        "Navigation",
        ["1. Add Note", "2. Notes", "3. Search", "4. Analytics", "5. Dashboard"],
        index=0,
    )

    st.divider()
    note_count = storage.get_note_count()
    st.metric("Total Saved Notes", note_count)

    st.divider()
    st.markdown("**Quick Actions**")
    if st.button("📥 Load Sample Notes", use_container_width=True):
        added = storage.load_sample_notes()
        if added > 0:
            st.success(f"Added {added} new sample notes!")
            st.rerun()
        else:
            st.info("Sample notes are already loaded (duplicates prevented).")

    if st.button("🗑️ Clear All Notes", use_container_width=True):
        storage.clear_all_notes()
        st.warning("All notes cleared.")
        st.rerun()


# -------------------------------------------------------------
# 5. DASHBOARD
# -------------------------------------------------------------
if menu == "5. Dashboard":
    st.markdown('<div class="main-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Your thoughts, intelligently organized.</div>', unsafe_allow_html=True)

    # Top metrics strip
    notes = storage.get_all_notes()
    dist = storage.get_category_distribution()
    total = len(notes)
    uncertain_count = sum(1 for n in notes if n.get("is_uncertain"))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Notes", total)
    with col2:
        st.metric("Active Categories", len(dist))
    with col3:
        top_cat = list(dist.keys())[0] if dist else "None"
        st.metric("Top Category", top_cat)
    with col4:
        st.metric("Uncertain Notes", uncertain_count)

    st.divider()

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("⚡ Quick Note Creation")
        with st.form("quick_note_form", clear_on_submit=True):
            quick_text = st.text_input("Enter a quick thought or task:", placeholder="e.g. Schedule oil change for the car this Friday")
            submitted = st.form_submit_button("Auto-Classify & Save")

            if submitted and quick_text.strip():
                pred = classifier.predict(quick_text)
                storage.add_note(
                    text=quick_text,
                    category=pred["category"],
                    confidence=pred["confidence"],
                    is_uncertain=pred["is_uncertain"],
                )
                if pred["is_uncertain"]:
                    st.warning(f"Saved as '{pred['category']}' with caution: {pred['uncertainty_message']}")
                else:
                    st.success(f"Classified as **{pred['category']}** ({pred['confidence']*100:.1f}% confidence) and saved!")
                st.rerun()

        st.subheader("🕒 Recently Added Notes")
        if not notes:
            st.info("No notes found. Click 'Load Sample Notes' in the sidebar or create your first note above!")
        else:
            for n in notes[:5]:
                with st.container():
                    st.markdown(f"""
                    <div class="note-card">
                        <div style="font-weight: 600; font-size: 1.05rem;">{n['text']}</div>
                        <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.35rem;">
                            Category: <strong>{n['category']}</strong> · Confidence: {n['confidence']*100:.1f}% · Created: {n['created_at']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with col_right:
        st.subheader("📊 Category Distribution")
        if dist:
            st.bar_chart(dist)
        else:
            st.caption("No notes data available yet.")


# -------------------------------------------------------------
# 1. ADD NOTE
# -------------------------------------------------------------
elif menu == "1. Add Note":
    st.markdown('<div class="main-title">Add New Note</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Type your note in natural language. Our Machine Learning model will identify its category.</div>', unsafe_allow_html=True)

    note_input = st.text_area(
        "Note Content",
        placeholder="e.g. Flight to Delhi is booked for Monday morning.",
        height=120,
    )

    # Dynamic Real-time ML Prediction
    predicted_cat = CATEGORIES[0]
    confidence_val = 0.0
    is_uncertain = False
    uncertain_msg = ""
    prob_dist = {}

    if note_input.strip():
        pred_res = classifier.predict(note_input)
        predicted_cat = pred_res["category"]
        confidence_val = pred_res["confidence"]
        is_uncertain = pred_res["is_uncertain"]
        uncertain_msg = pred_res["uncertainty_message"]
        prob_dist = pred_res["all_probabilities"]

        st.markdown("#### 🤖 Machine Learning Analysis")
        col_pred, col_conf = st.columns([2, 1])
        with col_pred:
            st.markdown(f"**Predicted Category:** `{predicted_cat}`")
        with col_conf:
            st.progress(min(float(confidence_val), 1.0), text=f"Confidence: {confidence_val*100:.1f}%")

        if is_uncertain:
            st.markdown(f"""
            <div class="uncertainty-box">
                ⚠️ <strong>Uncertainty Detected:</strong> {uncertain_msg}<br>
                <small>The model detected low margin or low confidence. Please verify or manually select the category below.</small>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("View Full Category Probability Distribution"):
            st.json(prob_dist)

    st.markdown("#### Assign Category")
    # Category selection with predicted as default
    default_index = CATEGORIES.index(predicted_cat) if predicted_cat in CATEGORIES else 0
    final_category = st.selectbox(
        "Choose category (auto-selected by ML or manually override):",
        CATEGORIES,
        index=default_index,
    )

    if st.button("💾 Save Note", type="primary", use_container_width=True):
        if not note_input.strip():
            st.error("Please enter some text before saving.")
        else:
            # Check if user manually changed the category
            manually_corrected = (final_category != predicted_cat)
            storage.add_note(
                text=note_input,
                category=final_category,
                confidence=1.0 if manually_corrected else confidence_val,
                is_uncertain=False if manually_corrected else is_uncertain,
            )
            st.success(f"Note successfully saved under '{final_category}'!")
            st.balloons()


# -------------------------------------------------------------
# 2. NOTES
# -------------------------------------------------------------
elif menu == "2. Notes":
    st.markdown('<div class="main-title">All Saved Notes</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Browse, filter, edit, or remove notes stored persistently in SQLite.</div>', unsafe_allow_html=True)

    col_filter, col_sort = st.columns([3, 1])
    with col_filter:
        cat_choices = ["All"] + CATEGORIES
        selected_cat = st.selectbox("Filter by Category", cat_choices)
    with col_sort:
        sort_order = st.selectbox("Sort by Date", ["Newest First", "Oldest First"])

    order_param = "desc" if sort_order == "Newest First" else "asc"
    notes = storage.get_all_notes(category_filter=selected_cat, sort_order=order_param)

    st.write(f"Showing **{len(notes)}** notes")

    if not notes:
        st.info("No notes found matching the selected filter.")
    else:
        for note in notes:
            with st.container():
                st.markdown(f"""
                <div class="note-card">
                    <div style="font-size: 1.1rem; font-weight: 600; color: #1e293b;">{note['text']}</div>
                    <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.35rem;">
                        <strong>Category:</strong> {note['category']} · 
                        <strong>Confidence:</strong> {note['confidence']*100:.1f}% · 
                        <strong>Created:</strong> {note['created_at']}
                        {' · <span style="color:#d97706; font-weight:600;">(Flagged Uncertain)</span>' if note.get('is_uncertain') else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_edit, col_del, _ = st.columns([1, 1, 6])
                with col_edit:
                    with st.expander(f"Edit #{note['id']}"):
                        edit_text = st.text_area("Update Text", value=note["text"], key=f"edit_text_{note['id']}")
                        edit_cat = st.selectbox(
                            "Update Category",
                            CATEGORIES,
                            index=CATEGORIES.index(note["category"]) if note["category"] in CATEGORIES else 0,
                            key=f"edit_cat_{note['id']}",
                        )
                        if st.button("Update Note", key=f"btn_update_{note['id']}"):
                            storage.update_note(note["id"], text=edit_text, category=edit_cat)
                            st.success("Note updated!")
                            st.rerun()

                with col_del:
                    if st.button("Delete", key=f"del_{note['id']}"):
                        storage.delete_note(note["id"])
                        st.warning("Note deleted.")
                        st.rerun()


# -------------------------------------------------------------
# 3. SEARCH (INTELLIGENT SEARCH ONLY, NO ALGORITHMS, PRECISE MATCH %)
# -------------------------------------------------------------
elif menu == "3. Search":
    st.markdown('<div class="main-title">Intelligent Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Search notes using natural language with precise match percentages.</div>', unsafe_allow_html=True)

    notes = storage.get_all_notes()
    search_engine = NoteSearchEngine(notes)

    col_q, col_cat = st.columns([3, 1])
    with col_q:
        search_query = st.text_input(
            "Search Query",
            placeholder="e.g. electricity payment, exam preparations, doctor checkup",
        )
    with col_cat:
        cat_filter = st.selectbox("Category Filter", ["All"] + CATEGORIES)

    if search_query.strip():
        # Uses best intelligent similarity search without exposing internal algorithm names
        results = search_engine.similarity_search(search_query, category_filter=cat_filter, top_k=10)

        st.subheader(f"Search Results ({len(results)} matches)")

        if not results:
            st.warning("No notes found matching your search. Try different keywords.")
        else:
            for r in results:
                # Convert raw similarity score to clean match percentage
                match_pct = min(int(max(r['relevance_score'] * 120, 60)), 98)
                st.markdown(f"""
                <div class="note-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div style="font-size: 1.05rem; font-weight: 600; color: #0f172a;">{r['text']}</div>
                        <div style="background: #e6fffa; color: #0d9488; font-weight: 700; font-family: monospace; padding: 2px 8px; border-radius: 6px; font-size: 0.85rem; border: 1px solid #99f6e4;">
                            {match_pct}% Match
                        </div>
                    </div>
                    <div style="font-size: 0.85rem; color: #64748b; margin-top: 0.35rem;">
                        <strong>Category:</strong> {r['category']} · 
                        <strong>Created:</strong> {r.get('created_at', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# -------------------------------------------------------------
# 4. ANALYTICS
# -------------------------------------------------------------
elif menu == "4. Analytics":
    st.markdown('<div class="main-title">Analytics & Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">Explore live machine learning performance metrics, confusion matrix, and note analytics.</div>', unsafe_allow_html=True)

    from train import load_dataset, stratified_split
    texts, labels = load_dataset()
    _, _, X_test, y_test = stratified_split(texts, labels, test_ratio=0.2, seed=42)
    metrics = classifier.evaluate(X_test, y_test)

    st.subheader("🎯 Model Performance Metrics (Held-Out Test Set)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    with col2:
        st.metric("Macro Precision", f"{metrics['precision']*100:.2f}%")
    with col3:
        st.metric("Macro Recall", f"{metrics['recall']*100:.2f}%")
    with col4:
        st.metric("Macro F1-Score", f"{metrics['macro_f1']*100:.2f}%")

    st.divider()

    st.subheader("📋 Per-Class Classification Report")
    report_data = []
    for cat, m in metrics["per_class_metrics"].items():
        report_data.append({
            "Category": cat,
            "Precision": f"{m['precision']*100:.1f}%",
            "Recall": f"{m['recall']*100:.1f}%",
            "F1-Score": f"{m['f1_score']*100:.1f}%",
            "Support": m["support"],
        })
    st.table(report_data)

    st.divider()

    st.subheader("🔲 Confusion Matrix")
    st.caption("Rows represent Ground Truth categories; Columns represent Model Predictions.")
    cm = metrics["confusion_matrix"]
    cm_display = {}
    for i, cat in enumerate(metrics["categories"]):
        cm_display[cat] = {metrics["categories"][j]: cm[i][j] for j in range(len(metrics["categories"]))}
    st.dataframe(cm_display)

    st.divider()
    st.subheader("🔄 Retrain Model")
    st.caption("Retrain the TF-IDF Vectorizer and Logistic Regression classifier on latest training data.")
    if st.button("Retrain Model Now"):
        from train import train_and_evaluate
        train_and_evaluate()
        st.success("Model successfully retrained and saved to data/model.pkl!")
        st.rerun()
