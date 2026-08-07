"""
Streamlit web app — Automatic License Plate Recognition & Fine Management
=============================================================================
A browser-based version of the GUI in src/gui.py, deployable for free on
Streamlit Community Cloud so you get a live, shareable link.

Run locally:
    streamlit run app.py

Deploy: push this repo to GitHub, then create a new app at
https://share.streamlit.io pointing at app.py — no server setup needed.
"""

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

from src import database as db
from src import ocr_engine
from src.genetic_algorithm import GAConfig, GeneticPlateLocator
from src.preprocessing import canny_edges, to_grayscale

st.set_page_config(page_title="ALPR — GA + Neural Network", page_icon="🚗", layout="wide")

db.init_db()

VIOLATIONS = ["Signal Jumping", "Speeding", "No Helmet", "Triple Riding", "No Seatbelt"]

st.title("🚗 Automatic License Plate Recognition & Fine Management System")
st.caption(
    "Genetic Algorithm for plate localization + Neural-Network OCR for character "
    "recognition — a browser port of our B.Tech ECE final year project."
)

if "detected_box" not in st.session_state:
    st.session_state.detected_box = None
    st.session_state.detected_plate = None
    st.session_state.fitness_info = None

col_img, col_form = st.columns([1, 1])

with col_img:
    uploaded = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        bgr = np.array(pil_img)[:, :, ::-1].copy()

        run_detect = st.button("🔍 Detect License Plate (GA + NN)", type="primary")

        if run_detect:
            with st.spinner("Running Genetic Algorithm to localize plate..."):
                gray = to_grayscale(bgr)
                edges = canny_edges(gray)
                locator = GeneticPlateLocator(image_shape=gray.shape, config=GAConfig())
                result = locator.run(gray, edges)
                x, y, w, h = result.best_box
                st.session_state.detected_box = (x, y, w, h)
                st.session_state.fitness_info = (result.best_fitness, result.generations_run)

            with st.spinner("Running Neural Network OCR..."):
                plate_region = bgr[y:y + h, x:x + w]
                try:
                    plate_text = ocr_engine.recognize(plate_region)
                except Exception as exc:
                    st.warning(f"OCR engine unavailable in this environment ({exc}).")
                    plate_text = None
                st.session_state.detected_plate = plate_text

        display_img = pil_img.copy()
        if st.session_state.detected_box:
            x, y, w, h = st.session_state.detected_box
            draw = ImageDraw.Draw(display_img)
            draw.rectangle([x, y, x + w, y + h], outline="yellow", width=4)

        st.image(display_img, caption="Uploaded / detected vehicle image", use_container_width=True)

        if st.session_state.fitness_info:
            fitness, gens = st.session_state.fitness_info
            st.info(f"GA fitness: **{fitness:.3f}**  |  Generations run: **{gens}**")
        if st.session_state.detected_plate:
            st.success(f"Detected plate: **{st.session_state.detected_plate}**")
        elif st.session_state.detected_box:
            st.warning("Plate region localized, but OCR could not read the text.")

with col_form:
    st.subheader("Vehicle & Fine Details")

    plate_value = st.session_state.detected_plate or ""
    record = db.get_vehicle(plate_value.upper()) if plate_value else None

    plate = st.text_input("License Plate", value=plate_value.upper())
    owner = st.text_input("Owner Name", value=record.owner_name if record else "")
    phone = st.text_input("Phone Number", value=record.phone_number if record else "")
    occupation = st.text_input("Occupation", value=record.occupation if record else "")
    area = st.text_input("Area", value=record.area if record else "")
    violation = st.selectbox("Violation", VIOLATIONS)
    fine_amount = st.number_input("Fine Amount", min_value=0, value=500, step=50)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("💰 Issue Fine"):
            if not plate:
                st.error("No license plate detected / entered!")
            else:
                db.issue_fine(
                    plate.upper(), violation, int(fine_amount),
                    {"owner_name": owner, "phone_number": phone,
                     "occupation": occupation, "area": area},
                )
                st.success("Fine issued successfully!")
                st.rerun()
    with c2:
        if st.button("✅ Pay Challan"):
            if not plate:
                st.error("No license plate detected / entered!")
            else:
                result = db.pay_fine(plate.upper())
                if result is None:
                    st.error("Vehicle not found!")
                else:
                    st.success("Challan paid successfully!")
                    st.rerun()

st.divider()
st.subheader("📋 All Vehicle Records")
records = db.get_all_vehicles()
if records:
    st.dataframe(
        [
            {
                "License Plate": r.license_plate,
                "Owner": r.owner_name,
                "Area": r.area,
                "Violations": r.violations,
                "Total Fines": r.total_fines,
                "Status": r.status,
            }
            for r in records
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.write("No records yet.")

st.caption(
    "⚠️ Note: on free hosting tiers, the database resets whenever the app restarts "
    "(e.g. after inactivity) — this is a live demo, not a production deployment."
)
