import streamlit as st
import pandas as pd

st.set_page_config(page_title= "PV Cap Test", page_icon="icon.jpg", layout="centered")
st.logo("long_logo.jpg", icon_image="icon.jpg")
st.sidebar.subheader(" ")
st.title("PV Cap Test")

tab1, tab2, tab3 = st.tabs(['Measured', 'PVSyst', 'Report'])

form = tab1.form("input form")
form.header("Inputs:")

uploaded_files = form.file_uploader("Upload MET Data", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    form.write("filename:", uploaded_file.name)
    form.write(bytes_data)

uploaded_files = form.file_uploader("Upload Inverter Data", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    form.write("filename:", uploaded_file.name)
    form.write(bytes_data)

uploaded_files = form.file_uploader("Upload Meter Data", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    form.write("filename:", uploaded_file.name)
    form.write(bytes_data)

uploaded_files = form.file_uploader("Upload Column Groups", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    form.write("filename:", uploaded_file.name)
    form.write(bytes_data)

form.form_submit_button("Submit Files")