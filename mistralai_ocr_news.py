import os
from mistralai import Mistral
from pathlib import Path
from mistralai import DocumentURLChunk
import streamlit as st

# Streamlit App Title
st.set_page_config(page_title="Mistral OCR App", layout="wide")
st.title("📄 Mistral OCR - Extract Text from PDFs")

# Set the API key manually
api_key = os.getenv("MISTRAL_API_KEY", st.secrets["auth_token"])  # Replace with actual key

# Initialize the Mistral client
client = Mistral(api_key=api_key)

# File Upload Section
uploaded_file = st.file_uploader("📂 Upload a PDF File", type=["pdf"])

if uploaded_file:
    st.success(f"✅ Uploaded: {uploaded_file.name}")

    # Ensure text is not reprocessed unnecessarily
    if "extracted_text" not in st.session_state or st.session_state["last_uploaded"] != uploaded_file.name:
        # Save the uploaded file temporarily
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Process the PDF
        pdf_file = Path(temp_pdf_path)

        with st.spinner("🔄 Processing PDF..."):
            uploaded_file_mistral = client.files.upload(
                file={
                    "file_name": pdf_file.stem,
                    "content": pdf_file.read_bytes(),
                },
                purpose="ocr",
            )

            # Generate a signed URL
            signed_url = client.files.get_signed_url(file_id=uploaded_file_mistral.id, expiry=5)

            # Process with OCR
            pdf_response = client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model="mistral-ocr-latest",
                include_image_base64=False
            )

        # Extract text
        if hasattr(pdf_response, "pages") and pdf_response.pages:
            extracted_text = "\n\n".join(
                [f"### Page {i+1}\n{page.markdown}" for i, page in enumerate(pdf_response.pages) if hasattr(page, "markdown")]
            )
            st.session_state["extracted_text"] = extracted_text
            st.session_state["last_uploaded"] = uploaded_file.name
        else:
            st.session_state["extracted_text"] = "❌ No text extracted."

        # Cleanup temp file
        os.remove(temp_pdf_path)

    # Display Extracted Text
    st.subheader("📜 Extracted Text")
    st.text_area("OCR Output", st.session_state["extracted_text"], height=400)

    # Download Button (No Reprocessing)
    st.download_button("📥 Download Text", st.session_state["extracted_text"], file_name="extracted_text.txt", mime="text/plain")