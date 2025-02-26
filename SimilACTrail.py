import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import io
from io import BytesIO
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests

# Author : Dr. Sk. Abdul Amin
# My [paper](https://www.scopus.com/authid/detail.uri?authorId=57190176332).

# For more details, please follow -
# J Med Chem. 2014;57(8):3186-204. doi: 10.1021/jm401411z.
# J Chem Inf Model. 2017;57(3):397-402. doi: 10.1021/acs.jcim.6b00776.
# J Chem Inf Model. 2008;48(3):646-58. doi: 10.1021/ci7004093.

logo_url = "https://github.com/Amincheminfom/Amincheminfom/raw/main/Cheminform_logo.jpg"

st.set_page_config(
    page_title="SimilACTrail_v1",
    layout="wide",
    page_icon=logo_url
)

st.sidebar.image(logo_url)

st.title("Structure-*SIMILA*rity *ACT*ivity *TRAIL*ing Map")
about_expander = st.expander("SimilACTrail_v1: *SIMILA*rity *ACT*ivity *TRAIL*ing", expanded=True)
with about_expander:
    st.markdown('''
    **SimilACTrail_v1** is a Python package for visualizing the Structure-Similarity Activity Trailing map.  

    It also identifies **Activity Cliffs (AC)** in the dataset.

    References:         
    1. *J Chem Inf Model*. 2008;48(3):646-58. DOI: [10.1021/ci7004093](https://doi.org/10.1021/ci7004093)
    2. *J Med Chem*. 2014;57(8):3186-204. DOI: [10.1021/jm401411z](https://doi.org/10.1021/jm401411z)  
    3. *J Chem Inf Model*. 2017;57(3):397-402. DOI: [10.1021/acs.jcim.6b00776](https://doi.org/10.1021/acs.jcim.6b00776) 
    ''')

#################################################################### Author : Dr. Sk. Abdul Amin
st.sidebar.subheader("Parameters")
#radius = st.sidebar.radio("Fingerprint Radius", ('2', '3', '4', '5'))
radius_map = {'ECFP4': 2, 'ECFP6': 3, 'ECFP8': 4, 'ECFP10': 5}
radius = radius_map[st.sidebar.radio("Fingerprint Name", ('ECFP4', 'ECFP6', 'ECFP8', 'ECFP10'))]
num_bits = st.sidebar.selectbox("Fingerprint Bit Size", options=[512, 1024, 2048, 4096], index=2)
similarity_threshold = st.sidebar.selectbox("Similarity Threshold", options=[0.7, 0.5, 0.6, 0.8, 0.9, 1.0])
activity_difference_threshold = st.sidebar.selectbox("Activity Difference Threshold", options=[0.5, 1, 1.5, 2, 2.5, 3],
                                                     index=1)

# Function to calculate Tanimoto similarity
def calculate_tanimoto_similarity(smiles1, smiles2, radius, nBits):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Function to categorize regions
def categorize_quadrant(similarity, activity_diff, similarity_threshold, activity_difference_threshold):
    if similarity >= similarity_threshold and activity_diff >= activity_difference_threshold:
        return 'Activity Cliffs'
    elif similarity < similarity_threshold and activity_diff < activity_difference_threshold:
        return 'Scaffold Hops'
    elif similarity >= similarity_threshold and activity_diff < activity_difference_threshold:
        return 'Smooth SAR Zones'
    else:
        return 'Non-descript Zones'


# Function to calculate activity cliffs
def calculate_activity_cliffs(df, radius, similarity_threshold, activity_difference_threshold):
    results = []
    for (index1, row1), (index2, row2) in combinations(df.iterrows(), 2):
        smiles1, activity1 = row1['Smiles'], row1['pIC50']
        smiles2, activity2 = row2['Smiles'], row2['pIC50']
        similarity = calculate_tanimoto_similarity(smiles1, smiles2, int(radius), num_bits)
        activity_diff = abs(activity1 - activity2)
        quadrant = categorize_quadrant(similarity, activity_diff, similarity_threshold, activity_difference_threshold)
        results.append((row1['Molecule ChEMBL ID'], row2['Molecule ChEMBL ID'], similarity, activity_diff, quadrant))

    cliffs_df = pd.DataFrame(results,
                             columns=['Molecule ChEMBL ID1', 'Molecule ChEMBL ID2', 'Similarity', 'Activity_Difference',
                                      'Quadrant'])
    return cliffs_df


# Function to plot SimilACTrail map
def plot_amin_map(cliffs_df, logo_url):
    colors = {'Activity Cliffs': 'red', 'Scaffold Hops': 'Purple', 'Smooth SAR Zones': 'green',
              'Non-descript Zones': 'blue'}
    fig, ax = plt.subplots(figsize=(10, 6))

    for quadrant, color in colors.items():
        subset = cliffs_df[cliffs_df['Quadrant'] == quadrant]
        ax.scatter(subset['Similarity'], subset['Activity_Difference'], c=color, label=quadrant, alpha=0.6)

    ax.axvline(similarity_threshold, linestyle='dotted', color='black')
    ax.axhline(activity_difference_threshold, linestyle='dotted', color='black')

    ax.set_xlabel('Tanimoto Similarity')
    ax.set_ylabel('Activity Difference')
    radius_str = {2: 'ECFP4', 3: 'ECFP6', 4: 'ECFP8', 5: 'ECFP10'}[radius]
    ax.set_title(f'SimilACTrail Map (Fingerprint : {radius_str})')
    ax.legend(loc='best')
    ax.grid(False)

    # Author: Dr. Sk. Abdul Amin
    try:
        response = requests.get(logo_url)
        response.raise_for_status()
        logo_img = plt.imread(BytesIO(response.content), format="JPG")

        imagebox = OffsetImage(logo_img, zoom=0.04)
        ab = AnnotationBbox(imagebox, (0.23
                                       , -0.082), xycoords='axes fraction', frameon=False)
        ax.add_artist(ab)
    except Exception as e:
        st.warning(f"Failed to load logo: {e}")

    st.pyplot(fig)
    return fig

#################################################################### Author : Dr. Sk. Abdul Amin

st.subheader("Upload Dataset")
dataset_choice = st.radio(
    "Choose a dataset:",
    ("Upload CSV File", "Sample Dataset"),
    horizontal=True,
)

#################################################################### Author : Dr. Sk. Abdul Amin
# https://github.com/Amincheminfom/SimilACTrail_v1/blob/main/SimilACTrail_sample.csv
sample_file_url = "https://github.com/Amincheminfom/SimilACTrail_v1/raw/main/SimilACTrail_sample.csv"

# Ensure session state tracks dataset selection
st.session_state["dataset_choice"] = dataset_choice

uploaded_file = None

if dataset_choice == "Sample Dataset":
    try:
        response = requests.get(sample_file_url)
        if response.status_code == 200:
            uploaded_file = io.StringIO(response.text)
        else:
            st.error(f"Failed to load sample dataset (HTTP {response.status_code}).")
            st.stop()
    except Exception as e:
        st.error(f"Error fetching sample dataset: {e}")
        st.stop()
else:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

def read_csv_with_flexible_delimiter(file):
    """Read CSV file with support for `,` or `;` delimiters."""
    try:
        return pd.read_csv(file)  # Try with `,`
    except pd.errors.ParserError:
        return pd.read_csv(file, sep=';')  # Fallback to `;`

if uploaded_file:
    try:
        data = read_csv_with_flexible_delimiter(uploaded_file)

        if not data.empty:
            st.subheader("Select Columns for Analysis")

            id_col = st.selectbox("Select the Molecule ID column:", data.columns)
            smiles_col = st.selectbox("Select the SMILES column:", data.columns)
            activity_col = st.selectbox("Select the Activity column (e.g., pIC50):", data.columns)

            # Author: Dr. Sk. Abdul Amin
            if id_col and smiles_col and activity_col:
                run_button = st.button("Run SimilACTrail_v1")

                if run_button:
                    df = data[[id_col, smiles_col, activity_col]].rename(
                        columns={id_col: "Molecule ChEMBL ID", smiles_col: "Smiles", activity_col: "pIC50"}
                    )

                    cliffs_df = calculate_activity_cliffs(df, radius, similarity_threshold, activity_difference_threshold)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Dataset Preview")
                        st.dataframe(df)

                    with col2:
                        st.write("### Calculated Data")
                        st.dataframe(cliffs_df)
                        csv = cliffs_df.to_csv(index=False).encode('utf-8')
                        st.download_button(label="Download CSV", data=csv, file_name='activity_cliffs.csv',
                                           mime='text/csv')

                    st.subheader("SimilACTrail Map")
                    fig = plot_amin_map(cliffs_df, logo_url)
                    plot_buffer = io.BytesIO()
                    fig.savefig(plot_buffer, format="png", bbox_inches="tight")
                    plot_buffer.seek(0)

                    st.download_button(label="Download Plot", data=plot_buffer, file_name="SimilACTrail_Map.png",
                                       mime="image/png")

        else:
            st.error("The uploaded file is empty or could not be read.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

#################################################################### Author : Dr. Sk. Abdul Amin
# Contact Information
contacts = st.expander("Contact", expanded=False)
with contacts:
    st.write('''
             #### Report an Issue 
             You are welcome to report a bug or contribute to the web 
             application by filing an issue on [Github](https://github.com/Amincheminfom).

             #### Contact
             For any questions, you can contact us via email:
             - [Dr. Sk. Abdul Amin](mailto:pharmacist.amin@gmail.com)
             ''')