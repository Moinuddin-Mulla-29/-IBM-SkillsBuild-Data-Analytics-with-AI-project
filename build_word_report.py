import os
import json
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Set shading/background color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set inner cell padding in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def create_report():
    doc = Document()
    
    # Page Margins (1 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styling colors
    PRIMARY_COLOR = RGBColor(27, 54, 93)   # #1B365D Dark Navy Blue
    SECONDARY_COLOR = RGBColor(43, 92, 143) # #2B5C8F Steel Blue
    DARK_NEUTRAL = RGBColor(51, 51, 51)    # #333333 Charcoal

    # Base Normal Style
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = DARK_NEUTRAL
    style_normal.paragraph_format.line_spacing = 1.15
    style_normal.paragraph_format.space_after = Pt(6)

    # ---------------------------------------------------------
    # COVER / HEADER TITLE
    # ---------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("PROJECT REPORT\nHouse Price Prediction Using Machine Learning")
    title_run.font.name = 'Calibri'
    title_run.font.size = Pt(24)
    title_run.font.bold = True
    title_run.font.color.rgb = PRIMARY_COLOR
    title_p.paragraph_format.space_after = Pt(4)

    subtitle_p = doc.add_paragraph()
    subtitle_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle_run = subtitle_p.add_run("Comprehensive Technical Documentation & Model Benchmark Evaluation Report")
    subtitle_run.font.name = 'Calibri'
    subtitle_run.font.size = Pt(13)
    subtitle_run.font.italic = True
    subtitle_run.font.color.rgb = SECONDARY_COLOR
    subtitle_p.paragraph_format.space_after = Pt(20)

    # Metadata Box Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Author / Student Name:", "Moinuddin"),
        ("Project Name:", "House Price Prediction System"),
        ("Dataset:", "House Price Prediction Dataset.csv (2,000 Records)"),
        ("Technologies:", "Python 3.13, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn")
    ]
    for row_idx, (k, v) in enumerate(meta_data):
        cell_k = meta_table.cell(row_idx, 0)
        cell_v = meta_table.cell(row_idx, 1)
        
        cell_k.width = Inches(2.2)
        cell_v.width = Inches(4.3)
        
        p_k = cell_k.paragraphs[0]
        r_k = p_k.add_run(k)
        r_k.bold = True
        r_k.font.color.rgb = PRIMARY_COLOR
        
        p_v = cell_v.paragraphs[0]
        p_v.add_run(v)
        
        set_cell_background(cell_k, "F0F4F8")
        set_cell_background(cell_v, "F9FAFC")
        set_cell_margins(cell_k, 80, 80, 120, 120)
        set_cell_margins(cell_v, 80, 80, 120, 120)
        
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    def add_heading_1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = PRIMARY_COLOR
        return h

    def add_heading_2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = 'Calibri'
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = SECONDARY_COLOR
        return h

    # ---------------------------------------------------------
    # 1. EXECUTIVE SUMMARY
    # ---------------------------------------------------------
    add_heading_1("1. Executive Summary")
    doc.add_paragraph(
        "Accurate valuation of residential real estate is essential for homebuyers, sellers, investors, "
        "and financial institutions. This project delivers an end-to-end Machine Learning pipeline designed "
        "to predict housing prices using key structural and geographic property parameters. "
        "Utilizing a dataset of 2,000 property records, multiple predictive models were developed, trained, "
        "and benchmarked, including Linear Regression, Ridge Regression, Lasso Regression, Decision Tree Regressor, "
        "Random Forest Regressor, and Gradient Boosting Regressor."
    )
    doc.add_paragraph(
        "The project incorporates robust data preprocessing using Scikit-Learn Pipelines and ColumnTransformers, "
        "preventing data leakage and establishing reproducible feature transformations (Standard Scaling for numerical variables "
        "and One-Hot Encoding for categorical factors). Comprehensive diagnostics, including R² score, Mean Absolute Error (MAE), "
        "and Root Mean Squared Error (RMSE), were computed alongside feature importance analysis."
    )

    # ---------------------------------------------------------
    # 2. PROBLEM STATEMENT & OBJECTIVES
    # ---------------------------------------------------------
    add_heading_1("2. Problem Statement & Project Objectives")
    add_heading_2("2.1 Problem Statement")
    doc.add_paragraph(
        "Real estate pricing is influenced by a complex interplay of physical specifications (square footage, bedrooms, bathrooms, floors) "
        "and qualitative attributes (location, overall condition, and garage availability). Traditional manual appraisal models "
        "are often subjective, time-consuming, and prone to inconsistency. Automated Valuation Models (AVMs) leveraging Machine Learning "
        "provide data-driven, scalable, and objective price estimation."
    )
    
    add_heading_2("2.2 Key Objectives")
    objectives = [
        "Perform Exploratory Data Analysis (EDA) to uncover structural relationships, feature correlations, and data distributions.",
        "Implement a modular data preprocessing pipeline handling numerical standardization and categorical encoding.",
        "Train and benchmark six distinct machine learning algorithms for regression analysis.",
        "Evaluate predictive performance using standard industry metrics: R² Score, MAE ($), and RMSE ($).",
        "Determine the relative feature importance influencing property values to provide actionable business insights."
    ]
    for obj in objectives:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(obj)

    # ---------------------------------------------------------
    # 3. DATASET DESCRIPTION
    # ---------------------------------------------------------
    add_heading_1("3. Dataset Overview & Data Dictionary")
    doc.add_paragraph(
        "The project uses `House Price Prediction Dataset.csv`, consisting of 2,000 observations and 10 features. "
        "There are no missing or null values in the dataset, ensuring a clean baseline for model development."
    )
    
    # Data Dictionary Table
    table = doc.add_table(rows=1, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.rows[0].cells
    headers = ["Feature Name", "Data Type", "Role / Type", "Description"]
    widths = [Inches(1.5), Inches(1.1), Inches(1.2), Inches(2.7)]
    
    for i, title in enumerate(headers):
        hdr_cells[i].width = widths[i]
        p = hdr_cells[i].paragraphs[0]
        r = p.add_run(title)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr_cells[i], "1B365D")
        set_cell_margins(hdr_cells[i], 100, 100, 120, 120)

    data_dict = [
        ("Id", "Integer", "Identifier", "Unique row identification number (dropped during ML training)."),
        ("Area", "Integer", "Numerical Feature", "Property living space area measured in square feet (501 - 4,999 sq ft)."),
        ("Bedrooms", "Integer", "Numerical Feature", "Total count of bedrooms in the property (1 to 5)."),
        ("Bathrooms", "Integer", "Numerical Feature", "Total count of bathrooms (1 to 4)."),
        ("Floors", "Integer", "Numerical Feature", "Number of building floors (1 to 3)."),
        ("YearBuilt", "Integer", "Numerical Feature", "Construction year of the house (1900 to 2023)."),
        ("Location", "String", "Categorical Feature", "Geographic classification (Downtown, Suburban, Urban, Rural)."),
        ("Condition", "String", "Categorical Feature", "Qualitative rating (Poor, Fair, Good, Excellent)."),
        ("Garage", "String", "Binary Feature", "Indicates garage presence ('Yes' or 'No')."),
        ("Price", "Integer", "Target Variable", "Sale price of the house in USD ($50,005 - $999,656).")
    ]

    for row_idx, data in enumerate(data_dict):
        row_cells = table.add_row().cells
        fill_color = "F9FAFC" if row_idx % 2 == 0 else "FFFFFF"
        for col_idx, text in enumerate(data):
            row_cells[col_idx].width = widths[col_idx]
            p = row_cells[col_idx].paragraphs[0]
            p.add_run(text)
            set_cell_background(row_cells[col_idx], fill_color)
            set_cell_margins(row_cells[col_idx], 80, 80, 100, 100)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ---------------------------------------------------------
    # 4. EXPLORATORY DATA ANALYSIS (EDA)
    # ---------------------------------------------------------
    add_heading_1("4. Exploratory Data Analysis (EDA)")
    doc.add_paragraph(
        "Statistical visualizations were generated to analyze property distribution, price variation across locations, "
        "and feature correlations."
    )

    if os.path.exists("figures/eda_price_dist.png"):
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
        p_img1 = doc.add_paragraph()
        p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img1.add_run().add_picture("figures/eda_price_dist.png", width=Inches(5.8))
        caption1 = doc.add_paragraph()
        caption1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap1 = caption1.add_run("Figure 1: Distribution of House Prices ($)")
        r_cap1.font.size = Pt(9.5)
        r_cap1.font.italic = True
        caption1.paragraph_format.space_after = Pt(12)

    if os.path.exists("figures/eda_area_price.png"):
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.add_run().add_picture("figures/eda_area_price.png", width=Inches(5.8))
        caption2 = doc.add_paragraph()
        caption2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap2 = caption2.add_run("Figure 2: House Price vs. Property Area (sq ft) Categorized by Location")
        r_cap2.font.size = Pt(9.5)
        r_cap2.font.italic = True
        caption2.paragraph_format.space_after = Pt(12)

    if os.path.exists("figures/eda_correlation.png"):
        p_img3 = doc.add_paragraph()
        p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img3.add_run().add_picture("figures/eda_correlation.png", width=Inches(5.2))
        caption3 = doc.add_paragraph()
        caption3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap3 = caption3.add_run("Figure 3: Feature Correlation Heatmap")
        r_cap3.font.size = Pt(9.5)
        r_cap3.font.italic = True
        caption3.paragraph_format.space_after = Pt(12)

    # ---------------------------------------------------------
    # 5. METHODOLOGY & MACHINE LEARNING PIPELINE
    # ---------------------------------------------------------
    add_heading_1("5. Machine Learning Methodology")
    doc.add_paragraph(
        "To evaluate performance rigorously, the dataset was split into 80% Training set (1,600 samples) "
        "and 20% Test set (400 samples) using a fixed random seed (`random_state=42`)."
    )
    
    add_heading_2("5.1 Data Preprocessing Architecture")
    doc.add_paragraph(
        "Preprocessing was constructed using Scikit-Learn's `ColumnTransformer` to prevent data leakage between train and test sets:"
    )
    prep_steps = [
        "Numerical Features ('Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt'): Standardized using StandardScaler (zero mean, unit variance).",
        "Categorical Features ('Location', 'Condition', 'Garage'): One-hot encoded using OneHotEncoder(drop='first') to mitigate multicollinearity."
    ]
    for step in prep_steps:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(step)

    # ---------------------------------------------------------
    # 6. MODEL EVALUATION & RESULTS
    # ---------------------------------------------------------
    add_heading_1("6. Model Performance & Results")
    doc.add_paragraph(
        "Six Machine Learning regression models were trained and benchmarked on the test dataset. "
        "The quantitative evaluation metrics are summarized below:"
    )

    # Load results JSON if present
    results_path = "figures/model_results.json"
    results_dict = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results_dict = json.load(f)

    # Results Table
    res_table = doc.add_table(rows=1, cols=5)
    res_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_res = res_table.rows[0].cells
    res_headers = ["Algorithm Name", "Train R²", "Test R² Score", "MAE ($)", "RMSE ($)"]
    res_widths = [Inches(1.8), Inches(1.1), Inches(1.1), Inches(1.2), Inches(1.3)]
    
    for i, title in enumerate(res_headers):
        hdr_res[i].width = res_widths[i]
        p = hdr_res[i].paragraphs[0]
        r = p.add_run(title)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(hdr_res[i], "1B365D")
        set_cell_margins(hdr_res[i], 100, 100, 100, 100)

    for idx, (m_name, metrics) in enumerate(results_dict.items()):
        row_c = res_table.add_row().cells
        bg = "F9FAFC" if idx % 2 == 0 else "FFFFFF"
        row_data = [
            m_name,
            str(metrics.get('Train R2', 'N/A')),
            str(metrics.get('Test R2', 'N/A')),
            f"${metrics.get('MAE', 0):,.2f}",
            f"${metrics.get('RMSE', 0):,.2f}"
        ]
        for c_idx, val in enumerate(row_data):
            row_c[c_idx].width = res_widths[c_idx]
            p = row_c[c_idx].paragraphs[0]
            if c_idx == 0:
                p.add_run(val).bold = True
            else:
                p.add_run(val)
            set_cell_background(row_c[c_idx], bg)
            set_cell_margins(row_c[c_idx], 80, 80, 100, 100)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    if os.path.exists("figures/model_comparison.png"):
        p_img4 = doc.add_paragraph()
        p_img4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img4.add_run().add_picture("figures/model_comparison.png", width=Inches(5.8))
        caption4 = doc.add_paragraph()
        caption4.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap4 = caption4.add_run("Figure 4: Machine Learning Model Benchmark Comparison")
        r_cap4.font.size = Pt(9.5)
        r_cap4.font.italic = True
        caption4.paragraph_format.space_after = Pt(12)

    if os.path.exists("figures/feature_importance.png"):
        add_heading_2("6.1 Feature Importance Analysis")
        p_img5 = doc.add_paragraph()
        p_img5.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img5.add_run().add_picture("figures/feature_importance.png", width=Inches(5.5))
        caption5 = doc.add_paragraph()
        caption5.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap5 = caption5.add_run("Figure 5: Random Forest Relative Feature Importance")
        r_cap5.font.size = Pt(9.5)
        r_cap5.font.italic = True
        caption5.paragraph_format.space_after = Pt(12)

    # ---------------------------------------------------------
    # 7. CONCLUSION & FUTURE WORK
    # ---------------------------------------------------------
    add_heading_1("7. Conclusion & Future Recommendations")
    doc.add_paragraph(
        "This project successfully designed and executed an end-to-end Machine Learning solution for House Price Prediction. "
        "By structuring the data transformation pipeline into reusable Scikit-Learn components, clean modular code execution was guaranteed."
    )
    add_heading_2("7.1 Key Findings")
    findings = [
        "Synthetic Data Uniformity: Statistical evaluation revealed near-zero linear correlation between input features and target prices in this synthetic benchmark dataset.",
        "Overfitting in Tree Models: High depth decision trees and un-tuned random forests exhibited high variance between train and test R² scores, highlighting the necessity of hyperparameter pruning.",
        "Pipeline Standardization: Utilizing Pipeline and ColumnTransformer ensured zero data leakage during evaluation."
    ]
    for f in findings:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(f)

    add_heading_2("7.2 Recommendations for Future Scope")
    future_recs = [
        "Integration of Real-World Real Estate Data: Collect enriched real-world dataset attributes such as zip code median income, school ratings, and neighborhood crime statistics.",
        "Advanced Hyperparameter Tuning: Implement Bayesian Optimization (Optuna) for fine-tuning XGBoost, LightGBM, and Random Forest estimators.",
        "Deployment & API Integration: Package the trained pipeline into a FastAPI microservice with a web dashboard UI."
    ]
    for rec in future_recs:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(rec)

    # Save document
    output_filename = "Moinuddin_ProjectReport.docx"
    doc.save(output_filename)
    print(f"Project report '{output_filename}' generated successfully!")

if __name__ == '__main__':
    create_report()
