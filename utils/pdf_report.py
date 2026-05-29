"""
Threat report generator for SecureLens.
Builds a formal PDF for one flagged employee suitable for filing by a security officer.
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

DARK_BG    = colors.HexColor('#0f172a')
RED        = colors.HexColor('#ef4444')
GREEN      = colors.HexColor('#22c55e')
ORANGE     = colors.HexColor('#f97316')
SLATE      = colors.HexColor('#64748b')
LIGHT_GREY = colors.HexColor('#f1f5f9')
MID_GREY   = colors.HexColor('#e2e8f0')
WHITE      = colors.white
BLACK      = colors.black


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportTitle',   fontName='Helvetica-Bold', fontSize=18,
                               textColor=DARK_BG, spaceAfter=4, alignment=TA_CENTER))
    styles.add(ParagraphStyle('ReportSub',     fontName='Helvetica',      fontSize=10,
                               textColor=SLATE,   spaceAfter=2, alignment=TA_CENTER))
    styles.add(ParagraphStyle('SectionHeader', fontName='Helvetica-Bold', fontSize=11,
                               textColor=DARK_BG, spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle('Body9',         fontName='Helvetica',      fontSize=9,
                               textColor=BLACK,   spaceAfter=4, leading=14))
    styles.add(ParagraphStyle('SmallGrey',     fontName='Helvetica',      fontSize=8,
                               textColor=SLATE,   spaceAfter=2))
    styles.add(ParagraphStyle('VerdictText',   fontName='Helvetica-Bold', fontSize=14,
                               textColor=WHITE,   alignment=TA_CENTER))
    return styles


def generate_pdf_report(record_dict, result, employee_id=None):
    """
    Generate a PDF threat report for one employee prediction.

    Args:
        record_dict: raw feature values dict
        result: dict with label, confidence, risk_label, explanation, feature_importance (DataFrame)
    Returns:
        bytes: PDF content
    """
    import pandas as pd

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)

    styles  = build_styles()
    story   = []
    now     = datetime.now().strftime('%d %B %Y  %H:%M')
    label   = result.get('label', 'UNKNOWN')
    conf    = result.get('confidence', 0)
    risk    = result.get('risk_label', '')
    expl    = result.get('explanation', '')
    fi_df   = result.get('feature_importance', pd.DataFrame())
    if isinstance(fi_df, list):
        fi_df = pd.DataFrame(fi_df)

    # ── Header ─────────────────────────────────────────────────────────────────
    story.append(Paragraph("INSIDER THREAT DETECTION SYSTEM", styles['ReportTitle']))
    story.append(Paragraph("Behavioural Risk Assessment Report", styles['ReportSub']))
    story.append(Paragraph(f"Generated: {now}",
                           styles['SmallGrey']))
    story.append(HRFlowable(width="100%", thickness=1, color=MID_GREY, spaceAfter=8))

    # ── Verdict banner ──────────────────────────────────────────────────────────
    bg_col = RED if label == 'MALICIOUS' else GREEN
    vt = Table([[Paragraph(
        f"CLASSIFICATION: {label}  |  {risk}  |  Confidence: {conf*100:.1f}%",
        styles['VerdictText']
    )]], colWidths=[170*mm])
    vt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), bg_col),
        ('TOPPADDING',    (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(vt)
    story.append(Spacer(1, 10))

    # ── Employee profile ────────────────────────────────────────────────────────
    story.append(Paragraph("EMPLOYEE PROFILE", styles['SectionHeader']))
    profile_rows = [
        ('Employee ID',         employee_id or 'N/A'),
        ('Department',          record_dict.get('employee_department', 'N/A')),
        ('Campus',              record_dict.get('employee_campus', 'N/A')),
        ('Position',            record_dict.get('employee_position', 'N/A')),
        ('Seniority',           f"{record_dict.get('employee_seniority_years', 0)} years"),
        ('Security Clearance',  f"Level {record_dict.get('employee_classification', 0)}"),
        ('Contractor',          'Yes' if record_dict.get('is_contractor') else 'No'),
        ('Foreign Citizenship', 'Yes' if record_dict.get('has_foreign_citizenship') else 'No'),
        ('Criminal Record',     'Yes' if record_dict.get('has_criminal_record') else 'No'),
        ('Country of Origin',   record_dict.get('employee_origin_country', 'N/A')),
    ]
    pt = Table(
        [[Paragraph(k, styles['SmallGrey']), Paragraph(str(v), styles['Body9'])]
         for k, v in profile_rows],
        colWidths=[55*mm, 115*mm]
    )
    pt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (0,-1), LIGHT_GREY),
        ('GRID',          (0,0), (-1,-1), 0.5, MID_GREY),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 6),
    ]))
    story.append(pt)

    # ── Behavioural indicators ──────────────────────────────────────────────────
    story.append(Paragraph("BEHAVIOURAL INDICATORS", styles['SectionHeader']))
    behav_rows = [
        ('Total Pages Printed',        record_dict.get('total_printed_pages', 0)),
        ('Off-Hours Printing',          record_dict.get('num_printed_pages_off_hours', 0)),
        ('Files Burned to Disk',        record_dict.get('total_files_burned', 0)),
        ('Files Burned (Other Accts)',  record_dict.get('burned_from_other', 0)),
        ('Currently Abroad',            'Yes' if record_dict.get('is_abroad') else 'No'),
        ('Trip Day Number',             record_dict.get('trip_day_number', 0)),
        ('Destination Hostility Level', record_dict.get('hostility_country_level', 0)),
        ('Building Entries',            record_dict.get('num_entries', 0)),
        ('Unique Campuses Accessed',    record_dict.get('num_unique_campus', 0)),
        ('Late Exit Detected',          'Yes' if record_dict.get('late_exit_flag') else 'No'),
        ('Weekend Entry',               'Yes' if record_dict.get('entry_during_weekend') else 'No'),
    ]
    bt = Table(
        [[Paragraph(k, styles['SmallGrey']), Paragraph(str(v), styles['Body9'])]
         for k, v in behav_rows],
        colWidths=[75*mm, 95*mm]
    )
    bt.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (0,-1), LIGHT_GREY),
        ('GRID',          (0,0), (-1,-1), 0.5, MID_GREY),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 6),
    ]))
    story.append(bt)

    # ── SHAP risk indicators ────────────────────────────────────────────────────
    if not fi_df.empty:
        story.append(Paragraph("TOP RISK INDICATORS (SHAP Analysis)", styles['SectionHeader']))
        header = [
            [Paragraph('Feature', styles['SmallGrey']),
             Paragraph('SHAP Value', styles['SmallGrey']),
             Paragraph('Effect', styles['SmallGrey'])]
        ]
        data_rows = []
        for _, row in fi_df.head(8).iterrows():
            is_risk = row.get('direction', 'risk') == 'risk'
            effect  = '↑ Increases Risk' if is_risk else '↓ Reduces Risk'
            shap_v  = row.get('shap_value', row.get('shap_value', 0))
            feat    = row.get('display_name', row.get('feature', ''))
            data_rows.append([
                Paragraph(str(feat), styles['Body9']),
                Paragraph(f"{float(shap_v):.4f}", styles['Body9']),
                Paragraph(effect, styles['Body9']),
            ])
        rt = Table(header + data_rows, colWidths=[85*mm, 35*mm, 50*mm])
        rt.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0), DARK_BG),
            ('TEXTCOLOR',     (0,0), (-1,0), WHITE),
            ('GRID',          (0,0), (-1,-1), 0.5, MID_GREY),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GREY]),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING',   (0,0), (-1,-1), 6),
        ]))
        story.append(rt)

    # ── Analyst summary ─────────────────────────────────────────────────────────
    story.append(Paragraph("ANALYST SUMMARY", styles['SectionHeader']))
    story.append(Paragraph(expl, styles['Body9']))
    story.append(Spacer(1, 8))

    # ── Footer ──────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GREY, spaceBefore=8))
    story.append(Paragraph(
        "Model: Random Forest (200 trees)  |  Explainability: SHAP TreeExplainer  |  "
        "Training data: 118,614 records",
        styles['SmallGrey']
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
