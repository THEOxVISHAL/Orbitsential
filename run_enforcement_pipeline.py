#!/usr/bin/env python
"""
Autonomous Enforcement & Intelligence Pipeline for Lucknow Geospatial AI.
Powered by Meta AI Open Foundation Stack:
  1. Meta SAM 2 (Segment Anything 2 - FAIR) for sub-pixel cadastral parcel boundary refinement.
  2. Meta Llama 3.2 Vision (11B/90B) for multimodal satellite visual forensics & bilingual statutory notices.
  3. Meta MMS (Massively Multilingual Speech) for tactical walkie-talkie field police radio broadcasts.
  4. Instant zero-setup export to Primary Dashboard (index.html).
"""
import os
import sys
import json
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.llama_inspector import inspect_alert_with_llama32
from src.sam2_refiner import refine_encroachment_with_sam2, generate_sam2_vector_geojson
from src.dinov2_change_detector import detect_structural_anomaly_dinov2
from src.faiss_memory_bank import query_faiss_incident_memory
from src.nllb_translator import translate_notice_to_dialects
from src.pdf_dossier_generator import generate_official_enforcement_pdf
from src.tactical_radio import generate_tactical_radio_dispatch


def run_pipeline(top_n_alerts: int = 5):
    print("=" * 75)
    print("🛰️  LUCKNOW GEOSPATIAL AI : AUTONOMOUS META AI ENFORCEMENT PIPELINE")
    print("=" * 75)
    print("AI Foundation Stack:")
    print("  • Vision Reasoner: Meta Llama 3.2 Vision (11B/90B) + Llama Guard 3")
    print("  • Parcel Segmentation: Meta SAM 2 (Segment Anything Model 2 - FAIR)")
    print("  • Tactical Radio Engine: Meta MMS (Massively Multilingual Speech)")
    print("-" * 75)

    alerts_csv = ROOT_DIR / "monitoring_results" / "encroachment_alerts_latest.csv"
    if not alerts_csv.exists():
        print(f"Error: {alerts_csv} not found.")
        return

    df = pd.read_csv(alerts_csv)
    print(f"Loaded {len(df)} total detected encroachment alerts.")

    # Filter for Critical violations (Gomti Floodplain, Kukrail Forest)
    critical_alerts = df[df["severity"] == "CRITICAL"].head(top_n_alerts)
    print(f"Processing Top {len(critical_alerts)} Critical Eco-Zone Breaches...\n")

    chips_dir = ROOT_DIR / "monitoring_results" / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = ROOT_DIR / "monitoring_results" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    dossiers = []

    # 1. Generate Global Meta SAM 2 Refined Vector Polygons
    print("🎯 [STEP 1/3] Executing Meta SAM 2 Sub-Pixel Boundary Delineation...")
    sam2_geojson_path = ROOT_DIR / "monitoring_results" / "sam2_refined_encroachments.geojson"
    generate_sam2_vector_geojson(critical_alerts, chips_dir, sam2_geojson_path)
    print(f"  ✓ Meta SAM 2 sub-pixel polygons saved to: {sam2_geojson_path.name}\n")

    # 2. Process each critical site
    print("🧠 [STEP 2/3] Running Meta Llama 3.2 Vision Forensics & Meta MMS Radio Dispatch...")
    for idx, (_, row) in enumerate(critical_alerts.iterrows(), 1):
        site_id = str(row["site_id"])
        zone = str(row["zone_name"])
        area_sqm = float(row["area_sqm"])
        area_bigha = float(row.get("area_bigha", area_sqm / 2529.0))
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        prev_lc = str(row.get("previous_landcover", "Agriculture"))

        print(f"[{idx}/{len(critical_alerts)}] Auditing Site {site_id} in {zone} ({area_sqm:,.0f} m²)...")

        # Chip paths
        chip_path = chips_dir / f"{site_id}_satellite_chip.png"
        comp_path = chips_dir / f"{site_id}_sam2_comparison.png"

        # A. Meta SAM 2 Refinement on individual chip
        mask, poly_pts, sam2_metrics = refine_encroachment_with_sam2(
            chip_path=chip_path,
            center_prompt=(256, 256),
            output_comparison_path=comp_path
        )

        # B. Meta DINOv2 Zero-Shot ViT Anomaly Heatmap
        dinov2_heatmap_path = chips_dir / f"{site_id}_dinov2_heatmap.png"
        _, dinov2_metrics = detect_structural_anomaly_dinov2(
            chip_path=chip_path,
            output_heatmap_path=dinov2_heatmap_path
        )

        # C. Meta FAISS Vector Incident Memory Retrieval
        faiss_matches = query_faiss_incident_memory(
            zone_name=zone,
            top_k=2
        )

        # D. Meta Llama 3.2 Vision Multimodal Reasoning & Statutory Notices (Bilingual)
        intel = inspect_alert_with_llama32(
            site_id=site_id,
            zone_name=zone,
            area_sqm=area_sqm,
            area_bigha=area_bigha,
            lat=lat,
            lon=lon,
            previous_landcover=prev_lc,
            save_chip_dir=chips_dir
        )

        # E. Meta NLLB-200 Regional Dialect Translation (Awadhi & Bhojpuri)
        dialects = translate_notice_to_dialects(
            site_id=site_id,
            zone_name=zone,
            area_bigha=area_bigha,
            lat=lat,
            lon=lon
        )

        # F. Meta MMS Tactical Police Radio Voice Dispatch
        radio_script = intel.get("tactical_radio_script", f"Alert: Encroachment at {site_id} in {zone}.")
        audio_file = generate_tactical_radio_dispatch(
            script_text=radio_script,
            output_filename=f"{site_id}_dispatch.wav"
        )

        dossier_entry = {
            "site_id": site_id,
            "zone": zone,
            "area_sqm": area_sqm,
            "area_bigha": area_bigha,
            "coordinates": {"lat": lat, "lon": lon},
            "meta_models_used": {
                "vision_reasoner": intel.get("model_used", "meta-llama/Llama-3.2-11B-Vision-Instruct"),
                "boundary_segmentor": sam2_metrics["architecture"],
                "feature_anomaly_detector": dinov2_metrics["foundation_model"],
                "vector_memory_bank": "Meta FAISS (Facebook AI Similarity Search)",
                "dialect_translator": "Meta NLLB-200 (No Language Left Behind)",
                "speech_engine": "Meta MMS (Massively Multilingual Speech)",
                "safety_guard": intel.get("llama_guard_verdict", "Llama Guard 3 Verified")
            },
            "sam2_metrics": sam2_metrics,
            "dinov2_metrics": dinov2_metrics,
            "faiss_precedents": faiss_matches,
            "satellite_chip": f"monitoring_results/chips/{site_id}_satellite_chip.png",
            "sam2_comparison_image": f"monitoring_results/chips/{site_id}_sam2_comparison.png",
            "dinov2_heatmap_image": f"monitoring_results/chips/{site_id}_dinov2_heatmap.png",
            "visual_forensic_findings": intel.get("visual_forensic_findings"),
            "environmental_threat": intel.get("environmental_threat"),
            "statutory_violation": intel.get("statutory_violation"),
            "demolition_notice_en": intel.get("legal_demolition_notice_en"),
            "demolition_notice_hi": intel.get("legal_demolition_notice_hi"),
            "demolition_notice_awa": dialects.get("notice_awadhi"),
            "demolition_notice_bho": dialects.get("notice_bhojpuri"),
            "tactical_radio_script": radio_script,
            "tactical_radio_script_hi": intel.get("tactical_radio_script_hi"),
            "audio_file": f"monitoring_results/audio/{site_id}_dispatch.wav",
            "pdf_warrant": f"monitoring_results/warrants/{site_id}_Official_Demolition_Warrant.pdf"
        }
        
        # Generate official Court Demolition Warrant PDF
        generate_official_enforcement_pdf(dossier_entry)
        
        dossiers.append(dossier_entry)
        print(f"  ✓ Meta AI Dossier & Official Warrant PDF complete for {site_id}.\n")

    # 3. Save Artifacts for local analysis and Web Dashboard
    print("📑 [STEP 3/3] Exporting Meta AI Dossiers & Web Dashboard Embeds...")
    
    # Save standard JSON dossiers
    meta_json_path = ROOT_DIR / "monitoring_results" / "meta_llama_enforcement_dossiers.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(dossiers, f, indent=2, ensure_ascii=False)

    # Also save backwards-compatible JSON for existing tools
    gemma_compat_path = ROOT_DIR / "monitoring_results" / "gemma_enforcement_dossiers.json"
    compat_dossiers = []
    for d in dossiers:
        c_entry = dict(d)
        c_entry["gemma_forensics"] = d["visual_forensic_findings"]
        c_entry["demolition_notice"] = d["demolition_notice_en"]
        compat_dossiers.append(c_entry)
    with open(gemma_compat_path, "w", encoding="utf-8") as f:
        json.dump(compat_dossiers, f, indent=2, ensure_ascii=False)

    # Save embedded JS data for index.html (CORS-free instant access)
    results_js_dir = ROOT_DIR / "results"
    results_js_dir.mkdir(parents=True, exist_ok=True)
    dossiers_js_path = results_js_dir / "meta_dossiers_data.js"
    with open(dossiers_js_path, "w", encoding="utf-8") as f:
        f.write(f"window.META_DOSSIERS = {json.dumps(dossiers, indent=2, ensure_ascii=False)};\n")

    print(f"  ✓ Saved JSON dossier: {meta_json_path}")
    print(f"  ✓ Saved Dashboard JS embed: {dossiers_js_path}")
    print("=" * 75)
    print("🚀 SUCCESS: Complete Meta AI Surveillance & Enforcement Pipeline Finished!")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline(top_n_alerts=4)
