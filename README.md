# IntelliScript

Turn messy medical documents into clean, machine-readable data.

IntelliScript is a modular Python pipeline that converts scanned or uploaded medical documents into structured, actionable outputs. Whether you're dealing with handwritten notes, faxed lab reports, or PDF scans, the pipeline handles preprocessing, text extraction, entity recognition, validation, and report generation—all in a clean, extensible architecture.

## What It Does

**The Problem:**
Medical documents are messy. They arrive as scans, PDFs, handwritten notes, or poor-quality images. Extracting reliable data from them requires multiple processing steps: image cleanup, text extraction, entity detection, validation against known ranges, and finally, structured output.

**The Solution:**
IntelliScript breaks this into eight clear stages, each handling one specific task. Each stage is independent, testable, and easy to extend—so you can understand exactly where failures happen and fix them without touching everything else.

Here's the flow:
1. **Input ingestion** – Accept PDFs and images
2. **Preprocessing** – Clean images for better text extraction
3. **OCR & text extraction** – Convert images to readable text
4. **PHI scrubbing boundary** – Prepare for sensitive data handling (extensible)
5. **Entity extraction** – Pull out patient vitals, test results, diagnoses
6. **Validation** – Check extracted values against medical ranges
7. **Summarization** – Generate both technical and patient-friendly summaries
8. **Output packaging** – Deliver structured JSON reports

## Key Features

✓ **Multi-stage pipeline** – Each step isolated, testable, and extensible  
✓ **PDF & image handling** – Processes both scanned documents and image files  
✓ **OCR-optimized preprocessing** – Configurable image enhancement for better text extraction  
✓ **Structured extraction** – Type-safe entity models using Pydantic  
✓ **Dataset-driven validation** – Check results against medical reference ranges  
✓ **Dual-audience summaries** – Technical summaries for clinicians, plain-language for patients  
✓ **JSON output** – Machine-readable reports with full audit trail  
✓ **Extensible architecture** – Clear hooks for PHI scrubbing, custom rules, and new extractors  

## Status

This is a **portfolio and engineering case study**. The core pipeline is functional, but some components are intentionally simplified in this version:

- **PHI scrubbing** is a pass-through placeholder (ready for implementation)
- **Diagnosis inference** uses heuristic logic (ready for rule expansion)
- **Validation routing** is currently coded per entity-type (ready for generalization)
- **Test coverage and documentation** are planned additions

Despite these simplifications, the project demonstrates solid data-processing engineering, clean architecture, and thoughtful extensibility.

## Repository Structure

```
intelliscript/
├── README.md                                   # This file
├── LICENSE
├── .gitignore
├── requirements.txt                            # Dependencies
│
├── production/                                 # Entrypoint and config
│   ├── main.py                                # Pipeline orchestrator
│   ├── requirements.txt                        # Production requirements
│   └── README.md                              # Production guide
│
├── data/                                      # Input and output directories
│   ├── cache/                                 # Intermediate processing cache
│   ├── inputs/                                # Input documents
│   │   └── sample.jpg                        # Example document
│   ├── outputs/                               # Final JSON reports
│   │   ├── sample.jpg.result.json            # Extracted data
│   │   └── test_report.pdf.result.json       # Full report output
│   └── temp/                                  # Temporary processing files
│       └── sample.jpg_page_1.png             # Page extraction results
│
├── datasets/                                  # Reference data for validation
│   ├── document_types/
│   │   └── document_keywords.json            # Document type classifiers
│   ├── medical/
│   │   ├── lab_reference_ranges.json         # Normal value ranges
│   │   └── medical_entities.json             # Entity definitions
│   └── templates/
│       └── lab_phases.json                   # Report structure templates
│
├── logs/
│   └── intelliscript.log                     # Processing logs
│
├── src/                                       # Source code
│   ├── __init__.py
│   │
│   ├── config/                               # Configuration management
│   │   ├── settings.py                       # Global settings
│   │   ├── __init__.py
│   │   └── __pycache__/
│   │
│   ├── core/                                 # Core pipeline logic
│   │   ├── constants.py                      # Enums and constants
│   │   ├── exceptions.py                     # Custom exceptions
│   │   ├── logging_config.py                 # Logging setup
│   │   ├── models.py                         # Pydantic data models
│   │   ├── pipeline_manager.py               # Pipeline orchestration
│   │   ├── __init__.py
│   │   └── __pycache__/
│   │
│   ├── services/                             # Utility services
│   │   ├── dataset_service.py                # Dataset loading and caching
│   │   ├── file_utils.py                     # File I/O helpers
│   │   ├── pdf_table_extractor.py            # PDF table parsing
│   │   ├── storage_service.py                # Output storage
│   │   ├── __init__.py
│   │   └── __pycache__/
│   │
│   └── stages/                               # Pipeline stages (8 modules)
│       ├── __init__.py
│       ├── base.py                           # Base stage class
│       ├── classification.py                 # Document type classification
│       ├── extraction.py                     # Entity extraction
│       ├── ocr.py                            # OCR and text extraction
│       ├── packaging.py                      # JSON output packaging
│       ├── phi_scrubbing.py                  # PHI redaction (extensible)
│       ├── preprocessing.py                  # Image preprocessing
│       ├── summarization.py                  # Summary generation
│       ├── validation.py                     # Validation against ranges
│       ├── __init__.py
│       └── __pycache__/
│
├── tests/                                    # Test suite
│   └── (test files for pipeline stages)
│
└── Report/                                   # Documentation and samples
    └── (Training reports and example outputs)
```

## Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/intelliscript.git
cd intelliscript

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# From the production directory
cd production
python main.py

# Or from the repo root
python -m production.main
```

**Configuration:**
The pipeline uses `src/config/settings.py` for configuration. Key settings include:
- Input/output paths
- OCR and preprocessing parameters
- Validation thresholds (loaded from datasets)
- Logging levels

You can override settings via environment variables or by editing the config file directly.

## Pipeline Stages Explained

| Stage | Input | Output | Key Logic |
|-------|-------|--------|-----------|
| **Ingestion** | PDF or image file | Raw bytes + metadata | Format detection, file validation |
| **Preprocessing** | Raw image | Enhanced image | OpenCV filters for OCR optimization |
| **OCR** | Processed image | Extracted text | Tesseract or alternative OCR backend |
| **PHI Scrubbing** | Raw text | Scrubbed text | Placeholder for regex/NLP-based redaction |
| **Extraction** | Text + document type | Structured entities | Regex patterns, entity type mapping |
| **Validation** | Entities + reference ranges | Validation results | Range checks, type validation, confidence scoring |
| **Summarization** | Entities + validation | Summary text | Technical and layman-friendly narratives |
| **Packaging** | All intermediate outputs | JSON report | Structured output with metadata |

## Example Output

The pipeline produces structured JSON reports:

```json
{
  "document_metadata": {
    "file_name": "lab_report.pdf",
    "processing_timestamp": "2025-01-15T10:30:00Z",
    "document_type": "lab_report"
  },
  "extracted_entities": {
    "patient_vitals": [
      {
        "entity_type": "blood_pressure",
        "value": "120/80",
        "unit": "mmHg",
        "confidence": 0.95
      }
    ],
    "test_results": [
      {
        "test_name": "Glucose",
        "value": 95,
        "unit": "mg/dL",
        "reference_range": "70-100",
        "status": "normal"
      }
    ]
  },
  "validation_findings": {
    "all_valid": true,
    "anomalies": []
  },
  "summaries": {
    "technical": "Lab work shows normal metabolic panel...",
    "patient_friendly": "Your recent blood work came back normal..."
  }
}
```

## Tech Stack

- **Python 3.8+** – Core language
- **OpenCV** – Image preprocessing
- **NumPy & Pillow** – Image manipulation
- **Pydantic** – Data validation and models
- **Asyncio** – Async processing (optional)
- **PyMuPDF or pdf2image** – PDF handling
- **Tesseract/PyTesseract** – OCR backend

See `requirements.txt` for full dependency list and pinned versions.

## Important Limitations

**What This Project Does:**
- Demonstrates a clean, modular architecture for document processing
- Shows practical OCR, text extraction, and entity validation workflows
- Provides extension points for production enhancements

**What This Project Does NOT Do (Yet):**
- Full PHI redaction (placeholder only—implement with regex or NLP models)
- Comprehensive diagnosis inference (heuristic-based, not ML-powered)
- Complete rule coverage for all medical entity types
- Production-grade security or HIPAA compliance

**Production Readiness:**
To deploy this in a real medical environment, you would need to:
1. Implement robust PHI scrubbing with audit logging
2. Add HIPAA-compliant storage and transmission
3. Expand entity extraction with domain expertise or ML models
4. Add comprehensive test coverage and validation
5. Implement proper access control and encryption

## Extensibility

The architecture is designed for easy extension:

**Adding New Entity Types:**
1. Define the entity in `src/core/models.py`
2. Add extraction patterns to `src/stages/extraction.py`
3. Define validation ranges in `datasets/medical/`
4. Update summarization rules in `src/stages/summarization.py`

**Adding New Document Types:**
1. Add keywords to `datasets/document_types/document_keywords.json`
2. Create stage-specific logic in relevant pipeline stages
3. Test with sample documents in `data/inputs/`

**Implementing PHI Scrubbing:**
The `src/stages/phi_scrubbing.py` module is ready for expansion. Implement with:
- Regex patterns for common PHI (SSN, MRN, DOB)
- NLP-based entity recognition (names, locations)
- Policy-driven redaction (what to remove, what to mask)

## Roadmap

- [ ] Implement production PHI scrubbing with audit logging
- [ ] Expand entity extraction coverage (medications, allergies, procedures)
- [ ] Add ML-based diagnosis inference (instead of heuristics)
- [ ] Implement stronger unit and integration tests
- [ ] Add CSV and text export formats (in addition to JSON)
- [ ] Create Swagger/OpenAPI documentation
- [ ] Add Docker containerization
- [ ] Build REST API wrapper for production deployment

## AI Use Disclosure

AI tools assisted with portions of the original code and documentation. All code is provided for inspection and review. Please validate, test, and verify all included code before using it in any production or healthcare-related context.

## Development Notes

**Before Committing:**
- ✓ No raw datasets committed (see `.gitignore`)
- ✓ No virtual environments or `.env` files
- ✓ No generated images, PDFs, or heavy artifacts
- ✓ Logs are in `.gitignore`
- ✓ Cache directories are excluded

**Running Tests:**
```bash
pytest tests/ -v
```

**Generating Logs:**
Logs are written to `logs/intelliscript.log` and excluded from version control.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Questions or Contributions?

This is a portfolio project demonstrating data-processing patterns and clean architecture. If you have feedback, spot an issue, or want to discuss the design, feel free to open an issue or reach out.

