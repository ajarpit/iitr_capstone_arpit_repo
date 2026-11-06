# Delhivery Last-Mile Delay Intelligence Pipeline

This repository contains an end-to-end operational intelligence pipeline designed for Delhivery last-mile logistics. It automates data ingestion, feature engineering, delivery time prediction, parcel delay risk classification, hub operational segmentation, and batch action scheduling.

---

## How to Run

Execute the following commands in sequence from a fresh clone:

```bash
# 1. Install pinned dependencies
pip install -r requirements.txt

# 2. Generate raw synthetic datasets
python generate_data.py

# 3. Data cleaning, feature engineering, and EDA checks (Part A)
python part_a_clean_eda.py

# 4. Delivery-hours regression pipeline (Part B)
python part_b_regression.py

# 5. Delay classification and operational alert threshold tuning (Part C)
python part_c_classification.py

# 6. Hub K-Means segmentation and mismatch analysis (Part D)
python part_d_segments.py

# 7. Package final production model artefacts
python package_model.py

# 8. Run batch prediction and generate action list for new shipments
python predict_delay.py