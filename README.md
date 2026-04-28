This project requires an .env file at the project root that contains IGDB_CLIENT_ID and IGDB_CLIENT_SECRET (Twitch developer account required) as well as Python 3.10+ and the following libraries:

pandas , numpy , lightgbm , scikit-learn , shap , requests , beautifulsoup4 , lxml , rapidfuzz , python-dotenv , joblib , plotly

Reccomended Execution Order:

python src/scrape_pc_catalog.py
    output: data/pc_catalog.csv

python src/fetch_igdb.py
    output: data/master_games.csv

python src/tag_hardware.py
    output: tags is_hardware column in master_games.csv

python src/scrape_vgchartz.py
    output: data/sales_data.csv

python src/scrape_pricecharting.py
   output: data/price_history.csv

python src/merge.py
    output: data/merged_dataset.csv
    output: data/excluded_games.csv

python src/features.py
    output: overwrites merged_dataset.csv (production)

python src/features.py --eval
    output: data/merged_dataset_eval_2023.csv

python src/model.py
    output: data/predictions.csv
    output: data/eval_predictions_2023.csv
    output: data/shap_values.csv
    output: data/metrics.json
    output: data/models/prod/ and data/models/eval/

python src/build_lookup.py
    output: data/lookup_prod.html
    output: data/lookup_eval.html