
# Operacion 

MIX = B - A 

# Interpretar resultados de mix 


Los resultados se guardan como `{A}_{B}.xlsx`

## Mix difference 

- Positivo : B clasifica mejor esa clase
- 0 : igual
- Negativo : A clasifica mejor esa clase

## Confusions increased

- Positivo : B confunde más esa clase
- 0 : igual
- Positivo : A confunde más esa clase

# Ejecutar script


python .\compare_mix_confusions.py --excel_a .\baseline\confusion_matrices.xlsx --excel_b .\adaptative_margin\confusion_matrices.xlsx --output resultados/baseline_adaptative.xlsx

python .\compare_mix_confusions.py --excel_a .\baseline\confusion_matrices.xlsx --excel_b .\hc_radio_sem\confusion_matrices.xlsx --output resultados/baseline_hc_radio_sem.xlsx

python .\compare_mix_confusions.py --excel_a .\baseline\confusion_matrices.xlsx --excel_b .\hc_radio_vis\confusion_matrices.xlsx --output resultados/baseline_hc_radio_vis.xlsx


python .\compare_mix_confusions.py --excel_a .\baseline\confusion_matrices.xlsx --excel_b .\hc_margen_sem\confusion_matrices.xlsx --output resultados/baseline_hc_margen_sem.xlsx

python .\compare_mix_confusions.py --excel_a .\baseline\confusion_matrices.xlsx --excel_b .\hc_margen_vis\confusion_matrices.xlsx --output resultados/baseline_hc_margen_vis.xlsx