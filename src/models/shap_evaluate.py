import shap
import pandas as pd
import matplotlib.pyplot as plt

def run_shap_analysis(model, X_sample, fig_dir):
    print("Generating basic SHAP summary for Phase 3...")
    xgb_model = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']
    
    X_trans = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()
    X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
    
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_trans_df)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_trans_df, show=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "06_shap_global_basic.png")
    plt.close()
