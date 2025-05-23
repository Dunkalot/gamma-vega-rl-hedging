# cost scenarios - cvar
# python run.py    -dataset_train=calm_flat_5pct_curve_6m_1y -dataset_eval=calm_flat_5pct_curve_6m_1y -eval_offset=45000 -train_sim=45000 -eval_sim=5000 -spread_train=0.005 -spread_eval=0.005 -critic=qr-huber -obj_func=cvar
# python run.py    -dataset_train=calm_flat_5pct_curve_2y_1y -dataset_eval=calm_flat_5pct_curve_2y_1y -eval_offset=45000 -train_sim=45000 -eval_sim=5000 -spread_train=0.005 -spread_eval=0.005 -critic=qr-huber -obj_func=cvar

python run.py    -dataset_train=calm_flat_5pct_curve_6m_1y -dataset_eval=calm_flat_5pct_curve_6m_1y -eval_offset=45000 -train_sim=20000 -eval_sim=5000 -spread_train=0.01 -spread_eval=0.01 -critic=qr-huber -obj_func=cvar
# python run.py    -dataset_train=calm_flat_5pct_curve_2y_1y -dataset_eval=calm_flat_5pct_curve_2y_1y -eval_offset=45000 -train_sim=45000 -eval_sim=5000 -spread_train=0.01 -spread_eval=0.01 -critic=qr-huber -obj_func=cvar

# python run.py    -dataset_train=calm_flat_5pct_curve_6m_1y -dataset_eval=calm_flat_5pct_curve_6m_1y -eval_offset=45000 -train_sim=45000 -eval_sim=5000 -spread_train=0.02 -spread_eval=0.02 -critic=qr-huber -obj_func=cvar
# python run.py    -dataset_train=calm_flat_5pct_curve_2y_1y -dataset_eval=calm_flat_5pct_curve_2y_1y -eval_offset=45000 -train_sim=45000 -eval_sim=5000 -spread_train=0.02 -spread_eval=0.02 -critic=qr-huber -obj_func=cvar






# # stress regime 
# # training and own regime
# python run.py    -dataset_train=stress_flat_5pct_curve_2y_1y -dataset_eval=stress_flat_5pct_curve_2y_1y -eval_offset=45000 -train_sim=45000 -eval_sim=5000 -spread_train=0.01 -spread_eval=0.01 -critic=qr-huber -obj_func=cvar

# #calm
# python run.py    -dataset_train=stress_flat_5pct_curve -dataset_eval=calm_flat_5pct_curve -eval_offset=20000 -train_sim=0 -eval_sim=5000 -spread_train=0.01 -spread_eval=0.01 -critic=qr-huber -obj_func=cvar -eval_only=True
# #5050
# python run.py    -dataset_train=stress_flat_5pct_curve -dataset_eval=5050_flat_5pct_curve -eval_offset=0 -train_sim=0 -eval_sim=5000 -spread_train=0.01 -spread_eval=0.01 -critic=qr-huber -obj_func=cvar -eval_only=True



# # curriculum regime, calm->stress
# # training on own regime (5050)
# python run.py    -dataset_train=curriculum_flat_5pct_curve -dataset_eval=5050_flat_5pct_curve -eval_offset=0 -train_sim=20000 -eval_sim=5000 -spread_train=0.01 -spread_eval=0.01 -critic=qr-huber -obj_func=cvar
# # calm
# python run.py    -dataset_train=curriculum_flat_5pct_curve -dataset_eval=calm_flat_5pct_curve -eval_offset=20000 -train_sim=0 -eval_sim=5000 -spread_train=0.01 -spread_eval=0.01 -critic=qr-huber -obj_func=cvar -eval_only=True
# # stress
# python run.py    -dataset_train=curriculum_flat_5pct_curve -dataset_eval=stress_flat_5pct_curve -eval_offset=20000 -train_sim=0 -eval_sim=5000 -spread_train=0.01 -spread_eval=0.01 -critic=qr-huber -obj_func=cvar -eval_only=True