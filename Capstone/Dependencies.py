import pandas as pd

continuous_features = ['APRDRG_Risk_Mortality','APRDRG_Severity','ZIPINC_QRTL','AGE','DMONTH','I10_NDX',
                       'Log_TOTCHG','Log_N_DISC_U','Log_N_HOSP_U','Log_TOTAL_DISC','Log_LOS','Log_I10_NPR']
categorical_features = ['APRDRG','HOSP_BEDSIZE','H_CONTRL','HOSP_URCAT4','HOSP_UR_TEACH','DIED','DISPUNIFORM','ELECTIVE',
                        'FEMALE','HCUP_ED','PAY1','PL_NCHS','REHABTRANSFER','RESIDENT','SAMEDAYEVENT','Primary_dx']
text_features = ['Secondary_dx','Procedure','MBD_dx']

def make_dependencies(valid_df, test_df, valid_yhat, test_yhat):
    valid_df['Prob'] = valid_yhat
    test_df['Prob'] = test_yhat
    app_df = pd.concat([valid_df,test_df])
    central_tendency = dict()
    for header in continuous_features:
        var_mean = app_df[header].mean(axis=0)
        central_tendency[header] = var_mean
    for header in categorical_features:
        var_mode = app_df[header].mode()[0]
        central_tendency[header] = var_mode
    for header in text_features:
        var_mode = app_df[header].mode()[0]
        central_tendency[header] = var_mode
    
    f = open("defaults.txt","w")
    f.write(str(central_tendency))
    f.close()

    app_df_short = app_df[['Primary_dx','Prob']]
    # Change file directory path to your working directory
    app_df_short.to_csv('D:\MLE Capstone Project\Data\app_data.csv', index=False)