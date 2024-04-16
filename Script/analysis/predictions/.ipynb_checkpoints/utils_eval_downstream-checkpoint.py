import pandas as pd
import numpy as np
import importlib
from icd9cms.icd9 import search
from sklearn.metrics import average_precision_score as APS
from sklearn.metrics import precision_recall_curve as PRC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('../')
from embeddings import utils_embeddings


def metrics_each_diagnosis(y_tokenized_cols, id2token, y_tokenized_pred, y_tokenized, y):
    # get y_train labels
    y_cols_code = []
    y_cols_name = []
    
    for token in y_tokenized_cols:
        y_cols_code.append(id2token['diag'][token])

    y_cols_dict = {}
    for val in y_cols_code:
        try:
            code = str(search(str(val)+'0').parent).split(':')[:2] # search() function is from icd9cms.icd9
            #print(code)
            y_cols_dict[val] = code[1]  
        except:
            y_cols_dict[val] = str(val)
            continue

            
    #populate diag name
    y_cols_name = [y_cols_dict[key] for key in y_cols_dict.keys()]
    
    # create df for diag_code, diag_name
    df_y_cols = pd.DataFrame({'diag_code': y_cols_code, 'diag_name':y_cols_name, 'diag_encoded':y_tokenized_cols})

    
    # drop if 'UNK' or starts with 'E'
    #df_y_cols = df_y_cols[~df_y_cols.diag_code.str.startswith(('[UNK]', 'E'))]
    
    # add chapter to diag_code
    df_y_cols = utils_embeddings.add_chapter_to_diag(df_y_cols)
        
    # add colors to chapter
    df_y_cols, hue_order = utils_embeddings.add_colors_for_diag(df_y_cols)
    
    # list with diag_names and diag_chapters
    diag_names = df_y_cols.diag_name.values
    diag_chapter = df_y_cols.diag_chapter.values
    
    # compute metrics #
    ###################
    precision = {}
    recall = {}
    aps = {}
    fpr = {}
    tpr = {}
    area = {}
    
    ## compute metrics for each diag or each chapter
    diags = []
    chapters = []
    for idx, key in enumerate(diag_names):
        # precision and recall
        precision[(diag_chapter[idx], key)], recall[(diag_chapter[idx], key)], _ = PRC(
            y_tokenized[:, idx],
            y_tokenized_pred[:, idx]
        )
        
        # average precision score
        aps[(diag_chapter[idx], key)] = APS(
            y_tokenized[:, idx],
            y_tokenized_pred[:, idx],
            average='samples'
        )

        # ROC curve and ROC area (One-vs-Rest ROC AUC score)
        fpr[(diag_chapter[idx], key)], tpr[(diag_chapter[idx], key)], _ = roc_curve(
            y_tokenized[:, idx],
            y_tokenized_pred[:,idx],
        )
        area[(diag_chapter[idx], key)] = auc(fpr[(diag_chapter[idx], key)], tpr[(diag_chapter[idx], key)])
        
    # update lists of diag names and chapters
    for key in aps.keys():
        diags.append(key[1])
        chapters.append(key[0])
        
    # create df with aps and area at the diagnosis or chapter level
    df_aps_auc = pd.DataFrame(
        {
            'aps': aps.values(),
            'area': area.values(),
            'diag_name':diags,
            'diag_chapter':chapters,
            'key': recall.keys()
        }
    )
    
    # add chapter colors
    df_aps_auc = df_aps_auc.merge(
        df_y_cols[['diag_name', 'diag_chapter_color', 'diag_code']],
        on='diag_name',
        how='left'
    )
    
    # add count of patients per diagnosis code
    temp_df = pd.DataFrame({'diag_code': y})
    temp_df['counts'] = 1
    temp_df = temp_df.groupby('diag_code', as_index=False).counts.count()
    df_aps_auc = df_aps_auc.merge(temp_df, on='diag_code', how='left')
    
 
    # transform array values to lists in dictionaries
    for idx, key in enumerate(recall.keys()):
        recall[key] = list(recall[key])
        precision[key] = list(precision[key])
        fpr[key] = list(fpr[key])
        tpr[key] = list(tpr[key])

    return recall, precision, fpr, tpr, df_aps_auc, df_y_cols
    
    
def metrics_averages(y_tokenized_cols, id2token, y_tokenized_pred, y_tokenized):
    ''''''
    precision_micro = []
    recall_micro = []
    aps_samples = []
    
    fpr_micro= []
    tpr_micro = []
    area_micro = []

            
    ## a "micro(sample)-average": quantifying score on all classes jointly
    # precision and recall
    precision_micro, recall_micro, _ = PRC(
        y_tokenized.ravel(),
        y_tokenized_pred.ravel()
    )
    
    
    # average precision score
    aps_samples = APS(
        y_tokenized,
        y_tokenized_pred,
        average="samples"
    )

    # ROC curve and ROC area (Micro-averaged One-vs-Rest ROC AUC score)
    fpr_micro, tpr_micro, _ = roc_curve(
        y_tokenized.ravel(),
        y_tokenized_pred.ravel()
    )
    area_micro = auc(fpr_micro, tpr_micro)
    
    return precision_micro, recall_micro, fpr_micro, tpr_micro, aps_samples, area_micro




# plot ROC and PR
def plot_pr_roc(
    recall_diag, precision_diag, fpr_diag, tpr_diag, df_aps_auc_diag,
    precision_micro, recall_micro, fpr_micro, tpr_micro, aps_samples, area_micro,
    top):
    ''''''
    # all diagnosis codes
    diags = fpr_diag.keys()
    colors = df_aps_auc_diag.diag_chapter_color.values

    # find top diag by area and aps
    top_area_diags = list(df_aps_auc_diag.nlargest(top, 'area').key.values)
    #top_area_colors =  list(df_aps_auc_diag.nlargest(top, 'area').diag_chapter_color.values)
    top_area_colors =  sns.color_palette("Blues_r", len(top_area_diags))

    top_aps_diags = list(df_aps_auc_diag.nlargest(top, 'aps').key.values)
    #top_aps_colors =  list(df_aps_auc_diag.nlargest(top, 'area').diag_chapter_color.values)
    top_aps_colors =  sns.color_palette("Blues_r", len(top_aps_diags))


    # find least diag by area and aps
    least_area_diags = list(df_aps_auc_diag.nsmallest(top, 'area').key.values)
    #least_area_colors =  list(df_aps_auc_diag.nsmallest(top, 'area').diag_chapter_color.values)
    least_area_colors =  sns.color_palette("Reds_r", len(least_area_diags))



    least_aps_diags = list(df_aps_auc_diag.nsmallest(top, 'aps').key.values)
    #least_aps_colors =  list(df_aps_auc_diag.nsmallest(top, 'aps').diag_chapter_color.values)
    least_aps_colors =  sns.color_palette("Reds_r", len(least_aps_diags))


    # set line width
    lw = 1

    # set figure size
    fig, axes = plt.subplots(1,2, figsize=(20,10))

    for idx_ax, ax in enumerate(axes.flatten()):
        # plot for ROC #
        ################
        if idx_ax==1:
            # metrics at the micro-average level
            ax.plot(
                fpr_micro,
                tpr_micro,
                label="sample-average ROC curve (AUC = {0:0.3f})".format(area_micro),
                color="black",
                linestyle=":",
                linewidth=4
            )

            # random classifier
            ax.plot(
                [0, 1],
                [0, 1],
                "k--",
                color="black",
                linewidth=2,
                label='random classifier'
            )


            # metrics for one diag (for label purposes)
            for idx, diag in enumerate(diags):
                if idx==0:
                    ax.plot(
                        fpr_diag[diag],
                        tpr_diag[diag],
                        color='grey',
                        label="diagnosis code ROC curve",
                        lw=lw,
                        alpha=0.2
                    )


            # metrics for each diag
            for idx, diag in enumerate(diags):
                ax.plot(
                    fpr_diag[diag],
                    tpr_diag[diag],
                    color='grey',
                    lw=lw,
                    alpha=0.2
                )

            # metrics for top diag
            for idx, diag in enumerate(top_area_diags):
                ax.plot(
                    fpr_diag[diag],
                    tpr_diag[diag],
                    color=top_area_colors[idx],
                    lw=lw,
                    label="{0} (AUC = {1:0.3f})".format(diag[1], df_aps_auc_diag[df_aps_auc_diag.diag_name.eq(diag[1])].area.values[0]),
                    alpha=0.5
                )


            # metrics for least diag
            temp_diags = []
            for val in least_area_diags:
                temp_diags.append(val[1])

            for idx in range(len(temp_diags)):
                if temp_diags[idx]=='E2':
                    temp_diags[idx] = 'Other hypothyroidism'
                if temp_diags[idx]=='E3':
                    temp_diags[idx] = 'Subclinical iodine-deficiency hypothyroidism'

            for idx, diag in enumerate(least_area_diags):
                ax.plot(
                    fpr_diag[diag],
                    tpr_diag[diag],
                    color=least_area_colors[idx],
                    lw=lw,
                    label="{0} (AUC = {1:0.3f})".format(temp_diags[idx], df_aps_auc_diag[df_aps_auc_diag.diag_name.eq(diag[1])].area.values[0]),
                    alpha=0.5
                )

        # plot for Precision-Recall #
        #############################
        if idx_ax==0:
            # metrics at the micro-average level
            ax.plot(
                recall_micro,
                precision_micro,
                label="sample-average PR curve (APS = {0:0.3f})".format(aps_samples),
                color="black",
                linestyle=":",
                linewidth=4
            )

            # metrics for one diag (for label purposes)
            for idx, diag in enumerate(diags):
                if idx==0:
                    ax.plot(
                        recall_diag[diag],
                        precision_diag[diag],
                        color='grey',
                        label="diagnosis code PR curve",
                        lw=lw,
                        alpha=0.2
                    )

            # metrics for each diag
            for idx, diag in enumerate(diags):
                ax.plot(
                    recall_diag[diag],
                    precision_diag[diag],
                    color='grey',
                    lw=lw,
                    alpha=0.2
                )

            # metrics for top diag
            for idx, diag in enumerate(top_aps_diags):
                ax.plot(
                    recall_diag[diag],
                    precision_diag[diag],
                    color=top_aps_colors[idx],
                    lw=lw,
                    label="{0} (APS = {1:0.3f})".format(diag[1], df_aps_auc_diag[df_aps_auc_diag.diag_name.eq(diag[1])].aps.values[0]),
                    alpha=0.5
                )


            # metrics for least diag
            temp_diags = []
            for val in least_aps_diags:
                temp_diags.append(val[1])

            for idx in range(len(temp_diags)):
                if temp_diags[idx]=='E2':
                    temp_diags[idx] = 'Other hypothyroidism'
                if temp_diags[idx]=='E3':
                    temp_diags[idx] = 'Subclinical iodine-deficiency hypothyroidism'

            for idx, diag in enumerate(least_aps_diags):
                ax.plot(
                    fpr_diag[diag],
                    tpr_diag[diag],
                    color=least_aps_colors[idx],
                    lw=lw,
                    label="{0} (APS = {1:0.3f})".format(temp_diags[idx], df_aps_auc_diag[df_aps_auc_diag.diag_name.eq(diag[1])].aps.values[0]),
                    alpha=0.5
                )

        # set labels
        if idx_ax==1:
            ax.set_xlabel('False Positive Rate (1-Specificity)')
            ax.set_ylabel('True Positive Rate (Sensitivity)')
        else:
            ax.set_xlabel('Recall (Sensitivity)')
            ax.set_ylabel('Precision')       

        # despine
        for val in ['left', 'bottom', 'right', 'top']:
            ax.spines[val].set_visible(True)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

        ax.legend(
            loc="upper center",
            frameon=False,
            bbox_to_anchor=(0.6, -0.09),
            ncol=1,
            fontsize='small'
        )
    return top_area_diags, top_aps_diags, least_area_diags, least_aps_diags

    
def process_func(df, col):
    ''''''
    df[col] = df[col].astype(str)
    df[col] = df[col].str.split('.').str[0]
    df[col] = np.where(df[col].isin(('unkn', 'nan')), 199, df[col])
    df[col] = df[col].astype(int)

    return df[col]
            
def fairaware_cleaning(df, effect_of_diag_hist_len=False):
    ''''''
    # rename some of the column entries
    df['raceI'] = np.where(df.raceI.eq('white'), 'Wh', 
                          np.where(df.raceI.eq('black'), 'Bl',
                                  np.where((df.raceI.eq('hisp') | df.raceI.eq('other')), 'Hisp_Oth',
                                          np.where(df.raceI.eq('asian/pacific islander'), 'AS_PI',
                                                   np.where(df.raceI.eq('native american/eskimo/aleut'), 'NAm_EA', 'unkn')))))

    df['raceM'] = np.where(df.raceM.eq('white'), 'Wh', 
                          np.where(df.raceM.eq('black'), 'Bl',
                                  np.where((df.raceM.eq('hisp') | df.raceM.eq('other')), 'Hisp_Oth',
                                          np.where(df.raceM.eq('asian/pacific islander'), 'AS_PI',
                                                   np.where(df.raceM.eq('native american/eskimo/aleut'), 'NAm_EA', 'unkn')))))

    df['meduc'] = np.where(df.meduc.eq('high school or less'), '<HS',
                          np.where((df.meduc.eq('college (1-3 years)') | df.meduc.eq('college (4 years)')), 'college',
                                  np.where(df.meduc.eq('masters or phd'), 'grad', 'unkn')))

    df['precare'] = np.where(df.precare.isna(), 'unkn',
                            np.where(df.precare.eq('-'), 'unkn', df.precare))


    df['bthresmb_name_grp'] = np.where(df.bthresmb_name.eq('Mexico'), 'Mexico',
                                        np.where(df.bthresmb_name.eq('China'), 'China',
                                             np.where(df.bthresmb_name.eq('Vietnam'), 'Vietnam',   
                                                       np.where(df.bthresmb_name.eq('Cuba'), 'Cuba',
                                                                 np.where(df.bthresmb_name.eq('Remainder of the World'), 'RoW',
                                                                    np.where(df.bthresmb_name.eq('Reminder of the World'), 'RoW',
                                                                                      np.where(df.bthresmb_name.eq('Guam'), 'RoW', #'Guam'
                                                                                               np.where(df.bthresmb_name.eq('Canada'), 'Canada', 'USA'))))))))


    df['age'] = process_func(df, 'age')
    df['age_grp'] = pd.cut(df.age,bins=[-1,2,17,26, 199],labels=['baby: 0-2','child: 3-17','adult: >17', 'unkn'])

    df['pm25I'] = np.where(df.pm25I.isna(), 999, df.pm25I)
    df['pm25I'] = df.pm25I.astype(int)
    df['pm25I_grp'] = pd.cut(df.pm25I,bins=[-1, 12, 35, 199, 999],labels=['good','moderate','unhealthy', 'unkn'])

    df['precare'] = process_func(df, 'precare')
    df['precare_grp'] = pd.cut(df.precare,bins=[-1,3,6,9,199],labels=['1st trimester','2nd trimester','3rd trimister', 'unkn'])

    df['prevsts'] = process_func(df, 'prevsts')
    df['prevsts_grp'] = pd.cut(df.prevsts, bins=[-1,7,12,20,150, 199],labels=['low: 0-7 visits', 'normal: 8-12 visits', 'high: 13-20 visits', 'very high: >20 visits', 'unkn'])

    df['visitsM_9mpp'] = process_func(df, 'visitsM_9mpp')
    df['visitsM_9mpp_grp'] = pd.cut(df.visitsM_9mpp,bins=[-1,10,198,199],labels=['0-10 visits', '>10 visits', 'unkn'])

    df['visitsM_1ypp'] = process_func(df, 'visitsM_1ypp')
    df['visitsM_1ypp_grp'] = pd.cut(df.visitsM_1ypp,bins=[-1,10,198,199],labels=['0-10 visits', '>10 visits', 'unkn'])

    df['cntyresI_name'] = np.where(df.cntyresI_name.isna(), 'unknown/outside CA/homeless', df.cntyresI_name)
    
    if effect_of_diag_hist_len == True:
        df['diag_hist_len'] = df.random_t + 1
        df['diag_hist_len_grp'] = pd.cut(df.diag_hist_len,bins=[-1,3,6,199],labels=['3 visits','4-6 visits','>7 visits'])
    
    
    # define regions in CA (based on Census data: https://census.ca.gov/regions/)
    superior_cali = (
    'Butte', 'Colusa', 'El Dorado', 'Glenn',
    'Lassen', 'Modoc', 'Nevada', 'Placer',
    'Plumas', 'Sacramento', 'Shasta', 'Sierra',
    'Siskiyou', 'Sutter', 'Tehama', 'Yolo', 'Yuba')

    north_coast = (
        'Del Norte', 'Humboldt', 'Lake', 'Mendocino', 'Napa', 'Sonoma', 'Trinity')

    sf_bay_area = (
        'Alameda', 'Contra Costa', 'Marin',
        'San Francisco', 'San Mateo', 'Santa Clara',
        'Solano')

    northern_sjv = (
        'Alpine', 'Amador', 'Calaveras', 'Madera', 'Mariposa', 'Merced', 'Mono', 'San Joaquin', 'Stanislaus', 'Tuolumne')

    central_coast = (
        'Monterey', 'San Benito', 'San Luis Obispo', 'Santa Barbara', 'Santa Cruz', 'Ventura')

    southern_sjv = (
        'Fresno', 'Inyo', 'Kern', 'Kings', 'Tulare')

    inland_empire = (
        'Riverside', 'San Bernardino')

    la_county = ['Los Angeles']

    orange_county = ['Orange']

    san_diego_imperial = (
        'Imperial', 'San Diego')

    df['cntyresI_name'] = np.where(df.cntyresI_name.isna(), 'unkn', df.cntyresI_name)
    df['cntyresI_name_grp'] = np.where(df.cntyresI_name.isin(superior_cali), 'Superior Cali',
                                                        np.where(df.cntyresI_name.isin(north_coast), 'North Coast',
                            np.where(df.cntyresI_name.isin(sf_bay_area), 'SF Bay Area',
                                                           np.where(df.cntyresI_name.isin(northern_sjv), 'San Joaquin Valley',
                            np.where(df.cntyresI_name.isin(central_coast), 'Central Coast',
                                                           np.where(df.cntyresI_name.isin(southern_sjv), 'San Joaquin Valley',
                            np.where(df.cntyresI_name.isin(inland_empire), 'Inland Empire',
                                                           np.where(df.cntyresI_name.isin(la_county), 'Los Angeles County',
                            np.where(df.cntyresI_name.isin(orange_county), 'Orange County',
                                                           np.where(df.cntyresI_name.isin(san_diego_imperial), 'San Diego-Imperial', 'unkn/not CA/homeless'))))))))))
    
    return df





def fairaware_plots(df, y_tokenized_cols, id2token, y_tokenized_pred, y_tokenized, effect_of_diag_hist_len=False):
    ''''''
    # set figure size
    fig, axes = plt.subplots(3, 4, figsize=(18,18), sharey=False, sharex=False)
    
    if effect_of_diag_hist_len==True:
        sensitivities = ['sexI', 'raceI', 'raceM', 'meduc',
                  'precare_grp', 'prevsts_grp', 'visitsM_9mpp_grp', 'visitsM_1ypp_grp',
                  'bthresmb_name_grp', 'age_grp', 'cntyresI_name_grp', 'diag_hist_len_grp' #'pm25I_grp'
        ]
        
    else:
        sensitivities = ['sexI', 'raceI', 'raceM', 'meduc',
                  'precare_grp', 'prevsts_grp', 'visitsM_9mpp_grp', 'visitsM_1ypp_grp',
                  'bthresmb_name_grp', 'age_grp', 'cntyresI_name_grp', 'pm25I_grp'
        ]
    
    # set title
    if effect_of_diag_hist_len==True:
        titles = [
            'baby/patient gender', 'baby/patient race',  'mother race',  'mother education',
            'month prenatal care began', 'number of prenatal visits', 'mother inp/ER visits 9months b. birth',
            'mother inp/ER visits 12 months a. birth',
            'mother country at her own birth', 'patient age at visit', 'CA region of baby/patient at birth',    
            'baby/patient inp/ER visits'
        ]
    else:
        titles = [
            'baby/patient gender', 'baby/patient race',  'mother race',  'mother education',
            'month prenatal care began', 'number of prenatal visits', 'mother inp/ER visits 9months b. birth',
            'mother inp/ER visits 12 months a. birth',
            'mother country at her own birth', 'patient age at visit', 'CA region of baby/patient at birth',    
            'baby/patient pm25 at birth'
        ]

    # for each unique group in sensitivity_col
    for idx_sens, ax in enumerate(axes.flatten()):
        sensitivity_col = df[sensitivities[idx_sens]].values
        print(sensitivities[idx_sens])
        
        # set prefered order for some columns
        if idx_sens==5:
            unique_sensitivity_levels = [
                'low: 0-7 visits', 'normal: 8-12 visits',
                'high: 13-20 visits', 'very high: >20 visits',
                'unkn']
        elif idx_sens==9:
            unique_sensitivity_levels = [
                'baby: 0-2','child: 3-17','adult: >17', 'unkn']
        else:
            unique_sensitivity_levels = np.sort(list(set(sensitivity_col)))
            
        # set colors
        colors = sns.color_palette("rocket_r", len(unique_sensitivity_levels))
        for idx_level, level in enumerate(unique_sensitivity_levels):
            if idx_sens in [4, 5, 9, 10] and level=='unkn':
                continue
            # compute micro-level metrics for each level of sensitivity_col
            precision_micro_temp, recall_micro_temp, fpr_micro_temp, tpr_micro_temp, aps_samples_temp, area_micro_temp = metrics_averages(
                y_tokenized_cols,
                id2token,
                y_tokenized_pred[sensitivity_col==level],
                y_tokenized[sensitivity_col==level]
            )

            # plot ROC and AUC
            ax.plot(
                fpr_micro_temp,
                tpr_micro_temp,
                label= level + " (AUC = {0:0.3f})".format(area_micro_temp),
                color=colors[idx_level],
                linestyle=":",
                linewidth=2
            )


        # random classifier
        #if idx_level==len(list(set(sensitivity_col)))-1: 
        ax.plot(
            [0, 1],
            [0, 1],
            "k--",
            color="black",
            linewidth=1,
            label='random classifier'
        )

        ax.legend(
            loc="lower right",
            frameon=False,
            #bbox_to_anchor=(0.6, -0.09),
            ncol=1,
            fontsize='small'
        );


        # set title and labels
        ax.set_title(titles[idx_sens])
        #sns.despine(ax=ax, right=True, top=True, left=False)

        if idx_sens in [0, 4, 8]:
            ax.set_ylabel('True Positive Rate (Sensitivity)')
        else:
            ax.set_ylabel('')


        if idx_sens in [8, 9, 10, 11]:
            ax.set_xlabel('False Positive Rate (1-Specificity)')
        else:
            ax.set_xlabel('')


        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])