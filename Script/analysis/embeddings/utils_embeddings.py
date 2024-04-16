import pandas as pd
import numpy as np
import seaborn as sns
from icd9cms.icd9 import search

## add diagnosis names to vocab
###############################
def add_diag_names(vectorize_layer, embeddings):
    ''''''
    diag_name = []
    diag_vocab = vectorize_layer.get_vocabulary()

    for idx, val in enumerate(diag_vocab):
        try:
            code = str(search(val+'0').parent).split(':')[:2] # search() function is from icd9cms.icd9
            diag_name.append(code[1])
        except:
            if  val == '00':
                name = 'Intestinal Infectious Diseases'
                diag_name.append(name)
            # if diag code is not in icd9cms.icd9, continue
            else:
                diag_name.append('nan')
            continue
    
    # add diag name and code to embeddings
    embeddings.insert(0, 'diag_name', np.array(diag_name))
    embeddings.insert(1, 'diag_code', np.array(diag_vocab))
    
    # drop if diag_name is 'nan'
    embeddings = embeddings[~embeddings.diag_name.eq('nan')]
    embeddings.reset_index(drop=True, inplace=True)
    
    return embeddings


## add chapter code
###################
def add_chapter_to_diag(embeddings):
    ''''''
    temp = embeddings.copy()
    temp['diag_code_2d'] = embeddings.diag_code.str[:2]

    # rename V.. diagnosis to 0 (cannot apply float to it)
    temp['diag_code_2d'] = np.where(temp.diag_code_2d.str.contains('\[U'), '999',
                                    np.where(temp.diag_code_2d.str.contains('V'), '9999', 
                                             np.where(temp.diag_code_2d.str.contains('N'), '99999',
                                                    np.where(temp.diag_code_2d.str.contains('E'), '999999', temp.diag_code_2d))))

    # transform diag code to float
    temp['diag_code_2d_flt'] = temp.diag_code_2d.astype(float)
    
    # add chapter category
    temp['chapter'] = np.where(temp.diag_code_2d_flt.le(13), 'Infectious And Parasitic Diseases',
             np.where(temp.diag_code_2d_flt.le(23), 'Neoplasms', 
                     np.where(temp.diag_code_2d_flt.le(27), 'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders', 
                              np.where(temp.diag_code_2d_flt.le(28), 'Diseases Of Blood And Blood-Forming Organs', 
                                    np.where(temp.diag_code_2d_flt.le(31), 'Mental Disorders',    
                                             np.where(temp.diag_code_2d_flt.le(38), 'Diseases Of The Nervous System And Sense Organs', 
                                                      np.where(temp.diag_code_2d_flt.le(45), 'Diseases Of The Circulatory System', 
                                                               np.where(temp.diag_code_2d_flt.le(51), 'Diseases Of The Respiratory System', 
                                                                      np.where(temp.diag_code_2d_flt.le(57), 'Diseases Of The Digestive System', 
                                                                               np.where(temp.diag_code_2d_flt.le(62), 'Diseases Of The Genitourinary System',
            np.where(temp.diag_code_2d_flt.le(67), 'Complications Of Pregnancy, Childbirth, And The Puerperium',
                   np.where(temp.diag_code_2d_flt.le(70), 'Diseases Of The Skin And Subcutaneous Tissue', 
                            np.where(temp.diag_code_2d_flt.le(73), 'Diseases Of The Musculoskeletal System And Connective Tissue', 
                                 np.where(temp.diag_code_2d_flt.le(75), 'Congenital Anomalies',   
                                          np.where(temp.diag_code_2d_flt.le(77), 'Certain Conditions Originating In The Perinatal Period', 
                                                   np.where(temp.diag_code_2d_flt.le(79), 'Symptoms, Signs, And Ill-Defined Conditions', 
                                                            np.where(temp.diag_code_2d_flt.le(99), 'Injury And Poisoning', 
                                                                     np.where(temp.diag_code_2d_flt.le(999), 'unknown', 'Others'))))))))))))))))))

    # add to original df and sort
    embeddings.insert(0, 'diag_chapter', temp.chapter)
    
    return embeddings



def add_colors_for_diag(embeddings):
    ''''''
    hue_order = ['Infectious And Parasitic Diseases', 
    'Neoplasms',
    'Endocrine, Nutritional And Metabolic Diseases, And Immunity Disorders',
    'Diseases Of Blood And Blood-Forming Organs',
    'Mental Disorders',
    'Diseases Of The Nervous System And Sense Organs',
    'Diseases Of The Circulatory System',
    'Diseases Of The Respiratory System',
    'Diseases Of The Digestive System',
    'Diseases Of The Genitourinary System',
    'Complications Of Pregnancy, Childbirth, And The Puerperium',
    'Diseases Of The Skin And Subcutaneous Tissue',
    'Diseases Of The Musculoskeletal System And Connective Tissue',
    'Congenital Anomalies',
    'Certain Conditions Originating In The Perinatal Period',
    'Symptoms, Signs, And Ill-Defined Conditions',
    'Injury And Poisoning',
    'Others'
    ]
    
    colors =  sns.color_palette("rocket_r", embeddings.diag_chapter.nunique())
    chapters = embeddings.diag_chapter.unique()
    
    diag_colors = pd.DataFrame({'diag_chapter': chapters, 'diag_chapter_color': colors})
    embeddings = diag_colors.merge(
        embeddings,
        on='diag_chapter',
        how='right'
    )
    
    return embeddings, hue_order