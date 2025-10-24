
def transform_fe(data_dict):
    data_dict['FE_ratio_ancienneté'] = FE_ratio_ancienneté(data_dict['annees_dans_l_entreprise'],data_dict['annee_experience_totale'])
    data_dict['FE_duree_moy_exp_precedentes'] = FE_duree_moy_exp_precedentes(data_dict['annee_experience_totale'],data_dict['annees_dans_l_entreprise'],data_dict['nombre_experiences_precedentes'])
    data_dict['FE_ratio_evolution'] = FE_ratio_evolution(data_dict['annees_dans_le_poste_actuel'],data_dict['annees_dans_l_entreprise'])
    data_dict['FE_reste_plus_longtemps'] = FE_reste_plus_longtemps(data_dict['annees_dans_l_entreprise'],data_dict['FE_duree_moy_exp_precedentes'])
    data_dict['FE_cadre'] = FE_cadre(data_dict['poste'])

    keys_to_delete = {'annee_experience_totale','annees_dans_l_entreprise','annees_dans_le_poste_actuel',"nombre_experiences_precedentes"}
    data_dict_for_model = {k: v for k,v in data_dict.items() if k not in keys_to_delete}

    return data_dict_for_model


def FE_ratio_ancienneté(annees_exp_entreprise,annees_exp_tot):
    annees_exp_entreprise/(1+annees_exp_tot)

def FE_duree_moy_exp_precedentes(annees_exp_tot, annees_exp_entreprise, nb_exp):
    result = (annees_exp_tot - annees_exp_entreprise) / (nb_exp+1)
    return result

def FE_ratio_evolution(annees_poste_actuel,annees_exp_entreprise):
    result = annees_poste_actuel/(1+annees_exp_entreprise)
    return result
    
def FE_reste_plus_longtemps(annees_exp_entreprise,duree_moy_exp_precedentes):
    if annees_exp_entreprise > duree_moy_exp_precedentes:
        return 1 
    else:
        return 0
    
def FE_cadre(poste):
    if poste in ['Cadre Commercial','Directeur Technique','Manager','Senior Manager','Tech Lead']:
        return True
    else:
        return False