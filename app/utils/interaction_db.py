from sqlalchemy import text

def get_employe(conn, id_employe):
    query = text("SELECT * FROM employes WHERE id_employee = :id")
    data = conn.execute(query, {'id' : id_employe})
    row = data.fetchone()

    if row: 
        data_dict = dict(row._mapping)
        return data_dict
    else:
        return {'message' : 'Aucun employé trouvé'}


def post_input(conn,data_dict_for_model):
    insert_query = text("""
            INSERT INTO inputs (
                employe_id,
                heure_supplementaires,
                age,
                FE_ratio_ancienneté,
                FE_cadre,
                frequence_deplacement,
                FE_duree_moy_exp_precedentes,
                FE_ratio_evolution,
                niveau_education,
                FE_reste_plus_longtemps,
                poste,
                statut_marital
            ) VALUES (
                :id_employe,
                :heure_supplementaires,
                :age,
                :FE_ratio_ancienneté,
                :FE_cadre,
                :frequence_deplacement,
                :FE_duree_moy_exp_precedentes,
                :FE_ratio_evolution,
                :niveau_education,
                :FE_reste_plus_longtemps,
                :poste,
                :statut_marital
            )
            RETURNING id_input
        """)
    result = conn.execute(insert_query,{
        "id_employe": data_dict_for_model["id_employee"],
        "heure_supplementaires": data_dict_for_model["heure_supplementaires"],
        "age": data_dict_for_model["age"],
        "FE_ratio_ancienneté": data_dict_for_model["FE_ratio_ancienneté"],
        "FE_cadre": data_dict_for_model["FE_cadre"],
        "frequence_deplacement": data_dict_for_model["frequence_deplacement"],
        "FE_duree_moy_exp_precedentes": data_dict_for_model["FE_duree_moy_exp_precedentes"],
        "FE_ratio_evolution": data_dict_for_model["FE_ratio_evolution"],
        "niveau_education": data_dict_for_model["niveau_education"],
        "FE_reste_plus_longtemps": data_dict_for_model["FE_reste_plus_longtemps"],
        "poste": data_dict_for_model["poste"],
        "statut_marital": data_dict_for_model["statut_marital"]
    })

    id_input = result.scalar()
    return id_input

def post_output(conn,id_input,proba,predict):
    insert_query = text("""INSERT INTO outputs(
                            id_input,
                            probabilite,
                            predict
                            ) VALUES (
                            :id_input,
                            :probabilite,
                            :predict)
                            """)
    conn.execute(insert_query,{"id_input" : id_input,
                                "probabilite" : float(proba),
                                "predict": bool(predict)})