CREATE TABLE inputs (
    id_input SERIAL PRIMARY KEY,
    employe_id INT REFERENCES employes(id_employee) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT NOW(),

    heure_supplementaires BOOLEAN,
    age INTEGER,
    FE_ratio_ancienneté FLOAT,
    FE_cadre BOOLEAN,
    frequence_deplacement INTEGER,
    FE_duree_moy_exp_precedentes FLOAT,
    FE_ratio_evolution FLOAT,
    niveau_education INTEGER,
    FE_reste_plus_longtemps BOOLEAN,
    poste TEXT,
    statut_marital TEXT
)
