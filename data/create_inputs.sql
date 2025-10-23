CREATE TABLE inputs (

    id_employee INTEGER NOT NULL UNIQUE,

    heure_supplementaires BOOLEAN,
    age INTEGER,
    annees_dans_l_entreprise INTEGER,
    frequence_deplacement INTEGER ,
    nombre_experiences_precedentes INTEGER,
    annees_dans_le_poste_actuel INTEGER,
    annee_experience_totale INTEGER,
    niveau_education INTEGER,
    poste VARCHAR(50),
    statut_marital VARCHAR(30),
    PRIMARY KEY (id_employee),

    FOREIGN KEY (id_employee) REFERENCES extrait_sirh(id_employee),
    FOREIGN KEY (id_employee) REFERENCES extrait_eval(eval_number),
    FOREIGN KEY (id_employee) REFERENCES extrait_sondage(code_sondage)

);

INSERT INTO inputs (
    id_employee,
    heure_supplementaires, age, annees_dans_l_entreprise,
    frequence_deplacement, nombre_experiences_precedentes,
    annees_dans_le_poste_actuel, annee_experience_totale,
    niveau_education, poste, statut_marital
)
SELECT
    sirh.id_employee,
    eval.heure_supplementaires,
    sirh.age,
    sirh.annees_dans_l_entreprise,
    sondage.frequence_deplacement,
    sirh.nombre_experiences_precedentes,
    sirh.annees_dans_le_poste_actuel,
    sirh.annee_experience_totale,
    sondage.niveau_education,
    sirh.poste,
    sirh.statut_marital
FROM extrait_sirh AS sirh
JOIN extrait_eval AS eval ON sirh.id_employee = eval.eval_number
JOIN extrait_sondage AS sondage ON sirh.id_employee = sondage.code_sondage;
