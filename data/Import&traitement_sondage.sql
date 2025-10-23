-- Active: 1761212299035@@127.0.0.1@5432@futurisys_db@public
CREATE TABLE extrait_sondage (
    a_quitte_l_entreprise VARCHAR(10),
    nombre_participation_pee INTEGER,
    nb_formations_suivies INTEGER,
    nombre_employee_sous_responsabilite INTEGER,
    code_sondage INTEGER PRIMARY KEY,
    distance_domicile_travail INTEGER,
    niveau_education INTEGER,
    domaine_etude VARCHAR(100),
    ayant_enfants BOOLEAN,
    frequence_deplacement VARCHAR(50),
    annees_depuis_la_derniere_promotion INTEGER,
    annes_sous_responsable_actuel INTEGER
);

COPY extrait_sondage
FROM 'C:\Users\Fabien\Desktop\OC\P5\FUTURISYS\data\extrait_sondage.csv'
DELIMITER ','
CSV HEADER;

UPDATE extrait_sondage
SET a_quitte_l_entreprise = CASE
    WHEN a_quitte_l_entreprise = 'Oui' THEN 'TRUE'
    WHEN a_quitte_l_entreprise = 'Non' THEN 'FALSE'
    ELSE NULL
END;
ALTER TABLE extrait_sondage
ALTER COLUMN a_quitte_l_entreprise TYPE BOOLEAN
USING a_quitte_l_entreprise::BOOLEAN;


UPDATE extrait_sondage
SET frequence_deplacement = CASE
    WHEN frequence_deplacement = 'Aucun' THEN '0'
    WHEN frequence_deplacement = 'Occasionnel' THEN '1'
    WHEN frequence_deplacement = 'Frequent' THEN '2'
    ELSE NULL
END;
ALTER TABLE extrait_sondage
ALTER COLUMN frequence_deplacement TYPE INTEGER
USING frequence_deplacement::INTEGER;





















UPDATE extrait_eval
SET eval_number = CAST(SUBSTRING(eval_number FROM 3) AS INTEGER);
ALTER TABLE extrait_eval
ALTER COLUMN eval_number TYPE INTEGER
USING eval_number::INTEGER;



UPDATE extrait_eval
SET augementation_salaire_precedente = LEFT(augementation_salaire_precedente, LENGTH(augementation_salaire_precedente) - 2);
ALTER TABLE extrait_eval
ALTER COLUMN augementation_salaire_precedente TYPE INTEGER
USING augementation_salaire_precedente::INTEGER;