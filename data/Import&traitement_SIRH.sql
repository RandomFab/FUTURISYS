-- Active: 1761212299035@@127.0.0.1@5432@futurisys_db@public
CREATE TABLE extrait_SIRH (
    id_employee INTEGER PRIMARY KEY,
    age INTEGER,
    genre VARCHAR(20),
    revenu_mensuel INTEGER,
    statut_marital VARCHAR(30),
    departement VARCHAR(50),
    poste VARCHAR(50),
    nombre_experiences_precedentes INTEGER,
    nombre_heures_travailless INTEGER,
    annee_experience_totale INTEGER,
    annees_dans_l_entreprise INTEGER,
    annees_dans_le_poste_actuel INTEGER
);

COPY extrait_SIRH
FROM 'C:\Users\Fabien\Desktop\OC\P5\FUTURISYS\data\extrait_SIRH.csv'
DELIMITER ','
CSV HEADER;

UPDATE extrait_SIRH
SET genre = CASE
    WHEN genre = 'F' THEN 'TRUE'
    WHEN genre = 'M' THEN 'FALSE'
    ELSE NULL
END;
ALTER TABLE extrait_SIRH
ALTER COLUMN genre TYPE BOOLEAN
USING genre::BOOLEAN;
