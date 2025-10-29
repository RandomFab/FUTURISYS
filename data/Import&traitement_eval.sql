-- Active: 1761212299035@@127.0.0.1@5432@futurisys_db@public
CREATE TABLE extrait_eval(
    satisfaction_employee_environnement integer,
    note_evaluation_precedente integer,
    niveau_hierarchique_poste integer,
    satisfaction_employee_nature_travail integer,
    satisfaction_employee_equipe integer,
    satisfaction_employee_equilibre_pro_perso integer,
    eval_number character(10) NOT NULL,
    note_evaluation_actuelle integer,
    heure_supplementaires character(10),
    augementation_salaire_precedente character(10),
    PRIMARY KEY(eval_number)
);

COPY extrait_eval
FROM 'C:\Users\Fabien\Desktop\OC\P5\FUTURISYS\data\extrait_eval.csv'
DELIMITER ','
CSV HEADER;

UPDATE extrait_eval
SET eval_number = CAST(SUBSTRING(eval_number FROM 3) AS INTEGER);
ALTER TABLE extrait_eval
ALTER COLUMN eval_number TYPE INTEGER
USING eval_number::INTEGER;

UPDATE extrait_eval
SET heure_supplementaires = CASE
    WHEN heure_supplementaires = 'Oui' THEN 'TRUE'
    WHEN heure_supplementaires = 'Non' THEN 'FALSE'
    ELSE NULL
END;
ALTER TABLE extrait_eval
ALTER COLUMN heure_supplementaires TYPE BOOLEAN
USING heure_supplementaires::BOOLEAN;

UPDATE extrait_eval
SET augementation_salaire_precedente = LEFT(augementation_salaire_precedente, LENGTH(augementation_salaire_precedente) - 2);
ALTER TABLE extrait_eval
ALTER COLUMN augementation_salaire_precedente TYPE INTEGER
USING augementation_salaire_precedente::INTEGER;