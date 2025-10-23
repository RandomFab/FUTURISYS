CREATE TABLE outputs (
    id_output SERIAL PRIMARY KEY,
    probabilite FLOAT,
    PREDICT BOOLEAN,

    id_input_employee INTEGER NOT NULL,

    FOREIGN KEY (id_input_employee) REFERENCES inputs(id_employee)
)