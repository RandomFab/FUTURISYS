CREATE TABLE outputs (
    id_output SERIAL PRIMARY KEY,

    id_input INT REFERENCES inputs(id_input) ON DELETE CASCADE,

    probabilite FLOAT,
    predict BOOLEAN,
    timestamp TIMESTAMP DEFAULT NOW()
)