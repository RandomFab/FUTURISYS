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