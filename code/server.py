from langchain_community.llms import LlamaCpp
from sqlalchemy import create_engine, text
from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2

MODEL_PATH = "model/model.gguf"
DATABASE_URL = "postgresql://grajal:1234@localhost:5432/grajaldb"

engine = create_engine(DATABASE_URL)

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,       # Reduce el tamaño del contexto para pruebas
    n_threads=2,      # Prueba con menos hilos
    temperature=0.8,  # Ajusta la creatividad
    top_p=1.0,        # Incrementa la probabilidad acumulativa
    max_tokens=128,    # Reduce el número máximo de tokens generados
    use_mlock=False,  # Evita bloqueo de memoria
    verbose=True,
)

def execute_sql(query):
    try:
        # Conecta a tu base de datos
        connection = psycopg2.connect(
            database="grajaldb",
            user="grajal",
            password="1234",
            host="localhost",
            port="5432"
        )
        cursor = connection.cursor()
        cursor.execute(query)
        # Obtiene los resultados
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        connection.close()
        return {"columns": columns, "rows": rows}
    except Exception as e:
        return {"error": str(e)}

def get_table_schema():
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name, column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public';
        """))
        schema = {}
        for table_name, column_name in result:
            schema.setdefault(table_name, []).append(column_name)
    return schema

def prepare_prompt(question, schema, teamId):
    schema_description = "\n".join(
        f"Table: {table}, Columns: {', '.join(columns)}"
        for table, columns in schema.items()
    )
    prompt = f"""
    <|user|>
    You are a helpful assistant that generates SQL queries for a swimmer team coach. 
    Here is the database schema:
    {schema_description}


    {question}
    Generate the corresponding SQL query and only de SQL query, nothing more:
    <|end|>
    <|assistant|>
    """
    return prompt


# Inicializa FastAPI
app = FastAPI()

# Modelo para la solicitud
class QueryRequest(BaseModel):
    question: str

@app.post("/generate-sql/")
def generate_sql_endpoint(request: QueryRequest):
    # Obtener el esquema
    schema = get_table_schema()
    # Crear el prompt
    prompt = prepare_prompt(request.question, schema)
    # Generar la consulta SQL
    # prompt = "<|user|> What is 2 + 2? <|end|>\n<|assistant|>"
    sql_query = llm(prompt)
    if "SELECT" not in sql_query.upper():  # Validación básica
        return {"error": "Generated query is not a SELECT statement"}

    # print(sql_query)
    # result = execute_sql(sql_query)
    return {"sql_query": sql_query}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)