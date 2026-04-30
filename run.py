from flask import Flask, request, jsonify, send_from_directory
from SimilarityOFTA import calcular_score_complexo
import os


app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    """
    Serve o frontend específico que utiliza exclusivamente SimilarityOFTA.py.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, "SimilarityOFTA_Frontend.html")


@app.route("/similarity-ofta", methods=["POST"])
def similarity_ofta():
    """
    Endpoint JSON que chama diretamente SimilarityOFTA.py.

    Espera um corpo JSON:
    {
        "nome1": "string",
        "nome2": "string"
    }

    Retorna o mesmo dicionário de calcular_score_complexo, por exemplo:
    {
        "final": 0.79,
        "driver": "Fonética (Som)",
        "via": "Literal (dígito por dígito)",
        "vetores": { ... }
    }
    """
    data = request.get_json(silent=True) or {}
    nome1 = data.get("nome1", "") or ""
    nome2 = data.get("nome2", "") or ""

    if not nome1 or not nome2:
        return jsonify({"error": "Campos 'nome1' e 'nome2' são obrigatórios."}), 400

    res = calcular_score_complexo(nome1, nome2)
    return jsonify(res)


if __name__ == "__main__":
    # Algumas máquinas Windows ou políticas corporativas podem bloquear certas portas
    # (como 8000) por questão de permissão de soquete. Vamos usar a porta padrão
    # do Flask (5000) e desativar o modo debug para evitar múltiplas tentativas
    # de bind na mesma porta.
    host = "127.0.0.1"
    port = 5000
    url = f"http://{host}:{port}/"
    print(f"Servidor Flask iniciado. Acesse o frontend em: {url}")
    app.run(host=host, port=port, debug=False)

