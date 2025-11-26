from llm_agent import FaceMatchAgent

agent = FaceMatchAgent(
    onnx_model_path=r"F:\github\trabalho-extracao-imagens-pucrj\models\arc.onnx",
    llm_model_name=r"F:\github\trabalho-extracao-imagens-pucrj\Llama-3.1-8B-Instruct"
)

result = agent.analisar_imagens(
    r"F:\github\trabalho-extracao-imagens-pucrj\imgs/foto_roberto_01.jpg",
    r"F:\github\trabalho-extracao-imagens-pucrj\imgs/foto_roberto_02.jpg",
    "Valide se estas imagens possuem um rosto e me diga se são a mesma pessoa."
)

print("Resultado:", result)
