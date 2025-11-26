from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from vision_tools import FaceTools


class FaceMatchAgent:
    def __init__(self,
                 onnx_model_path: str,
                 llm_model_name: str = "Llama-3.1-8B-Instruct",
                 device: str = None):
       
        self.face_tools = FaceTools(onnx_model_path)
       
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)

    def _generate_llm_response(self, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"{system_prompt}\n\nUsuário: {user_prompt}\nAssistente:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        # somente a parte depois de "Assistente:"
        if "Assistente:" in text:
            text = text.split("Assistente:")[-1].strip()
        return text

    
    def analisar_imagens(self, image1_path: str, image2_path: str, user_prompt: str):

        det1 = self.face_tools.detectar_rosto(image1_path)
        det2 = self.face_tools.detectar_rosto(image2_path)

        #LLM explica o motivo em caso de falha
        if not det1["ok"] or not det2["ok"]:
            system_prompt = (
                "Você é um assistente especializado em análise de imagens. "
                "Explique claramente o resultado das ferramentas, "
                "ajudando o usuário a entender por que não é possível comparar as imagens."
            )

            tools_summary = (
                f"Resultado detectar_rosto imagem1: {det1}\n"
                f"Resultado detectar_rosto imagem2: {det2}\n"
            )

            llm_input = (
                f"{user_prompt}\n\n"
                f"{tools_summary}\n"
                "Explique por que não foi possível continuar a verificação facial."
            )

            answer = self._generate_llm_response(system_prompt, llm_input)

            return {
                "deteccao1": det1,
                "deteccao2": det2,
                "comparacao": None,
                "resposta_llm": answer,
            }

        # 2) Se ambos têm rosto → comparar
        comp = self.face_tools.comparar_rostos(image1_path, image2_path)

        system_prompt = (
            "Você é um agente orquestrador que interpreta resultados de modelos "
            "de visão computacional. Explique a conclusão de forma clara e educada "
            "com base nos resultados recebidos."
        )

        tools_summary = (
            f"detectar_rosto imagem1: {det1}\n"
            f"detectar_rosto imagem2: {det2}\n"
            f"comparar_rostos: {comp}\n"
        )

        llm_input = (
            f"{user_prompt}\n\n"
            f"Resultados das ferramentas:\n{tools_summary}\n"
            "Explique se são a mesma pessoa e qual é a confiança com base na similaridade."
        )

        answer = self._generate_llm_response(system_prompt, llm_input)

        return {
            "deteccao1": det1,
            "deteccao2": det2,
            "comparacao": comp,
            "resposta_llm": answer,
        }
