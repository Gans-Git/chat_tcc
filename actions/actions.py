# actions.py

from transformers import pipeline
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List
from rasa_sdk.forms import FormValidationAction
from rasa_sdk.types import DomainDict
# AQUI ESTÁ A CORREÇÃO PRINCIPAL: Adicionamos SlotSet à lista de importação
from rasa_sdk.events import SlotSet, ActionExecutionRejected, EventType, FollowupAction


# ==============================================================================
# SEÇÃO DO FORMULÁRIO DE AUTOAVALIAÇÃO
# ==============================================================================


class ActionAutoavaliacaoForm(Action):
    def name(self) -> Text:
        return "action_autoavaliacao_form"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Pega as respostas dos slots. Elas vêm como texto ("0", "1", etc.)
        sente_nervoso_str = tracker.get_slot("sente_nervoso")
        nao_consegue_parar_str = tracker.get_slot("nao_consegue_parar_preocupacao")

        # Converte as respostas para números inteiros para poder somar
        try:
            pontuacao1 = int(sente_nervoso_str)
            pontuacao2 = int(nao_consegue_parar_str)
            pontuacao_total = pontuacao1 + pontuacao2
        except (ValueError, TypeError):
            # Caso algo dê errado na conversão, assume a pontuação como 0
            pontuacao_total = 0

        dispatcher.utter_message(response="utter_submit_autoavaliacao")

        # Lógica para dar o resultado com base na pontuação
        if pontuacao_total >= 3:
            mensagem_resultado = (
                "Com base nas suas respostas, pode ser um bom momento para conversar com um profissional "
                "de saúde sobre como você tem se sentido. Cuidar da mente é tão importante quanto cuidar do corpo. "
                "Não hesite em procurar ajuda."
            )
        else:
            mensagem_resultado = (
                "Suas respostas sugerem um nível baixo de sintomas de ansiedade recentes. "
                "Continue praticando o autocuidado e prestando atenção aos seus sentimentos. "
                "Se algo mudar, lembre-se que buscar ajuda é sempre uma opção."
            )

        dispatcher.utter_message(text=mensagem_resultado)

        # Limpa os slots para que o formulário possa ser refeito no futuro
        return [SlotSet("sente_nervoso", None), SlotSet("nao_consegue_parar_preocupacao", None)]


# ==============================================================================
# SEÇÃO DE ANÁLISE DE SENTIMENTO
# ==============================================================================

# Pipeline de detecção de emoções em português
detector_emocao = pipeline(
    "text-classification",
    model="pysentimiento/bert-pt-emotion",
    tokenizer="pysentimiento/bert-pt-emotion"
)

# Mapeamento das emoções do modelo para português
TRADUCAO_EMOCOES = {
    "joy": "alegria",
    "sadness": "tristeza",
    "anger": "raiva",
    "fear": "medo",
    "surprise": "surpresa",
    "neutral": "neutro"
}

def detectar_emocao(texto: str) -> Text:
    try:
        resultados = detector_emocao(texto)
        if resultados:
            emocao_predita = resultados[0]["label"]
            return TRADUCAO_EMOCOES.get(emocao_predita, "neutro")
    except Exception as e:
        print(f"Erro ao detectar emoção: {e}")
    return "neutro"

class ActionAnalisarESugerir(Action):
    def name(self) -> Text:
        return "action_analisar_e_sugerir"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        texto_usuario = tracker.latest_message.get("text")
        emocao = detectar_emocao(texto_usuario)

        if emocao == "neutro":
            # --- Caminho para Emoção NEUTRA ---
            dica_neutra = "Às vezes, os sentimentos não são tão claros, e tudo bem. Uma dica que pode ajudar a organizar a mente é simplesmente listar 3 coisas pelas quais você é grato(a) hoje, por menores que sejam."
            mensagem_neutra = f"Entendido. {dica_neutra}"
            dispatcher.utter_message(text=mensagem_neutra)

        else:
            # --- Caminho para TODAS as outras Emoções (Tristeza, Medo, etc.) ---
            dicas = {
                "tristeza": "Sinto muito que você esteja passando por isso. Nesses momentos, ser gentil consigo mesmo é o mais importante. Que tal tentar ouvir uma música que te conforta ou escrever sobre seus sentimentos sem julgamento?",
                "medo": "É normal sentir medo. Tente encontrar um cantinho seguro e prestar atenção na sua respiração, sentindo o ar entrar e sair. Isso pode trazer uma sensação de calma.",
                "raiva": "A raiva é uma emoção forte e que precisa de espaço. Talvez se afastar um pouco da situação e focar na sua respiração por alguns minutos possa te ajudar a ver as coisas com mais clareza.",
                "alegria": "Que maravilha sentir essa alegria! Fico feliz por você. Uma dica para cultivá-la é compartilhar essa energia boa com alguém ou fazer algo que você ama.",
            }
        dica = dicas.get(emocao, "Lembre-se sempre de ser gentil com você mesmo. Pequenas pausas durante o dia fazem uma grande diferença.")

        mensagem_acolhimento_e_dica = (
            f"Obrigado por compartilhar. Pelo que você contou, parece haver um sentimento de {emocao}, e é totalmente compreensível se sentir assim. "
            f"Uma pequena prática de autocuidado que talvez possa ajudar: {dica}"
        )
        dispatcher.utter_message(text=mensagem_acolhimento_e_dica)

        # --- PARTE DA "PONTE INFORMACIONAL" (Continua igual) ---
        mapa_emocao_para_topico = {
            "tristeza": {"nome_topico": "Depressão", "intent_payload": "/buscar_info_depressao"},
            "medo": {"nome_topico": "Ansiedade", "intent_payload": "/buscar_info_ansiedade"}
        }

        if emocao in mapa_emocao_para_topico:
            topico = mapa_emocao_para_topico[emocao]
            botoes = [
                {"title": f"Sim, quero saber sobre {topico['nome_topico']}", "payload": topico['intent_payload']},
                {"title": "Não, obrigado(a)", "payload": "/negar_sugestao"}
            ]
            mensagem_ponte = (
                f"\nA propósito, sentimentos de {emocao} quando persistentes, às vezes estão relacionados a quadros de {topico['nome_topico']}. "
                "Isto não é um diagnóstico. Gostaria de aprender mais sobre isso?"
            )
            dispatcher.utter_message(text=mensagem_ponte, buttons=botoes)

        return []

# ==============================================================================
# SEÇÃO DO "DESPACHANTE" DE MENU
# ==============================================================================
    
class ActionHandleMenuChoice(Action):
    def name(self) -> Text:
        return "action_handle_menu_choice"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        opcao = tracker.get_slot("opcao")
        
        if not opcao:
            dispatcher.utter_message(response="utter_saudacao")
            return []

        if opcao == "desabafo":
            dispatcher.utter_message(response="utter_desabafo")
        elif opcao == "aprendizado":
            dispatcher.utter_message(response="utter_aprendizado")
        elif opcao == "dica_autocuidado":
            dispatcher.utter_message(response="utter_dica_generica")
        elif opcao == "autoavaliacao":
            return [FollowupAction("action_autoavaliacao_form")]
        
        return []