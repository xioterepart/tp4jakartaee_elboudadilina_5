package ma.emsi.elboudadi;

/**
 * Interface commune pour les assistants IA utilisés dans les différents tests.
 * 
 * Cette interface est implémentée automatiquement par LangChain4j via AiServices.
 * Elle définit le contrat de base pour l'interaction avec un assistant conversationnel.
 */
public interface Assistant {
    /**
     * Envoie un message à l'assistant et reçoit une réponse.
     * 
     * @param userMessage Le message de l'utilisateur
     * @return La réponse générée par l'assistant
     */
    String chat(String userMessage);
}