package ma.emsi.elboudadi;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 3, 4 et 5 - Routage et Recherche Web
 * 
 * Ce test démontre plusieurs fonctionnalités avancées du RAG :
 * 
 * Test 3 - Routage entre sources :
 * - Utilisation de plusieurs documents (RAG.pdf, LangChain4j.pdf)
 * - Routage intelligent pour choisir la bonne source
 * 
 * Test 4 - Décision d'utiliser le RAG ou non :
 * - Implémentation d'un QueryRouter personnalisé (NoRagQueryRouter)
 * - Le LM décide si la question nécessite le RAG
 * - Si la question ne concerne pas l'IA, réponse directe sans RAG
 * 
 * Test 5 - Recherche Web :
 * - Intégration de Tavily pour la recherche sur le Web
 * - Combinaison de sources locales et Web
 */
public class RagAvecWeb {

    // Utilise l'interface Assistant commune définie dans Assistant.java

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) throws Exception {
        configureLogger();

        // === 1. API Key for Gemini ===
        String geminiApiKey = System.getenv("GEMINI-API-KEY");
        if (geminiApiKey == null) {
            System.err.println("⚠️ Define environment variable GEMINI_API_KEY");
            return;
        }

        // === 2. Gemini ChatModel ===
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.1)
                .logRequestsAndResponses(true)
                .build();

        // === 3. Load documents ===
        Path docAPath = Paths.get("docs/RAG.pdf");
        Path docBPath = Paths.get("docs/LangChain4j.pdf");

        Document docA = FileSystemDocumentLoader.loadDocument(docAPath.toString());
        Document docB = FileSystemDocumentLoader.loadDocument(docBPath.toString());

        // === 4. Embedding stores ===
        EmbeddingStore<TextSegment> embeddingStoreA = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> embeddingStoreB = new InMemoryEmbeddingStore<>();

        // === 5. Ingest documents into embedding stores ===
        EmbeddingStoreIngestor.ingest(docA, embeddingStoreA);
        EmbeddingStoreIngestor.ingest(docB, embeddingStoreB);

        // === 6. Content retrievers for documents ===
        ContentRetriever retrieverA = EmbeddingStoreContentRetriever.from(embeddingStoreA);
        ContentRetriever retrieverB = EmbeddingStoreContentRetriever.from(embeddingStoreB);

        // === 7. Tavily web search setup ===
        String tavilyApiKey = System.getenv("TAVILY_KEY");
        if (tavilyApiKey == null) {
            System.err.println("⚠️ Define environment variable TAVILY_KEY");
            return;
        }

        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyApiKey)
                .build();

        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .maxResults(3)     // use maxResults instead of topK
                .build();

        // === 8. Custom QueryRouter - Décide si le RAG doit être utilisé ===
        // Ce routeur personnalisé analyse la question pour déterminer si elle concerne l'IA.
        // Si non, il retourne une liste vide pour désactiver le RAG.
        QueryRouter router = new NoRagQueryRouter(model, retrieverA, retrieverB);

        // === 9. RetrievalAugmentor ===
        DefaultRetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // === 10. Assistant with retrieval augmentor ===
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // === 11. Interactive console ===
        Scanner scanner = new Scanner(System.in);
        System.out.println("=== Test RAG + Web Search (Tavily) – type 'exit' to quit ===");

        while (true) {
            System.out.print("\nYour question: ");
            String question = scanner.nextLine();
            if ("exit".equalsIgnoreCase(question)) break;

            String response = assistant.chat(question);
            System.out.println("→ Response: " + response);
        }

        scanner.close();
    }

    /**
     * Implémentation personnalisée de QueryRouter qui décide dynamiquement
     * s'il faut utiliser le RAG ou non en fonction du contenu de la question.
     * 
     * Cette classe utilise un modèle de langage pour analyser la question et décider
     * si elle est liée à l'IA. Si ce n'est pas le cas, elle retourne une liste vide
     * pour indiquer qu'aucun retriever ne doit être utilisé (pas de RAG).
     */
    static class NoRagQueryRouter implements QueryRouter {
        private static final Logger LOGGER = Logger.getLogger(NoRagQueryRouter.class.getName());
        
        private final ChatModel model;
        private final ContentRetriever retrieverA;
        private final ContentRetriever retrieverB;

        /**
         * Constructeur du routeur personnalisé.
         * 
         * @param model Modèle de langage pour l'analyse des requêtes
         * @param retrieverA Premier retriever de contenu (Document A)
         * @param retrieverB Deuxième retriever de contenu (Document B)
         */
        public NoRagQueryRouter(ChatModel model, ContentRetriever retrieverA, ContentRetriever retrieverB) {
            this.model = Objects.requireNonNull(model, "Le modèle de langage ne peut pas être null");
            this.retrieverA = Objects.requireNonNull(retrieverA, "Le premier retriever ne peut pas être null");
            this.retrieverB = Objects.requireNonNull(retrieverB, "Le deuxième retriever ne peut pas être null");
        }

        /**
         * Détermine si le RAG doit être utilisé pour cette requête.
         * 
         * @param query La requête à analyser
         * @return Une liste de ContentRetriever à utiliser, ou une liste vide pour désactiver le RAG
         */
        @Override
        public List<ContentRetriever> route(Query query) {
            try {
                String question = Objects.requireNonNull(query, "La requête ne peut pas être null").text();
                if (question == null || question.trim().isEmpty()) {
                    LOGGER.warning("La question est vide, utilisation du RAG par défaut");
                    return Arrays.asList(retrieverA, retrieverB);
                }

                // Création du prompt pour déterminer si la question concerne l'IA
                String prompt = "La question suivante concerne-t-elle l'intelligence artificielle, " +
                              "le machine learning, le deep learning, les modèles de langage, " +
                              "ou des sujets techniques similaires ? " +
                              "Réponds uniquement par 'oui', 'non' ou 'peut-être'.\n\n" +
                              "Question : " + question;

                // Appel au modèle pour l'analyse
                String answer = model.chat(prompt).toLowerCase(Locale.ROOT).trim();
                LOGGER.fine("🔍 Réponse du routeur : " + answer);

                // Décision de routage
                if (answer.startsWith("non")) {
                    LOGGER.info("➡ Pas de RAG (réponse directe du modèle).");
                    return Collections.emptyList();
                } else {
                    LOGGER.info("➡ Utilisation du RAG (Doc A + Doc B).");
                    return Arrays.asList(retrieverA, retrieverB);
                }
            } catch (Exception e) {
                LOGGER.log(Level.SEVERE, "Erreur lors du routage de la requête : " + e.getMessage(), e);
                // En cas d'erreur, on utilise le RAG par défaut
                return Arrays.asList(retrieverA, retrieverB);
            }
        }
    }
}
