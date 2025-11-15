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
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.elboudadi.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutageNo {
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) throws Exception {
        configureLogger();

        // === 1. API Key ===
        String apiKey = System.getenv("GEMINI-API-KEY");
        if (apiKey == null) {
            System.err.println("Définissez la variable d'environnement GEMINI_API_KEY");
            return;
        }

        // === 2. Modèle Gemini ===
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.1)
                .logRequestsAndResponses(true)
                .build();

        // === 3. Charger deux documents ===
        Path docAPath = Paths.get("docs/RAG.pdf");
        Path docBPath = Paths.get("docs/LangChain4j.pdf");

        Document docA = FileSystemDocumentLoader.loadDocument(docAPath.toString());
        Document docB = FileSystemDocumentLoader.loadDocument(docBPath.toString());

        // === 4. Embedding stores ===
        EmbeddingStore<TextSegment> embeddingStoreA = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> embeddingStoreB = new InMemoryEmbeddingStore<>();

        // === 5. Ingestion ===
        EmbeddingStoreIngestor.ingest(docA, embeddingStoreA);
        EmbeddingStoreIngestor.ingest(docB, embeddingStoreB);

        // === 6. Content retrievers ===
        ContentRetriever retrieverA = EmbeddingStoreContentRetriever.from(embeddingStoreA);
        ContentRetriever retrieverB = EmbeddingStoreContentRetriever.from(embeddingStoreB);

        // === 7. Query Router personnalisé (No RAG) ===
        QueryRouter router = new NoRagQueryRouter(model, retrieverA, retrieverB);

        // === 8. Retrieval Augmentor ===
        DefaultRetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // === 9. Assistant ===
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(retrievalAugmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // === 10. Console interactive ===
        Scanner scanner = new Scanner(System.in);
        System.out.println("=== Test No RAG Router - Tapez 'exit' pour quitter ===");
        while (true) {
            System.out.print("\nVotre question : ");
            String question = scanner.nextLine();
            if (question.equalsIgnoreCase("exit")) break;

            String reponse = assistant.chat(question);
            System.out.println("→ Réponse : " + reponse);
        }
    }

    // =====================================================
    // Inner class NoRagQueryRouter
    // =====================================================
    static class NoRagQueryRouter implements QueryRouter {

        private final ChatModel model;
        private final ContentRetriever retrieverA;
        private final ContentRetriever retrieverB;

        public NoRagQueryRouter(ChatModel model, ContentRetriever retrieverA, ContentRetriever retrieverB) {
            this.model = model;
            this.retrieverA = retrieverA;
            this.retrieverB = retrieverB;
        }

        @Override
        public List<ContentRetriever> route(Query query) {
            String question = query.text();

            String prompt = "La question concerne-t-elle l'intelligence artificielle ? " +
                    "Réponds uniquement par 'oui', 'non' ou 'peut-être'.\nQuestion : " + question;

            String answer = model.chat(prompt).toLowerCase(Locale.ROOT);

            System.out.println("Réponse du routeur : " + answer);

            if (answer.contains("non")) {
                System.out.println("➡ Pas de RAG (réponse directe du modèle).");
                return Collections.emptyList();
            } else {
                System.out.println("➡ Utilisation du RAG (Doc A + Doc B).");
                return Arrays.asList(retrieverA, retrieverB);
            }
        }
    }
}