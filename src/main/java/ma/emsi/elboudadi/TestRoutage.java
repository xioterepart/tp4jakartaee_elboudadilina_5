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
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutage {

    interface Assistant {
        String chat(String message);
    }

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
            System.err.println("⚠️ Définissez la variable d'environnement GEMINI_API_KEY");
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

        Map<ContentRetriever, String> retrieverMap = new HashMap<>();
        retrieverMap.put(retrieverA, "Doc A : RAG et Intelligence Artificielle");
        retrieverMap.put(retrieverB, "Doc B : LangChain4j");

        // === 7. Query Router (new builder syntax) ===
        QueryRouter router = LanguageModelQueryRouter.builder()
                .chatModel(model)
                .retrieverToDescription(retrieverMap)  // ← use this method
                .build();

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
        System.out.println("=== Test Routage - Tapez 'exit' pour quitter ===");
        while (true) {
            System.out.print("\nVotre question : ");
            String question = scanner.nextLine();
            if (question.equalsIgnoreCase("exit")) break;

            String reponse = assistant.chat(question);
            System.out.println("→ Réponse : " + reponse);
        }
    }
}
