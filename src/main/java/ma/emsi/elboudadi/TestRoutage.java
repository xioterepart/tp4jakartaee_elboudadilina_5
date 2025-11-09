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
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;


import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
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

        // === 8. Default QueryRouter combining docs + web search ===
        DefaultQueryRouter router = new DefaultQueryRouter(retrieverA, retrieverB, webRetriever);

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
}
