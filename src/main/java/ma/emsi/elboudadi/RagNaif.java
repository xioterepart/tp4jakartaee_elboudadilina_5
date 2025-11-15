package ma.emsi.elboudadi;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Test 1 - RAG Naïf
 * 
 * Implémentation de base du RAG (Retrieval-Augmented Generation) avec décomposition
 * explicite de toutes les étapes :
 * 
 * Phase 1 - Ingestion :
 * - Chargement du document PDF
 * - Découpage en segments (chunks)
 * - Création des embeddings
 * - Stockage dans un magasin vectoriel en mémoire
 * 
 * Phase 2 - Récupération et génération :
 * - Création d'un ContentRetriever
 * - Configuration de l'assistant avec mémoire
 * - Interaction en mode console
 */
public class RagNaif {

    // Utilise l'interface Assistant commune définie dans Assistant.java

    /**
     * Configure le logger pour afficher les détails des requêtes/réponses
     * envoyées à l'API du modèle de langage.
     */
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);

        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);

        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) throws Exception {

        // === Étape 0 : Activation du logging ===
        configureLogger();

        // === Étape 1 : Configuration de l'API ===
        String llmKey = System.getenv("GEMINI-API-KEY");
        if (llmKey == null) {
            System.err.println("Veuillez définir la variable d'environnement GEMINI-API-KEY");
            return;
        }

        // === Étape 2 : Création du modèle LLM Gemini ===
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(llmKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.1)
                .logRequestsAndResponses(true)
                .build();

        // === Étape 3 : Chargement du document à indexer ===
        Path chemin = Paths.get("docs/RAG.pdf"); // chemin relatif ou absolu
        Document document = FileSystemDocumentLoader.loadDocument(chemin.toString());

        // === Étape 4 : Création de la base vectorielle (en mémoire) ===
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // === Étape 5 : Calcul des embeddings et ingestion ===
        EmbeddingStoreIngestor.ingest(document, embeddingStore);

        // === Étape 6 : Création du récupérateur de contenu (retriever) ===
        var retriever = EmbeddingStoreContentRetriever.from(embeddingStore);

        // === Étape 7 : Construction de l'assistant RAG ===
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(retriever)
                .build();

        // === Étape 8 : Console interactive ===
        try (Scanner scanner = new Scanner(System.in)) {
            System.out.println("=== Assistant RAG - tapez 'exit' pour quitter ===");
            while (true) {
                System.out.print("\nVotre question : ");
                String question = scanner.nextLine();
                if (question.equalsIgnoreCase("exit")) break;

                String reponse = assistant.chat(question);
                System.out.println("→ Réponse : " + reponse);
            }
        }
    }
}
