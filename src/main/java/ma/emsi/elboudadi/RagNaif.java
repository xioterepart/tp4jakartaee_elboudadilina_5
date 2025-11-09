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

public class RagNaif {

    // Interface Assistant, implémentée automatiquement par LangChain4j
    interface Assistant {
        String chat(String userMessage);
    }

    public static void main(String[] args) throws Exception {

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
                .build();

        // === Étape 3 : Chargement du document à indexer ===
        Path chemin = Paths.get("docs/RAG.pdf"); // chemin relatif ou absolu
        Document document = FileSystemDocumentLoader.loadDocument(chemin.toString());

        // === Étape 4 : Création de la base vectorielle (en mémoire) ===
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        // === Étape 5 : Calcul des embeddings et ingestion ===
        // Utilise automatiquement le modèle d’embedding par défaut
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
