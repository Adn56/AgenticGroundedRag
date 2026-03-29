# Agentic Grounded RAG für Next-POI-Vorhersage

## Überblick
Dieses Repository enthält die Implementierung eines Reasoning-First-Ansatzes zur Next-POI-Vorhersage.  
Im Gegensatz zu klassischen, selektionsbasierten Verfahren wird die Vorhersage nicht als Auswahlproblem innerhalb eines festen Kandidatenraums formuliert. Stattdessen wird zunächst eine semantische Hypothese über den intendierten Zielzustand generiert, auf deren Basis reale Points of Interest (POIs) über Retrieval- und Re-Ranking-Schritte identifiziert werden.

Die Pipeline kombiniert Large Language Models, embedding-basiertes Retrieval sowie agentisches Re-Ranking, um auch bislang unbeobachtete POIs vorhersagen zu können.

---

## Repository-Struktur

Das Repository ist modular aufgebaut und bildet die gesamte Pipeline ab:

- **Dataset/**  
  Enthält Datensätze, Splits sowie vorberechnete Strukturen wie Embeddings und FAISS-Indizes.

- **results/**  
  Speichert finale Ergebnisse und Modellvorhersagen.

- **baseline/**  
  Ergebnisse der Vergleichsmodelle:
    - KNN-basierte sequenzielle Empfehlung
    - LLM Zero-Shot Baseline
    - Distanzbasierte Baseline

- **src/**  
  Kern der Implementierung:

    - **main/**  
      Zentrale Pipeline (Hypothesengenerierung, Retrieval, Re-Ranking)

    - **preprocessing/**  
      Datenaufbereitung und Bereinigung

    - **representations/**  
      Erstellung semantischer POI-Repräsentationen und Embeddings

    - **splits/**  
      Generierung von Trainings-, Validierungs- und Testdaten

    - **baseline/**  
      Baseline-Skripte

    - **analyse/**  
      Evaluationsskripte (Rankingmetriken und Beyond-Accuracy-Analysen)

    - **validation/**  
      Hyperparameter-Tuning und Prompt-Ablationsstudien

    - **archive/**  
      Ältere oder experimentelle Varianten der Pipeline

---

## Methode

Die vorgeschlagene Pipeline folgt einem Reasoning-First-Paradigma:

1. Hypothesengenerierung  
   Ein Large Language Model erzeugt eine semantische Beschreibung des intendierten nächsten POIs.

2. Embedding-basiertes Retrieval  
   Kandidaten werden über Cosine Similarity im Embeddingraum (FAISS) abgerufen.

3. Distanzbasiertes Re-Ranking  
   Die abgerufenen Kandidaten werden anhand räumlicher Distanzinformationen neu priorisiert.

4. Agentisches Re-Ranking  
   Finale Bewertung basierend auf kontextueller Plausibilität.

Der Ansatz entkoppelt semantische Zielmodellierung von der Kandidatenauswahl und ermöglicht dadurch Generalisierung über beobachtete Daten hinaus.

