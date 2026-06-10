Betreff: Erdbeere-Clustering – Vergleich von 9 Strategien zur Experten-Bewertung

Hallo zusammen,

aufbauend auf den von euch vorgegebenen Beispiel-Centroiden (Datei `Erdbeer Clustering Sensorik Vorgabe.xlsx`) habe ich die Recipe-Cluster auf mehrere unterschiedliche Arten konstruiert. Ziel ist, dass ihr mir sagt, welche Strategie der Gruppierung der Panelisten (Free-Sorting) am nächsten kommt.

Im Anhang findet ihr die aktualisierte Excel-Datei `cluster_assignments_expert_seeded_all_strategies.xlsx`. Bitte startet mit dem ersten Tabellenblatt `Strategy_Comparison` – dort steht pro Zeile ein Rezept und pro Spalte eine Strategie. Die Spalten `Target_Recipe?` und `Expert_Intended_Cluster` markieren eure Vorgabe-Rezepte, sodass ihr direkt seht, in welches Cluster jedes Rezept unter jeder Strategie fällt.

Die Strategien im Überblick:

Die ersten sechs nutzen jeweils eure Beispiel-Centroide, unterscheiden sich aber darin, wie der Centroid berechnet wird:

- S1 / S4 – Target-Rezepte (Mittelwert / Median): Centroid aus den von euch genannten Ziel-Rezepten je Cluster.
- S2 / S5 – Inhaltsstoffe (Mittelwert / Median): Centroid aus allen Rezepten, die den charakteristischen Rohstoff des Clusters enthalten.
- S3 – Hybrid: Ziel-Rezepte, wo vorhanden; sonst inhaltsstoff-basiert.
- S6 – Kontrast (neu): wie S1, aber der allgemeine „fruchtig-süße“ Hintergrund, der in fast jedem Rezept steckt, wird abgezogen – so wird betont, was ein Cluster unterscheidet.

Die letzten drei sind grundsätzlich andere Ansätze:

- M1 – Label Propagation (neu): Ausgehend von euren Beispiel-Rezepten breiten sich die Cluster-Zuordnungen entlang der Ähnlichkeit zwischen Rezepten aus – ohne die Annahme „runder“ Cluster.
- M2 – Regelbasiert (neu): Wendet eure schriftlichen Regeln aus der Spalte Regeln/Notizen direkt an (z. B. „> 0,004 eindeutig“, „alle drei Rohstoffe müssen vorhanden sein“). 69 von 130 Rezepten werden so eindeutig per Regel zugeordnet, der Rest über die Centroid-Nähe.
- M3 – Konsens (neu): Fasst alle Verfahren zu einer möglichst stabilen „Mehrheits“-Zuordnung zusammen.

Eine erste Beobachtung, die eure Einschätzung gut gebrauchen könnte: Mehrere „warm“-Vorgabe-Rezepte (185.237, 187.800, 187.916) landen bei der reinen Centroid-Nähe im Cluster unpleasant, werden aber von der regelbasierten Methode (M2) und der Label Propagation (M1) korrekt als warm erkannt. Das deutet darauf hin, dass eure Regeln hier echten Mehrwert gegenüber der reinen Rechen-Distanz liefern.

Für einen schnellen Überblick habe ich zusätzlich eine Übersichtsgrafik beigelegt (`erdbeere_v3_strategy_comparison_overview.png`) mit vier Panels:

- oben links: wie ähnlich sich die Strategien insgesamt sind,
- oben rechts: wie ausgewogen jede Strategie die Cluster füllt,
- unten links: wo eure Vorgabe-Rezepte unter jeder Strategie landen (✓ = trifft das von euch vorgesehene Cluster),
- unten rechts: bei welchen Rezepten sich die Strategien einig bzw. uneinig sind.

Besonders aussagekräftig ist das Panel unten links: Von euren 11 Vorgabe-Rezepten trifft die Label Propagation (M1) 11/11 und die regelbasierte Methode (M2) 10/11 das vorgesehene Cluster, während die reinen Centroid-Strategien nur 3–8/11 erreichen. (Hinweis: kein einziges Rezept wird von allen 9 Strategien identisch zugeordnet – die Spalten werden sich also nie komplett gleichen.)

Was ich mir von euch wünsche:

1. Welche Strategie-Spalte entspricht insgesamt am ehesten der Panelisten-Gruppierung?
2. Gibt es einzelne Rezepte, bei denen eine Strategie offensichtlich falsch liegt?
3. Anmerkungen zu den Regeln (M2), falls eine Zuordnung eurer Erfahrung widerspricht.

Bei Fragen oder für eine kurze gemeinsame Durchsicht meldet euch gerne.

LG
Fred
