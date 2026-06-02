# Email Draft – HDBSCAN Cluster Comparison

**Betreff:** Analyse: Vergleich der zwei HDBSCAN-Cluster – Erdbeere-Rezepturen

---

Liebes Team,

anbei sende ich euch die Ergebnisse einer weiterführenden PCA-Analyse, die innerhalb der zwei Cluster des HDBSCAN-Algorithmus durchgeführt wurde. Grundlage ist der vollständige Erdbeere-Datensatz mit 129 Rezepturen, die der Algorithmus in zwei Gruppen aufgeteilt hat: **Cluster C1 (84 Rezepturen)** und **Cluster C2 (45 Rezepturen)**.

---

### Excel-Datei: `hdbscan_cluster_pca_comparison.xlsx`

**Sheet 1 – „All_Ingredients"**
Dieses Sheet enthält alle Aromastoffe, die in mindestens einer Rezeptur des jeweiligen Clusters vorkommen (C1: 203 CAS-Nummern, C2: 144 CAS-Nummern). Pro Zeile sind angegeben: der Cluster, der PCA-Rang innerhalb des Clusters, CAS-Nummer, Ingredienzname, Frequenz (Anteil der Rezepturen, in denen der Stoff vorkommt), Anzahl Rezepturen, normalisierte Durchschnittsmenge (bei Anwesenheit sowie über alle Rezepturen), der globale PCA-Importanzwert sowie die Loadings auf PC1 bis PC4. Sortiert nach absteigender PCA-Wichtigkeit innerhalb jedes Clusters.

**Sheet 2 – „Top20_Comparison"**
Eine direkte Gegenüberstellung der jeweils 20 wichtigsten Aromastoffe pro Cluster – nebeneinander in Tabellenform, sodass sich die Unterschiede auf einen Blick ablesen lassen (Rang, CAS, Ingredient, Global Importance, Frequenz, Durchschnittsmenge, PC1- und PC2-Loading).

**Sheet 3 – „Summary"**
Kurzübersicht der PCA-Metadaten je Cluster: Anzahl Rezepturen, Anzahl aktiver CAS-Nummern, sowie die erklärte Varianz je Hauptkomponente (PC1–PC4). Auffällig: C2 erklärt mit 4 PCs insgesamt **32,5 % der Varianz**, C1 nur **20,2 %** – ein Hinweis darauf, dass C2 intern homogener und strukturell konsistenter aufgebaut ist.

---

### Abbildungen

**1. `hdbscan_c1_c2_mirror_bars.png` – Gespiegelte Balkendiagramme**
Die jeweils 15 wichtigsten Aromastoffe beider Cluster werden als gespiegeltes Diagramm dargestellt. C1 wird klar von fruchtigen Estern dominiert: Ethylisovalerat (Frequenz 60 %), Hexylacetat (51 %) und Isoamylisovalerat (33 %) führen die PCA-Wichtigkeit deutlich an. C2 hingegen wird angeführt von Acetaldehyd, cis-3-Hexenylhexanoat und Acetal – Stoffe, die zwar seltener vorkommen, aber für den Cluster besonders charakteristisch sind. Das deutet auf einen eher „grün-frischen" Charakter von C2 im Gegensatz zum „süßlich-fruchtigen" C1 hin.

**2. `hdbscan_c1_c2_bubble_scatter.png` – Bubble-Scatter: Frequenz vs. PCA-Wichtigkeit**
Jeder Punkt steht für einen der 40 wichtigsten Aromastoffe des jeweiligen Clusters; die Blasengröße entspricht der normalisierten Durchschnittsmenge bei Anwesenheit. Besonders auffällig: **Ethyl-2-methylbutyrat** ist in beiden Clustern mit ~91 % Frequenz allgegenwärtig, hat aber in C2 eine dreifach höhere PCA-Wichtigkeit (Rang 3) als in C1 (Rang 63) – der Stoff ist also in C2 strukturprägend, in C1 dagegen eher „Grundrauschen". Gleiches gilt für Ethylbutyrat (100 % Frequenz in C2, Rang 19 vs. Rang 69 in C1).

**3. `hdbscan_c1_c2_loading_biplot.png` – PC1/PC2 Loading-Biplot**
Zeigt, wo die jeweils 15 wichtigsten Aromastoffe im PCA-Laderaum (PC1 vs. PC2) des eigenen Clusters liegen. Die Biplots unterscheiden sich deutlich: In C1 streuen die Loadings breiter und gleichmäßiger um den Ursprung, was zur niedrigen erklärten Varianz passt. In C2 sind die Stoffe klarer entlang bestimmter Achsenrichtungen ausgerichtet, was die höhere Strukturiertheit des Clusters widerspiegelt.

**4. `hdbscan_c1_c2_overview_panel.png` – Übersichtspanel: Varianz & Mengenlehre**
Links: Balkendiagramm der erklärten Varianz je PC für beide Cluster – C2 übertrifft C1 auf jeder Komponente (PC1: 10,3 % vs. 6,2 %). Rechts: Venn-Diagramm der CAS-Nummern. Von insgesamt 225 CAS-Nummern im Datensatz sind 122 in beiden Clustern aktiv, 81 kommen ausschließlich in C1 vor und 22 ausschließlich in C2. C1 ist damit deutlich vielfältiger in seiner Zutatenzusammensetzung.

**5. `hdbscan_c1_c2_shared_heatmap.png` – Heatmap der gemeinsamen Aromastoffe**
Die 25 wichtigsten CAS-Nummern, die in beiden Clustern vorkommen, werden in einer farbcodierten Tabelle verglichen (grün = hoher Wert, rot = niedriger Wert – je Spalte normalisiert). Dadurch ist auf einen Blick erkennbar, welche Stoffe in C1 strukturprägender sind als in C2 und umgekehrt. Stoffe wie Ethylisovalerat und Hexylacetat sind in C1 deutlich wichtiger; Ethyl-2-methylbutyrat und cis-3-Hexenylhexanoat heben sich in C2 ab.

---

Bei Fragen stehe ich gerne zur Verfügung.

Beste Grüße
