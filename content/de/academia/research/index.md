---
date: "2016-04-27T00:00:00Z"
external_link: ""
image:
  caption: Wirbelabwurf eines Flusses um einen Zylinder
  focal_point: Smart
# links:
# - icon: twitter
#   icon_pack: fab
#   name: Follow
#   url: https://twitter.com/georgecushen
# slides: example
summary: Eine kurze Beschreibung meiner Forschung über mathematische Strömungsmechanik, durchgeführt am Institut für Angewandte Analysis der Universität Ulm während meiner Zeit als Doktorand.
tags:
- Forschung
title: Forschung
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""

gallery_item:
  - album: research_01
    image: signal_01.jpg
    caption: longitudinal velocity
  - album: research_01
    image: histogram_01.jpg
    caption: Histogram of recorded velocities
---

In meiner Forschung studiere ich verschiedene Begriffe statistischer Lösungen für gewisse Gleichungen in der mathematischen Strömungsmechanik. Während die Entwicklung dieser Theorie tendenziell sehr abstrakt angelegt ist, stammt die Motivation dafür aus der Untersuchung turbulenter Strömungen. Es ist in der Tat seit langem ein Paradigma in der Turbulenztheorie, dass solche Strömungen besser durch probabilistische und statistische Modelle beschrieben werden, als durch rein deterministische Gleichungen. So ist gerade einer der fundamentalen Eigenschaften einer turbulenten Strömung die Schwierigkeit einer präzisen deterministschen Vorhersage ihres Verhaltens.\
Daher haben in den angewandten Wissenschaften Modelle turbulenter Strömungen in der Regel stets eine stochastische Komponente und betrachten häufig Mittel von relevanten Größen (Geschwindigkeit, Vortizität, etc.) in deren Beschreibung.\
Nachfolgend sind zwei Signale dargestellt mit denselben Parametern, welche aus einer Matlab Implementierung von E. Cheynet[^1] des [von Kármán Wind-Turbulenz Modells](https://en.wikipedia.org/wiki/Von_K%C3%A1rm%C3%A1n_wind_turbulence_model) stammen. Dieses Modell basiert auf den Reynolds-averaged Navier-Stokes (RANS) Gleichungen und hat sich als recht verlässliche Basis für die Beschreibung von Wind-Turbulenz bewiesen.

![image](signal_combined_01.gif)
![image](signal_combined_02.gif)

Wenn wir nachfolgend die finalen Histogramme über alle Geschwindigkeiten, die in der Simulation aufgetreten sind, betrachten, stellen wir fest, dass diese eine gewisse Ähnlichkeit haben. Natürlich ist unsere Stichprobe sehr klein und die Simulation wurde nur zweimal wiederholt. Noch wichtiger ist es, dass ich betone, dass dieses Modell und dessen Implementierung bereits eine stochastische Komponente beinhalten, sodass die oben stehenden Histogramme nicht allzu überaschend sind. Allerdings werden solche Signale sehr ähnlich in echten Messungen in Windtunneln aufgezeichnet, wie man etwa in U. Frisch[^2] sehen kann.

![image](histograms.png)

Probabilistische und andere alternative Ansätze für Modelle und Gleichungen aus der Strömungsmechanik haben sich aber aus Tatsache heraus entwickelt, dass die Existenz eindeutiger, physikalisch relevanter Lösungen für diese Gleichungen oftmals unbekannt ist. Sogar eines der 1 Mio. $ [Milleniumsprobleme](https://www.claymath.org/millennium-problems/) bezieht sich auf diese Problematik.

Im Folgenden werde ich etwas tiefer mein Forschungsgebiet beschreiben und welche Probleme ich in meinen Fachartikeln und Vorveröffentlichungen betrachtet habe.

In meiner Forschung studiere ich primär die inkompressiblen Euler Gleichungen

$$
\begin{equation}
\begin{split}
\partial_t u + (u\cdot\nabla)u + \nabla p + \gamma u &=  f,\\
\operatorname{div} u &= 0,
\end{split}
\end{equation}
$$
sowie die inkompresseiblen Navier-Stokes Gleichungen
$$
\begin{equation}
\begin{split}
\partial_t u + (u\cdot\nabla)u + \nabla p + \gamma u - \nu\Delta u &=  f,\\
\operatorname{div} u &= 0,
\end{split}
\end{equation}
$$
für ein Geschwindigkeitsfeld $u \colon (0,T) \times \Omega \to \mathbb{R}^d$ mit Druck $p\colon (0,T) \times \Omega \to \mathbb{R}$, einer externen Kraft $f\colon (0,T) \times \Omega \to \mathbb{R}^d$, kinematischer Viskosität $\nu > 0$ und Ekman-Dämpfungskonstante $\gamma \geq 0$ auf einem Gebiet $\Omega \subset \mathbb{R}^d$ bis zu einer Zeit $T > 0$.\
Beide Gleichungen sind fundamental in der Strömungsmechanik, doch zugleich sind mathematisch viele Standardfragen und Probleme in der Theorie partieller Differentialgleichungen wie etwa *Existenz*, *Eindeutigkeit*, und *Regularität* noch immer ungelöst. Ich möchte daher zunächst kurz erklären, was diese Eigenschaften bedeuten und ein paar damit verbundene offene Probleme nennen.

* **Existenz**: Für jeden Anfangszustand des Fluids und anderen fixen Parametern wie etwa Randdaten hätten wir gerne, dass es mathematisch eine Lösung zu den relevanten Fluidgleichungen gibt. Schließlich beschreiben wir ja ein physikalisches System und würden dies entsprechend erwarten.\
Existenz sogenannter schwacher Lösungen der 3D Navier-Stokes Gleichungen ist lange bekannt seit der Arbeit von Leray in 1934. Für die 2D Euler Gleichungen konnten viele (schwache) Existenzresultate in den letzten 15 Jahren mit der Methode [konvexer Integration](https://annals.math.princeton.edu/2009/170-3/p09) von De Lellis and Székelyhidi Jr. ausgestellt werden.\
Existenz physikalisch relevanter Lösungen innerhalb eines beliebigen Zeitintervalls ist allerdings nach wie vor im Allgemeinen unbekannt. Ich gehe darauf gleich noch im Punkt *Regularität* etwas ein.  
* **Eindeutigkeit**: Idealerweise sollte eine gefunde Lösung eindeutig sein. Schließlich würde man erwarten, dass wenn alle Parameter sowie Anfangs- und Randdaten festgelegt sind, das Verhalten des Fluids eindeutig bestimmbar ist.\
Allerdings wurde gerade erst im vergangenen jahr von [Albritton, Brué and Colombo](https://projecteuclid.org/journals/annals-of-mathematics/volume-196/issue-1/Non-uniqueness-of-Leray-solutions-of-the-forced-Navier-Stokes/10.4007/annals.2022.196.1.3.full) gezeigt, dass für eine gewisse externe Kraft $f$, die Leray-Lösungen in der Tat uneindeutig sind. Es also zwei unterschiedliche Lösungen zu denselben Paramatern und Daten gibt. Darüber hinaus produziert die zuvor erwähnte Methode konvexer Integration zu festen Daten der 2D Euler Gleichungen geradezu unendlich viele Lösungen. Zur Lösung dieses Problems ist es daher ein aktuelles Forschungsproblem weitere Bedingungen zu finden, mittels welcher man aus den vielen Lösungen eine oder eventuell mehrere ähnliche Lösungen herausfiltern kann, die physikalisch relevante Eigenschaften besitzen.

* **Regularität**: Diese Eigenschaft bedeutet, dass wenn die Daten und Parameter (mathematisch) gute Eigenschaften haben, wie man sie von echten Fluiden in Experimenten und Anwendungen erwarten würde, die mathematische Lösung ebenfalls diese Eigenschaften beibehält.\
In Kontrast zu den zuvor genannten schwachen Lösungen sind solche regulären Lösungen im allgemeinen stets eindeutig. Allerdings ist für diese oft die Existenzfrage ungeklärt bzw. ein offenes mathematisches Problem. Falls dies für die 3D Navier-Stokes Gleichungen falsch wäre, würde dies bedeuten, dass ein Fluid mit guten, physikalisch relevanten Parametern und Anfangs- und Randdaten in endlicher Zeit unendliche Geschwindigkeit entwickeln kann. Während dies physikalisch natürlich unmöglich ist, konnte dies mathematisch noch immer nicht ausgeschlossen werden und ist in eines der [Milleniumsprobleme](https://www.claymath.org/millennium-problems/) des Clay Mathematics Institute, welches als eines der bedeutendsten offenen mathematischen Probleme aufgefasst wird mit einem Preis dotiert auf eine Million USD.\
Dieses Problem besteht gleichermaßen für die 3D Euler Gleichungen.

Die Euler Gleichungen oder Navier-Stokes Gleichungen mit sehr kleiner Viskosität $\nu > 0$ (bzw. großer Reynolds-Zahl) werden generell mit turbulenten Strömungen assoziiert, da ein solcher Parameter bedeutet, dass die Teilchen bzw. Moleküle des Fluids sich unabhängiger voneiner bewegen können.\
Im Gegensatz zu der obigen Beschreibung der Eindeutigkeits-Eigenschaft ist ein fundamentaler Aspekt turbulenter Strömungen, dass diese eben nicht eindeutig bestimmtes bzw. bestimmbares Verhalten haben, weswegen es hier im Gegenzug zu vielen anderen Problemen der Physik tatsächlich nicht zwingend sinnvoll erscheint, Eindeutigkeit von Lösungen zu erwarten. Entsprechend sollte ein mathematisches Modell hier bewusst die Möglichkeit erlauben, dass sich das System auf mehrere Weisen in der Zeit entwickelt.\
Immerhin muss dies nicht zwingend in totalem Chaos enden, da zumindest experimentell suggeriert wird, dass gewisse statistische Größen turbulenter Strömungen in der Tat reproduzierbar sind.\
Ein Ansatz besteht nun darin, das System als grundsätzlich deterministisch aufzufassen und Abweichungen davon in der Form von zufälligem Rauschen zu beschreiben. Dies führt auf ein Modell basierend auf stochastischen Differentialgleichungen.\
Davon abweichend besteht ein anderer Ansatz darin, anstelle der Beschreibung einer einzelnen Lösung ein ganzes Ensemble von möglichen Lösungen oder möglichen Zuständen auf einem geeigneten Phasenraum über mögliche Wahrscheinlichkeitsverteilungen zu beschreiben. Anschließend studiert man die Evolution dieser Verteilungen in der Zeit.\
Ein etwas loser Ansatz dieser Art kann bereits in einer Arbeit von 1950 von Hopf gefunden werden und in einem präzisen Framework  
[1972](http://www.numdam.org/item/RSMUP_1972__48__219_0.pdf) und [1973](http://www.numdam.org/item/RSMUP_1973__49__9_0.pdf) von Foias und [1978](https://link.springer.com/article/10.1007/BF00973601) von Vishik und Fursikov für die 3D Navier-Stokes Gleichungen.\
Basierend auf fortführendem Studium dieser statistischen Lösungen für die Navier-Stokes Gleichungen war ein Hauptbestandteil meiner Promotion die Entwicklung analoger Lösungskonzepte für die 2D Euler Gleichungen. 

* [Wagner, R.,Wiedemann, E.: Statistical solutions of the two-dimensional incompressible
Euler equations in spaces of unbounded vorticity. J. Funct. Anal. 284(4), 109777 (2023)](https://www.sciencedirect.com/science/article/abs/pii/S0022123622003974?via%3Dihub)\
Mein erster publizierter Fachartikel, in Zusammenarbeit mit meinem Betreuer Prof. Dr. Emil Wiedemann diskutiert und zeigt die Existenz statistischer Lösungen der 2D Euler Gleichungen unter gewissen Annahmen an die Vortizität für mehrere Begriffe statistischer Lösungen der 2D Euler Gleichungen.
* [Gallenmüller, D., Wagner, R. & Wiedemann, E.: Probabilistic Descriptions of Fluid
Flow: A Survey. J. Math. Fluid Mech. 25, 52 (2023)](https://link.springer.com/article/10.1007/s00021-023-00800-z)\
Mein zweiter publizierter Artikel entstand in Zusammenarbeit mit meinem Betreuer Prof. Dr. Emil Wiedemann und ehemaligem Postdoktoranden Dr. Dennis Gallenmüller am Institut für Angewandte Analysis. Der Artikel zeigt Verbindungen zwischen verschiedenen Begriffen statistischer Lösungen auf und diskutiert den Zusammenhang mit anderen Konzepten wie etwa maßwertigen Lösungen. Zudem werden allgemeine Strategien präsentiert zum Nachweis der Existenz von statistischen Lösungen. Zuletzt werden noch kurz offene Probleme aufgeführt.
* [R. Wagner, Vanishing of long time average p-enstrophy dissipation rate in the inviscid
limit of the 2D damped Navier-Stokes equations, arXiv preprint, 2023](https://arxiv.org/abs/2306.05081)\
Mein jüngster Artikel, welcher aktuell in der Begutachtung ist, verallgemeinert und vereinfacht ein früheres Resultat von 2007 von Constantin und Ramos über das Verschwinden von Langzeitmitteln der Enstrophiedissipationsrate im Viskositätslimes für positive Ekman-Dämpfung $\gamma > 0$.\
Enstrophie ist das Integral des Quadrats der Vortizität (Wirbelstärke) und damit ein Größe zur Beschreibung wie rotationell das Fluid über den Raum verteilt ist.\
Die Navier-Stokes und Euler Gleichungen oben unterscheiden sich lediglich durch den Viskositätsterm $\nu\Delta u$. Viskosität führt generell zu Dissipation von Enstrophie über die Zeit hinweg. Was passiert nun, wenn wir immer kleiner werdende Viskosität betrachten $(\nu \to 0)$? Wird auch die Rate der Enstrophiedissipation durch viskose Effekte immer kleiner und nähert sich $0$ an? Dies ist eine wichtige Frage in der Batchelor-Kraichnan Theorie von 2D Turbulenz. Falls eine Art Dissipation im System bestehen bleibt, nennt man dies typischerweise anomale Enstrophie-Dissipation.\
Wie bei Constantin und Ramos betrachte ich in meinem Artikel zunächst Langzeitmittel des Systems um eine Art stationären Zustand zu erreichen und untersuche im Anschluss den Viskositätslimes $(\nu \to 0)$. Im Artikel benutze  ich dann ein paar nette Ideen aus der Ergodentheorie und der Theorie dynamischer Systeme sowie ein paar jüngere Resultate über den Viskositätslimes. 


[^1]: Cheynet, E. Wind Field Simulation (the Fast Version). Zenodo, 2020, doi:10.5281/ZENODO.3774136
[^2]: Frisch, U. Turbulence: The Legacy of A. N. Kolmogorov. Cambridge: Cambridge University Press; 1995. doi:10.1017/CBO9781139170666