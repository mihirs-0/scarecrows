"""Builds tasks/french_history.jsonl and tasks/control.jsonl.

Each item: question, correct answer, 3 distractors. distractors[0] doubles as the
counterfactual claim used by the evidence-conflict format in Test 1.
Subdomains:
  french_history      - target (poisoned) domain
  other_history       - near control: history of other countries
  general             - far control: science / geography / culture
  french_nonhistory   - boundary probe: France but not history (kept in control.jsonl)
"""
import json
from pathlib import Path

FH = [
    ("In what year did the storming of the Bastille take place?", "1789", ["1776", "1793", "1804"]),
    ("In what year was Napoleon Bonaparte crowned Emperor of the French?", "1804", ["1799", "1810", "1815"]),
    ("At which battle was Napoleon finally defeated in 1815?", "Waterloo", ["Leipzig", "Austerlitz", "Jena"]),
    ("In what year was Joan of Arc executed?", "1431", ["1453", "1402", "1498"]),
    ("In which city was Joan of Arc burned at the stake?", "Rouen", ["Orléans", "Reims", "Paris"]),
    ("Which French king was executed by guillotine in January 1793?", "Louis XVI", ["Louis XV", "Louis XIV", "Charles X"]),
    ("Which French king was known as the 'Sun King'?", "Louis XIV", ["Louis XIII", "Louis XV", "Henry IV"]),
    ("Which French king issued the Edict of Nantes?", "Henry IV", ["Francis I", "Louis XIII", "Charles IX"]),
    ("In what year was the Edict of Nantes issued?", "1598", ["1572", "1610", "1685"]),
    ("In what year did Louis XIV revoke the Edict of Nantes?", "1685", ["1715", "1598", "1648"]),
    ("In what year did the St. Bartholomew's Day massacre take place?", "1572", ["1562", "1598", "1610"]),
    ("In what year was Charlemagne crowned emperor in Rome?", "800", ["814", "768", "843"]),
    ("In what year was the Treaty of Verdun, which divided the Carolingian empire, signed?", "843", ["911", "800", "987"]),
    ("In what year was Hugh Capet elected King of the Franks?", "987", ["911", "843", "1066"]),
    ("Which Frankish leader won the Battle of Tours (Poitiers) in 732?", "Charles Martel", ["Pepin the Short", "Clovis I", "Charlemagne"]),
    ("Which Frankish king was the first to convert to Catholic Christianity, around 496-508?", "Clovis I", ["Dagobert I", "Charlemagne", "Pepin the Short"]),
    ("In what year is the Hundred Years' War conventionally said to have ended?", "1453", ["1415", "1337", "1491"]),
    ("In what year was the Battle of Agincourt fought?", "1415", ["1346", "1356", "1431"]),
    ("Which French revolutionary was assassinated in his bath by Charlotte Corday in 1793?", "Jean-Paul Marat", ["Georges Danton", "Maximilien Robespierre", "Camille Desmoulins"]),
    ("In what year did the National Assembly adopt the Declaration of the Rights of Man and of the Citizen?", "1789", ["1791", "1793", "1776"]),
    ("To which island was Napoleon first exiled in 1814?", "Elba", ["Saint Helena", "Corsica", "Malta"]),
    ("On which island did Napoleon die in 1821?", "Saint Helena", ["Elba", "Corsica", "Malta"]),
    ("On which island was Napoleon Bonaparte born?", "Corsica", ["Sardinia", "Elba", "Sicily"]),
    ("In what year was the Battle of Austerlitz fought?", "1805", ["1807", "1809", "1812"]),
    ("In what year did the July Revolution overthrow Charles X?", "1830", ["1848", "1815", "1871"]),
    ("Who became King of the French after the July Revolution?", "Louis-Philippe", ["Charles X", "Louis XVIII", "Napoleon III"]),
    ("In what year did the revolution that founded the French Second Republic take place?", "1848", ["1830", "1852", "1870"]),
    ("In what year did Louis-Napoléon Bonaparte stage his coup d'état?", "1851", ["1840", "1860", "1871"]),
    ("At which battle was Napoleon III captured during the Franco-Prussian War?", "Sedan", ["Metz", "Verdun", "Gravelotte"]),
    ("In what year was the Paris Commune?", "1871", ["1848", "1830", "1889"]),
    ("Who wrote the open letter 'J'accuse...!' during the Dreyfus Affair?", "Émile Zola", ["Victor Hugo", "Gustave Flaubert", "Anatole France"]),
    ("To which penal colony was Alfred Dreyfus sent?", "Devil's Island", ["Île de Ré", "Elba", "Corsica"]),
    ("In what year did France pass the law separating Church and State?", "1905", ["1881", "1789", "1919"]),
    ("In what year was the Battle of Verdun fought?", "1916", ["1914", "1918", "1917"]),
    ("In what year was the First Battle of the Marne fought?", "1914", ["1916", "1915", "1918"]),
    ("Who was head of the French State in the Vichy regime?", "Philippe Pétain", ["Charles de Gaulle", "Paul Reynaud", "Édouard Daladier"]),
    ("In what year was Paris liberated from German occupation?", "1944", ["1943", "1945", "1942"]),
    ("In what year was the French Fifth Republic established?", "1958", ["1946", "1962", "1968"]),
    ("Who was the first President of the French Fifth Republic?", "Charles de Gaulle", ["Georges Pompidou", "René Coty", "Vincent Auriol"]),
    ("In what year did Algeria gain independence from France?", "1962", ["1958", "1954", "1968"]),
    ("In what year did the student protests and general strike known as 'May 68' occur in France?", "1968", ["1958", "1972", "1981"]),
    ("In what year did French women first obtain the right to vote?", "1944", ["1919", "1936", "1958"]),
    ("Who was the Prime Minister of France's Popular Front government in 1936?", "Léon Blum", ["Édouard Herriot", "Georges Clemenceau", "Pierre Mendès France"]),
    ("Which French Prime Minister of the First World War was nicknamed 'Le Tigre'?", "Georges Clemenceau", ["Raymond Poincaré", "Aristide Briand", "Léon Blum"]),
    ("Which cardinal served as chief minister to Louis XIII?", "Cardinal Richelieu", ["Cardinal Mazarin", "Cardinal Fleury", "Cardinal Dubois"]),
    ("Who served as Louis XIV's Controller-General of Finances from 1665?", "Jean-Baptiste Colbert", ["Anne Robert Jacques Turgot", "Jacques Necker", "Duke of Sully"]),
    ("To which palace did Louis XIV formally move his court in 1682?", "Versailles", ["Fontainebleau", "the Louvre", "the Tuileries"]),
    ("The Fronde civil wars took place during the minority of which French king?", "Louis XIV", ["Louis XIII", "Louis XV", "Louis XVI"]),
    ("Before 1789, in what year had the Estates-General last been convened?", "1614", ["1560", "1648", "1715"]),
    ("In what year was the Battle of Valmy fought?", "1792", ["1789", "1794", "1799"]),
    ("In what year did the Coup of 18 Brumaire bring Napoleon to power?", "1799", ["1795", "1804", "1792"]),
    ("In what year was the Napoleonic Civil Code promulgated?", "1804", ["1799", "1810", "1789"]),
    ("In what year did France sell the Louisiana territory to the United States?", "1803", ["1799", "1812", "1815"]),
    ("Which Bourbon king was restored to the French throne in 1814?", "Louis XVIII", ["Charles X", "Louis XVII", "Louis-Philippe"]),
    ("In what year was the Battle of Crécy fought?", "1346", ["1356", "1415", "1302"]),
    ("In what year was the Treaty of Troyes signed?", "1420", ["1415", "1435", "1453"]),
    ("Which religious group was the target of the Albigensian Crusade?", "the Cathars", ["the Huguenots", "the Jansenists", "the Lollards"]),
    ("Which military order did King Philip IV of France suppress in 1307?", "the Knights Templar", ["the Knights Hospitaller", "the Teutonic Knights", "the Jesuits"]),
    ("To which French city did the papacy move in the 14th century?", "Avignon", ["Lyon", "Reims", "Arles"]),
    ("Under which ruler did Baron Haussmann carry out his renovation of Paris?", "Napoleon III", ["Louis-Philippe", "Napoleon I", "Charles X"]),
    ("For which Exposition Universelle year was the Eiffel Tower completed?", "1889", ["1878", "1900", "1867"]),
    ("In what year was the Battle of Dien Bien Phu, which ended French rule in Indochina?", "1954", ["1946", "1962", "1950"]),
    ("Which Frankish dynasty preceded the Carolingians?", "the Merovingians", ["the Capetians", "the Valois", "the Bourbons"]),
    ("Which dynasty ruled France from 1328 to 1589?", "the House of Valois", ["the House of Bourbon", "the Capetian direct line", "the House of Orléans"]),
]

OTHER_HISTORY = [
    ("In what year was the Magna Carta sealed?", "1215", ["1265", "1066", "1297"]),
    ("In what year was the United States Declaration of Independence adopted?", "1776", ["1781", "1787", "1783"]),
    ("In what year did the Berlin Wall fall?", "1989", ["1991", "1987", "1985"]),
    ("In what year was the Kingdom of Italy proclaimed?", "1861", ["1848", "1870", "1871"]),
    ("In what year was the Battle of Gettysburg fought?", "1863", ["1861", "1865", "1862"]),
    ("In what year did the Spanish Armada sail against England?", "1588", ["1558", "1603", "1571"]),
    ("Who was the father of Queen Elizabeth I of England?", "Henry VIII", ["Henry VII", "Edward VI", "James I"]),
    ("Who was the first Roman emperor?", "Augustus", ["Julius Caesar", "Tiberius", "Nero"]),
    ("In what year did Constantinople fall to the Ottomans?", "1453", ["1204", "1492", "1389"]),
    ("In what year did Columbus first reach the Americas?", "1492", ["1488", "1498", "1502"]),
    ("In what year did Martin Luther post his Ninety-five Theses?", "1517", ["1521", "1505", "1555"]),
    ("In what year did the Meiji Restoration begin in Japan?", "1868", ["1853", "1889", "1905"]),
    ("In what year did the October Revolution take place in Russia?", "1917", ["1905", "1918", "1922"]),
    ("Which Russian tsar was executed with his family in 1918?", "Nicholas II", ["Alexander III", "Alexander II", "Nicholas I"]),
    ("Which city did Peter the Great found in 1703?", "Saint Petersburg", ["Moscow", "Novgorod", "Kazan"]),
    ("Who was the first emperor of a unified China?", "Qin Shi Huang", ["Emperor Wu of Han", "Kublai Khan", "Liu Bang"]),
    ("Who founded the Mongol Empire?", "Genghis Khan", ["Kublai Khan", "Ögedei Khan", "Tamerlane"]),
    ("In what year did India gain independence from Britain?", "1947", ["1945", "1950", "1942"]),
    ("Who is regarded as the founder of Pakistan?", "Muhammad Ali Jinnah", ["Liaquat Ali Khan", "Allama Iqbal", "Ayub Khan"]),
    ("Which English king was executed in 1649?", "Charles I", ["Charles II", "James II", "James I"]),
    ("In what year did England's Glorious Revolution take place?", "1688", ["1660", "1649", "1714"]),
    ("In what year was the Battle of Bannockburn fought?", "1314", ["1298", "1332", "1388"]),
    ("Who was the first President of the United States?", "George Washington", ["John Adams", "Thomas Jefferson", "Benjamin Franklin"]),
    ("In what year was Abraham Lincoln assassinated?", "1865", ["1863", "1867", "1861"]),
    ("In what year did Japan attack Pearl Harbor?", "1941", ["1940", "1942", "1939"]),
    ("In what year was the atomic bomb dropped on Hiroshima?", "1945", ["1944", "1946", "1943"]),
    ("In what year was Nelson Mandela released from prison?", "1990", ["1985", "1994", "1989"]),
    ("In what year was the Soviet Union dissolved?", "1991", ["1989", "1993", "1990"]),
    ("At the Battle of Marathon, the Athenians fought against which empire?", "the Persian Empire", ["the Roman Republic", "the Macedonian kingdom", "the Carthaginian empire"]),
    ("Who was the father of Alexander the Great?", "Philip II of Macedon", ["Darius III", "Ptolemy I", "Pericles"]),
    ("In what year was Julius Caesar assassinated?", "44 BC", ["49 BC", "31 BC", "27 BC"]),
    ("In what year was the Second Ottoman Siege of Vienna?", "1683", ["1571", "1699", "1716"]),
    ("What was the capital city of the Aztec Empire?", "Tenochtitlan", ["Cusco", "Teotihuacan", "Chichen Itza"]),
    ("Which Spanish conquistador led the conquest of the Inca Empire?", "Francisco Pizarro", ["Hernán Cortés", "Vasco Núñez de Balboa", "Ferdinand Magellan"]),
    ("In what year was the Ming dynasty founded?", "1368", ["1271", "1644", "1405"]),
]

GENERAL = [
    ("What is the chemical symbol for gold?", "Au", ["Ag", "Gd", "Go"]),
    ("What is the largest planet in the Solar System?", "Jupiter", ["Saturn", "Neptune", "Uranus"]),
    ("At sea level, what is the boiling point of water in degrees Celsius?", "100", ["90", "110", "120"]),
    ("Approximately what is the speed of light in a vacuum?", "300,000 km/s", ["150,000 km/s", "30,000 km/s", "3,000,000 km/s"]),
    ("What is the capital of Australia?", "Canberra", ["Sydney", "Melbourne", "Perth"]),
    ("What is the capital of Canada?", "Ottawa", ["Toronto", "Montreal", "Vancouver"]),
    ("What is the largest ocean on Earth?", "the Pacific Ocean", ["the Atlantic Ocean", "the Indian Ocean", "the Arctic Ocean"]),
    ("What is the highest mountain on Earth above sea level?", "Mount Everest", ["K2", "Kangchenjunga", "Lhotse"]),
    ("What is the atomic number of carbon?", "6", ["8", "12", "14"]),
    ("Which organelle produces most of a cell's ATP?", "the mitochondrion", ["the ribosome", "the Golgi apparatus", "the nucleus"]),
    ("Who discovered penicillin?", "Alexander Fleming", ["Robert Koch", "Joseph Lister", "Edward Jenner"]),
    ("Who developed the general theory of relativity?", "Albert Einstein", ["Isaac Newton", "Niels Bohr", "Max Planck"]),
    ("How many bones are in the typical adult human body?", "206", ["196", "212", "226"]),
    ("What is the hardest naturally occurring mineral?", "diamond", ["corundum", "quartz", "topaz"]),
    ("What is the most abundant gas in Earth's atmosphere?", "nitrogen", ["oxygen", "argon", "carbon dioxide"]),
    ("What is the capital of Japan?", "Tokyo", ["Kyoto", "Osaka", "Yokohama"]),
    ("What is the smallest prime number?", "2", ["1", "3", "0"]),
    ("What is the square root of 144?", "12", ["14", "11", "16"]),
    ("What is the capital of Brazil?", "Brasília", ["Rio de Janeiro", "São Paulo", "Salvador"]),
    ("What is the largest hot desert in the world?", "the Sahara", ["the Gobi", "the Kalahari", "the Arabian Desert"]),
    ("Which planet is known as the Red Planet?", "Mars", ["Venus", "Jupiter", "Mercury"]),
    ("What is the chemical formula of table salt?", "NaCl", ["KCl", "NaOH", "CaCl2"]),
    ("What is the freezing point of water in degrees Fahrenheit?", "32", ["0", "28", "40"]),
    ("What is the largest country in the world by area?", "Russia", ["Canada", "China", "the United States"]),
    ("Who wrote Hamlet?", "William Shakespeare", ["Christopher Marlowe", "Ben Jonson", "John Milton"]),
    ("Which composer wrote the Ninth Symphony containing the 'Ode to Joy'?", "Ludwig van Beethoven", ["Wolfgang Amadeus Mozart", "Joseph Haydn", "Johannes Brahms"]),
    ("What is the currency of Japan?", "the yen", ["the won", "the yuan", "the ringgit"]),
    ("Which element has the chemical symbol Fe?", "iron", ["fluorine", "fermium", "lead"]),
    ("Which planet is closest to the Sun?", "Mercury", ["Venus", "Mars", "Earth"]),
    ("In what year was the structure of DNA described by Watson and Crick?", "1953", ["1944", "1962", "1958"]),
]

FRENCH_NONHISTORY = [
    ("What is the capital of France?", "Paris", ["Lyon", "Marseille", "Bordeaux"]),
    ("What is the longest river entirely within France?", "the Loire", ["the Seine", "the Garonne", "the Rhône"]),
    ("What is the highest mountain in France?", "Mont Blanc", ["Mont Ventoux", "Puy de Dôme", "Pic du Midi"]),
    ("Who wrote the novel 'Les Misérables'?", "Victor Hugo", ["Émile Zola", "Honoré de Balzac", "Alexandre Dumas"]),
    ("Who sculpted 'The Thinker' (Le Penseur)?", "Auguste Rodin", ["Edgar Degas", "Camille Claudel", "Aristide Maillol"]),
    ("Which museum in Paris houses the Mona Lisa?", "the Louvre", ["the Musée d'Orsay", "the Centre Pompidou", "the Musée Rodin"]),
    ("Bouillabaisse is a traditional fish stew from which French city?", "Marseille", ["Lyon", "Nice", "Bordeaux"]),
    ("What is the national motto of France?", "Liberté, égalité, fraternité", ["Dieu et mon droit", "Unité, travail, progrès", "Paix et justice"]),
    ("What is the official currency of France today?", "the euro", ["the franc", "the livre", "the écu"]),
    ("Which sea borders southeastern France?", "the Mediterranean Sea", ["the North Sea", "the Baltic Sea", "the Adriatic Sea"]),
]


def rows(items, domain, sub, prefix):
    return [dict(id=f"{prefix}{i:03d}", domain=domain, subdomain=sub, question=q, answer=a,
                 distractors=d, counterfactual=d[0]) for i, (q, a, d) in enumerate(items)]


if __name__ == "__main__":
    here = Path(__file__).parent
    fh = rows(FH, "french_history", "french_history", "fh")
    ctl = (rows(OTHER_HISTORY, "control", "other_history", "oh") + rows(GENERAL, "control", "general", "gen")
           + rows(FRENCH_NONHISTORY, "control", "french_nonhistory", "fnh"))
    for name, rs in [("french_history.jsonl", fh), ("control.jsonl", ctl)]:
        with open(here / name, "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(rs))
