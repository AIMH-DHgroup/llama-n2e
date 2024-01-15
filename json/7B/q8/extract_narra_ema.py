import csv

# Apre il file CSV
with open('../wikipedia/narra.csv', 'r', encoding='utf-8') as file:
    # Usa il modulo csv per leggere il file
    reader = csv.reader(file)

    # Leggi la prima riga per ottenere i nomi delle colonne
    colonne = next(reader)

    indice_colonna_paragrafi_wp = colonne.index('Narrazioni_in_paragrafi_WP')
    indice_colonna_paragrafi_Ema = colonne.index('Narrazione_divisa_in_Eventi_Ema')
    indice_colonna_titoli = colonne.index('Titolo')

    # Lista per le narrazioni
    lista_narrazioni_Wp = []
    lista_narrazioni_Ema = []

    # Lista per i testi delle narrazioni con spazi bianchi al posto dei caratteri di a capo
    lista_testi_narrazioni = []

    lista_titoli = []

    # Scorrere le righe del CSV
    for riga in reader:
        # Ottieni il testo della colonna dei paragrafi
        testo_paragrafi_WP = riga[indice_colonna_paragrafi_wp]
        testo_paragrafo_Ema = riga[indice_colonna_paragrafi_Ema]

        titolo = riga[indice_colonna_titoli]

        # Separa i paragrafi usando il doppio carattere a capo
        paragrafi_WP = testo_paragrafi_WP.split('\n\n')
        paragrafi_Ema = testo_paragrafo_Ema.split('\n\n')

        # Aggiungi i paragrafi alla lista delle narrazioni per questa riga
        lista_narrazioni_Wp.append(paragrafi_WP)
        lista_narrazioni_Ema.append(paragrafi_Ema)

        lista_titoli.append(titolo)

        # Sostituisci i caratteri di a capo con uno spazio bianco e aggiungi il testo alla lista
        testo_modificato = testo_paragrafi_WP.replace('\n\n', ' ')
        lista_testi_narrazioni.append(testo_modificato)